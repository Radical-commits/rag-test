"""Vector storage and retrieval using ChromaDB."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import numpy as np
import chromadb
from chromadb.config import Settings
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document chunks with embeddings"}
        )
        logger.info(f"Using collection '{collection_name}' with {self.collection.count()} existing documents")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 5000
    ) -> None:
        """
        Add documents to the vector store in batches.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs (generated if not provided)
            batch_size: Maximum number of documents to add per batch (default: 5000)

        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Input lists must have same length: "
                f"texts={len(texts)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if not texts:
            logger.warning("No documents to add")
            return

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        # Ensure metadata values are serializable (ChromaDB requirement)
        processed_metadatas = []
        for metadata in metadatas:
            processed = {}
            for key, value in metadata.items():
                # Convert complex types to strings
                if isinstance(value, (dict, list)):
                    processed[key] = str(value)
                elif isinstance(value, (int, float, str, bool)):
                    processed[key] = value
                else:
                    processed[key] = str(value)
            processed_metadatas.append(processed)

        try:
            total_docs = len(texts)
            logger.info(f"Adding {total_docs} documents to vector store")

            # If total documents exceed batch size, split into batches
            if total_docs > batch_size:
                logger.info(f"Splitting into batches of {batch_size} (ChromaDB batch size limit)")

                for i in range(0, total_docs, batch_size):
                    end_idx = min(i + batch_size, total_docs)
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_docs + batch_size - 1) // batch_size

                    logger.info(f"Adding batch {batch_num}/{total_batches} ({end_idx - i} documents)")

                    self.collection.add(
                        documents=texts[i:end_idx],
                        embeddings=embeddings[i:end_idx],
                        metadatas=processed_metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )

                    logger.info(f"Batch {batch_num}/{total_batches} completed")
            else:
                # Small batch, add all at once
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=processed_metadatas,
                    ids=ids
                )

            logger.info(f"Successfully added {total_docs} documents. Total count: {self.collection.count()}")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _filter_near_duplicates(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        ids: List[str],
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Filter out near-duplicate documents from search results.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            distances: List of distance scores
            ids: List of document IDs
            similarity_threshold: Threshold for considering documents as duplicates (0-1)

        Returns:
            Filtered results dictionary with unique documents
        """
        if not documents:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }

        # Keep track of unique documents
        unique_docs = []
        unique_metadatas = []
        unique_distances = []
        unique_ids = []

        for i, doc in enumerate(documents):
            is_duplicate = False

            # Compare with already selected unique documents
            for unique_doc in unique_docs:
                similarity = SequenceMatcher(None, doc, unique_doc).ratio()
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Filtered duplicate chunk (similarity={similarity:.2f}): "
                        f"'{doc[:100]}...'"
                    )
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                unique_metadatas.append(metadatas[i])
                unique_distances.append(distances[i])
                unique_ids.append(ids[i])

        filtered_count = len(documents) - len(unique_docs)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} near-duplicate chunks from {len(documents)} results")

        return {
            "documents": unique_docs,
            "metadatas": unique_metadatas,
            "distances": unique_distances,
            "ids": unique_ids
        }

    def _apply_mmr(
        self,
        query_embedding: List[float],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        ids: List[str],
        lambda_param: float = 0.5,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Apply Maximal Marginal Relevance (MMR) to diversify search results.

        MMR balances relevance (similarity to query) and diversity (dissimilarity to already selected docs).

        Args:
            query_embedding: Query vector
            documents: List of document texts
            embeddings: List of document embeddings
            metadatas: List of metadata dictionaries
            distances: List of distance scores from original search
            ids: List of document IDs
            lambda_param: Balance between relevance (1.0) and diversity (0.0), default 0.5
            top_k: Number of results to return (default: all)

        Returns:
            Reranked results dictionary
        """
        if not documents:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }

        if top_k is None:
            top_k = len(documents)

        # Convert to numpy arrays for vectorized operations
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(embeddings)

        # Normalize vectors for cosine similarity
        query_vec = query_vec / np.linalg.norm(query_vec)
        doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

        # Calculate relevance scores (1 - distance, higher is better)
        relevance_scores = 1 - np.array(distances)

        # Track selected documents and remaining candidates
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Iteratively select documents
        for _ in range(min(top_k, len(documents))):
            if not remaining_indices:
                break

            mmr_scores = []
            for idx in remaining_indices:
                # Relevance component (similarity to query)
                relevance = relevance_scores[idx]

                # Diversity component (maximum similarity to already selected docs)
                if selected_indices:
                    # Calculate similarity to all selected documents
                    selected_vecs = doc_vecs[selected_indices]
                    similarities = np.dot(selected_vecs, doc_vecs[idx])
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0

                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Reorder results based on MMR selection
        reranked_docs = [documents[i] for i in selected_indices]
        reranked_metadatas = [metadatas[i] for i in selected_indices]
        reranked_distances = [distances[i] for i in selected_indices]
        reranked_ids = [ids[i] for i in selected_indices]

        logger.info(f"Applied MMR reranking with λ={lambda_param} to {len(documents)} documents")

        return {
            "documents": reranked_docs,
            "metadatas": reranked_metadatas,
            "distances": reranked_distances,
            "ids": reranked_ids
        }

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.8,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search for similar documents using a query embedding.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "document.pdf"})
            deduplicate: If True, filter out near-duplicate chunks (default: True)
            similarity_threshold: Threshold for duplicate detection (default: 0.8)
            use_mmr: If True, apply MMR (Maximal Marginal Relevance) for diverse results
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance, default: 0.5)

        Returns:
            Dictionary containing:
                - documents: List of matching text chunks
                - metadatas: List of metadata dictionaries
                - distances: List of distance scores (lower = more similar)
                - ids: List of document IDs

        Raises:
            ValueError: If query_embedding is invalid
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        try:
            logger.info(f"Searching for {n_results} similar documents (MMR={use_mmr}, dedup={deduplicate})")

            # Determine if we need embeddings (for MMR)
            include_fields = ["documents", "metadatas", "distances"]
            if use_mmr:
                include_fields.append("embeddings")

            # Perform similarity search
            # Fetch more results if using MMR (to have candidates for reranking)
            fetch_count = n_results * 3 if use_mmr else n_results

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_count,
                where=filter_metadata,
                include=include_fields
            )

            # ChromaDB returns nested lists, flatten them
            search_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }

            # Extract embeddings if needed for MMR
            embeddings = None
            if use_mmr and results.get("embeddings"):
                embeddings = results["embeddings"][0]

            # Apply MMR first (if enabled) - operates on larger candidate set
            if use_mmr and embeddings is not None and len(embeddings) > 0 and search_results["documents"]:
                search_results = self._apply_mmr(
                    query_embedding=query_embedding,
                    documents=search_results["documents"],
                    embeddings=embeddings,
                    metadatas=search_results["metadatas"],
                    distances=search_results["distances"],
                    ids=search_results["ids"],
                    lambda_param=mmr_lambda,
                    top_k=n_results
                )

            # Apply deduplication after MMR (if enabled)
            if deduplicate and search_results["documents"]:
                search_results = self._filter_near_duplicates(
                    documents=search_results["documents"],
                    metadatas=search_results["metadatas"],
                    distances=search_results["distances"],
                    ids=search_results["ids"],
                    similarity_threshold=similarity_threshold
                )

            logger.info(f"Returning {len(search_results['documents'])} matching documents")
            return search_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory)
        }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.

        Warning:
            This operation cannot be undone.
        """
        logger.warning(f"Clearing all documents from collection '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Document chunks with embeddings"}
        )
        logger.info("Collection cleared successfully")

    def get_filtering_stats(self) -> Dict[str, Any]:
        """
        Retrieve chunk filtering statistics from collection metadata.

        Returns:
            Dictionary with filtering metrics, or default zeros if not found
        """
        metadata = self.collection.metadata or {}

        # Stats are stored as JSON string in metadata (ChromaDB doesn't support nested dicts)
        stats_json = metadata.get("filtering_stats_json", None)

        if stats_json:
            try:
                stats = json.loads(stats_json)
            except (json.JSONDecodeError, TypeError):
                # Fallback to default if JSON is invalid
                stats = {
                    "total_chunks_created": 0,
                    "chunks_filtered": 0,
                    "filter_reasons": {}
                }
        else:
            # Default stats for new collections
            stats = {
                "total_chunks_created": 0,
                "chunks_filtered": 0,
                "filter_reasons": {}
            }

        total = stats.get("total_chunks_created", 0)
        filtered = stats.get("chunks_filtered", 0)

        return {
            "total_chunks_created": total,
            "chunks_filtered": filtered,
            "chunks_kept": total - filtered,
            "filter_rate": filtered / total if total > 0 else 0,
            "filter_reasons": stats.get("filter_reasons", {})
        }

    def update_filtering_stats(
        self,
        chunks_created: int,
        chunks_filtered: int,
        filter_reasons: Dict[str, int]
    ) -> None:
        """
        Update chunk filtering statistics in collection metadata.

        Args:
            chunks_created: Number of chunks created in this batch
            chunks_filtered: Number of chunks filtered in this batch
            filter_reasons: Dictionary of filter reason → count
        """
        # Get existing stats
        current_stats = self.get_filtering_stats()

        # Update totals (cumulative)
        new_total = current_stats["total_chunks_created"] + chunks_created
        new_filtered = current_stats["chunks_filtered"] + chunks_filtered

        # Merge filter reasons (cumulative)
        new_reasons = dict(current_stats["filter_reasons"])
        for reason, count in filter_reasons.items():
            new_reasons[reason] = new_reasons.get(reason, 0) + count

        # Prepare stats object
        stats_obj = {
            "total_chunks_created": new_total,
            "chunks_filtered": new_filtered,
            "filter_reasons": new_reasons
        }

        # Serialize to JSON (ChromaDB metadata only supports simple types)
        stats_json = json.dumps(stats_obj)

        # Update collection metadata
        updated_metadata = dict(self.collection.metadata or {})
        updated_metadata["filtering_stats_json"] = stats_json

        # Persist to ChromaDB
        self.collection.modify(metadata=updated_metadata)

        logger.info(
            f"Updated filtering stats: total={new_total}, filtered={new_filtered}, "
            f"rate={new_filtered/new_total:.1%}"
        )

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document with the given ID exists.

        Args:
            document_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            result = self.collection.get(ids=[document_id])
            return len(result["ids"]) > 0
        except Exception:
            return False
