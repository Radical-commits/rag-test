"""Main RAG pipeline orchestration."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from .document_processor import DocumentProcessor, DocumentChunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .llm_client import LLMClient
from .chunk_validator import ChunkValidationConfig

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the complete RAG pipeline."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        claude_model: Optional[str] = None,
        chroma_db_path: str = "./chroma_db",
        anthropic_api_key: Optional[str] = None,
        chunk_validation_config: Optional[ChunkValidationConfig] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model to use
            claude_model: Claude model to use (reads from CLAUDE_MODEL_NAME env var if not provided)
            chroma_db_path: Path to ChromaDB persistence directory
            anthropic_api_key: API key (reads from LITELLM_API_KEY or ANTHROPIC_API_KEY env var if not provided)
            chunk_validation_config: Configuration for chunk quality filtering
        """
        logger.info("Initializing RAG pipeline components...")

        # Initialize components with validation config
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            validation_config=chunk_validation_config
        )

        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)

        self.vector_store = VectorStore(
            persist_directory=chroma_db_path,
            collection_name="documents"
        )

        self.llm_client = LLMClient(
            api_key=anthropic_api_key,
            model=claude_model
        )

        # Cache to track processed documents (to avoid reprocessing)
        self.processed_documents = set()
        self._load_processed_documents()

        logger.info("RAG pipeline initialized successfully")

    def _load_processed_documents(self) -> None:
        """Load list of already processed documents from vector store."""
        try:
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Found {stats['document_count']} existing documents in vector store")
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")

    def _get_document_hash(self, file_path: Path) -> str:
        """
        Generate a hash for a document based on filename and size.

        Args:
            file_path: Path to document

        Returns:
            Hash string
        """
        file_stat = file_path.stat()
        hash_input = f"{file_path.name}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def index_documents(
        self,
        pdf_paths: List[Path],
        skip_duplicates: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Index PDF documents into the vector store.

        Args:
            pdf_paths: List of paths to PDF files
            skip_duplicates: If True, skip documents that are already indexed
            progress_callback: Optional callback function(current, total, message) for progress updates

        Returns:
            Dictionary with indexing results:
                - total_files: Number of files processed
                - new_chunks: Number of new chunks added
                - skipped_files: Number of files skipped
                - errors: List of error messages
        """
        logger.info(f"Starting document indexing for {len(pdf_paths)} files")

        results = {
            "total_files": len(pdf_paths),
            "new_chunks": 0,
            "skipped_files": 0,
            "errors": []
        }

        all_chunks = []
        processed_files = 0

        # Accumulate filtering stats across all files
        total_stats = {
            "chunks_created": 0,
            "chunks_filtered": 0,
            "filter_reasons": {}
        }

        for i, pdf_path in enumerate(pdf_paths):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(
                        i + 1,
                        len(pdf_paths),
                        f"Processing {pdf_path.name}..."
                    )

                # Check if already processed
                doc_hash = self._get_document_hash(pdf_path)
                if skip_duplicates and doc_hash in self.processed_documents:
                    logger.info(f"Skipping already indexed document: {pdf_path.name}")
                    results["skipped_files"] += 1
                    continue

                # Process document (now returns chunks and stats)
                chunks, batch_stats = self.document_processor.process_pdf(pdf_path)

                if chunks:
                    all_chunks.extend(chunks)
                    self.processed_documents.add(doc_hash)
                    processed_files += 1

                # Accumulate stats
                total_stats["chunks_created"] += batch_stats["chunks_created"]
                total_stats["chunks_filtered"] += batch_stats["chunks_filtered"]
                for reason, count in batch_stats["filter_reasons"].items():
                    total_stats["filter_reasons"][reason] = total_stats["filter_reasons"].get(reason, 0) + count

            except Exception as e:
                error_msg = f"Error processing {pdf_path.name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Generate embeddings and store in vector database
        if all_chunks:
            try:
                if progress_callback:
                    progress_callback(
                        len(pdf_paths),
                        len(pdf_paths),
                        f"Generating embeddings for {len(all_chunks)} chunks..."
                    )

                # Extract texts
                texts = [chunk.text for chunk in all_chunks]

                # Generate embeddings in batch
                embeddings = self.embedding_generator.generate_embeddings_batch(texts)

                # Prepare metadata
                metadatas = [
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata
                    }
                    for chunk in all_chunks
                ]

                # Add to vector store
                if progress_callback:
                    progress_callback(
                        len(pdf_paths),
                        len(pdf_paths),
                        f"Storing {len(all_chunks)} chunks in vector database..."
                    )

                self.vector_store.add_documents(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                # Persist filtering stats to vector store
                if total_stats["chunks_created"] > 0:
                    self.vector_store.update_filtering_stats(
                        chunks_created=total_stats["chunks_created"],
                        chunks_filtered=total_stats["chunks_filtered"],
                        filter_reasons=total_stats["filter_reasons"]
                    )

                results["new_chunks"] = len(all_chunks)
                logger.info(f"Successfully indexed {processed_files} files with {len(all_chunks)} chunks")

                # Log chunk filtering statistics
                filter_rate = total_stats["chunks_filtered"] / total_stats["chunks_created"] if total_stats["chunks_created"] > 0 else 0
                logger.info(
                    f"Chunk filtering summary: "
                    f"{total_stats['chunks_filtered']} filtered out of {total_stats['chunks_created']} total "
                    f"({filter_rate:.1%} filter rate)"
                )
                if total_stats['filter_reasons']:
                    logger.info(f"Filter reasons: {total_stats['filter_reasons']}")

            except Exception as e:
                error_msg = f"Error storing embeddings: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.0,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            temperature: Claude temperature setting (0 = deterministic, 1 = creative)
            use_mmr: If True, apply MMR for diverse results (default: True)
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance, default: 0.5)

        Returns:
            Dictionary containing:
                - answer: Claude's response
                - sources: List of source documents used
                - retrieved_chunks: Text chunks that were retrieved
                - model: Model used for generation
                - usage: Token usage statistics

        Raises:
            ValueError: If question is empty or vector store is empty
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        stats = self.vector_store.get_collection_stats()
        if stats["document_count"] == 0:
            raise ValueError(
                "Vector store is empty. Please index some documents first."
            )

        logger.info(f"Processing query: '{question[:50]}...'")

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(question)

            # Retrieve relevant chunks with deduplication and optional MMR
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=top_k,
                deduplicate=True,  # Enable deduplication by default
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda
            )

            retrieved_chunks = search_results["documents"]
            metadatas = search_results["metadatas"]
            distances = search_results["distances"]

            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")

            # Query LLM with context
            llm_response = self.llm_client.query_with_context(
                query=question,
                context_chunks=retrieved_chunks,
                metadata_list=metadatas,
                temperature=temperature
            )

            # Add retrieval information to response
            llm_response["retrieved_chunks"] = [
                {
                    "text": chunk,
                    "metadata": metadata,
                    "distance": distance
                }
                for chunk, metadata, distance in zip(retrieved_chunks, metadatas, distances)
            ]

            return llm_response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.

        Returns:
            Dictionary with system statistics including chunk filtering metrics
        """
        vector_stats = self.vector_store.get_collection_stats()
        embedding_dim = self.embedding_generator.get_embedding_dimension()

        # Load filtering stats from vector store (persisted in metadata)
        filter_stats = self.vector_store.get_filtering_stats()

        return {
            **vector_stats,
            "embedding_dimension": embedding_dim,
            "embedding_model": self.embedding_generator.model_name,
            "llm_model": self.llm_client.model,
            "processed_documents_count": len(self.processed_documents),
            "chunk_filtering": filter_stats
        }

    def clear_index(self) -> None:
        """
        Clear all indexed documents from the vector store.

        Warning:
            This operation cannot be undone.
        """
        logger.warning("Clearing vector store index")
        self.vector_store.clear_collection()
        self.processed_documents.clear()
        logger.info("Index cleared successfully")
