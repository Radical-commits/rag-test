"""Embedding generation module using BAAI/bge-small-en-v1.5 model."""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using the BGE model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the Hugging Face model to use

        Note:
            Model is loaded once and cached in memory. This is expensive (takes several seconds
            and uses ~100MB of RAM), so we avoid reloading it for every query.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        try:
            # Generate embedding
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once (higher = faster but uses more memory)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If any text is empty or if generation fails
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []

        # Validate all texts are non-empty (filtering should happen upstream)
        empty_indices = [i for i, text in enumerate(texts) if not text or not text.strip()]
        if empty_indices:
            raise ValueError(
                f"Found {len(empty_indices)} empty texts at indices {empty_indices[:10]}. "
                "Chunk validation should happen before embedding generation. "
                "Check DocumentProcessor validation configuration."
            )

        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunks")

            # Generate embeddings using the model
            embeddings = self.model.encode(
                texts,
                show_progress_bar=len(texts) > 100,
                batch_size=batch_size,
                normalize_embeddings=True
            )

            # Convert to list of lists for ChromaDB compatibility
            embeddings_list = embeddings.tolist()

            logger.info(
                f"Successfully generated {len(embeddings_list)} embeddings, "
                f"dimension={len(embeddings_list[0])}"
            )
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (384 for bge-small)
        """
        return self.model.get_sentence_embedding_dimension()
