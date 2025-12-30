"""Claude API client for RAG queries."""

import logging
import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

# Set up dedicated file handler for LLM interactions
llm_log_dir = Path("./logs")
llm_log_dir.mkdir(exist_ok=True)
llm_log_file = llm_log_dir / "llm_interactions.log"

# Create file handler with detailed formatting
file_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Add file handler to logger
llm_logger = logging.getLogger(__name__)
llm_logger.addHandler(file_handler)
llm_logger.setLevel(logging.DEBUG)


class LLMClient:
    """Client for interacting with Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key (reads from ANTHROPIC_API_KEY or LITELLM_API_KEY env var if not provided)
            model: Claude model to use (reads from CLAUDE_MODEL_NAME env var if not provided)
            max_tokens: Maximum tokens in response
            base_url: Optional base URL for LiteLLM proxy (reads from LITELLM_BASE_URL env var if not provided)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        # Try LiteLLM key first, then Anthropic key
        self.api_key = (
            api_key or os.getenv("LITELLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "API key not found. "
                "Set LITELLM_API_KEY or ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Get model name from env var or use provided value or default
        self.model = model or os.getenv("CLAUDE_MODEL_NAME")
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL")

        # Create httpx client with SSL verification disabled for corporate environments
        http_client = httpx.Client(verify=False, timeout=60.0)

        # Configure client based on whether we're using LiteLLM proxy or direct Anthropic
        if self.base_url:
            # Using LiteLLM proxy
            self.client = Anthropic(
                api_key=self.api_key, base_url=self.base_url, http_client=http_client
            )
            logger.info(
                f"Initialized LLM client with model: {self.model} via LiteLLM proxy at {self.base_url} (SSL disabled)"
            )
        else:
            # Direct Anthropic API
            self.client = Anthropic(api_key=self.api_key, http_client=http_client)
            logger.info(
                f"Initialized LLM client with model: {model} via direct Anthropic API (SSL disabled)"
            )

    def query_with_context(
        self,
        query: str,
        context_chunks: List[str],
        metadata_list: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query Claude with retrieved context chunks.

        Args:
            query: User's question
            context_chunks: List of relevant text chunks
            metadata_list: List of metadata for each chunk
            temperature: Sampling temperature (0 = deterministic, 1 = creative)

        Returns:
            Dictionary containing:
                - answer: Claude's response
                - model: Model used
                - usage: Token usage information
                - sources: List of sources used

        Raises:
            ValueError: If query or context is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not context_chunks:
            raise ValueError("Context chunks cannot be empty")

        try:
            # Build context string with source citations
            context_parts = []
            sources = []

            for i, (chunk, metadata) in enumerate(
                zip(context_chunks, metadata_list), 1
            ):
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "?")
                chunk_idx = metadata.get("chunk_index", "?")

                context_parts.append(
                    f"[Source {i}: {source}, Page {page}, Chunk {chunk_idx}]\n{chunk}\n"
                )
                sources.append(
                    {"source": source, "page": page, "chunk_index": chunk_idx}
                )

            context_text = "\n---\n".join(context_parts)

            # Build prompt
            prompt = self._build_prompt(query, context_text)

            # Log the request in human-readable format
            logger.info("="*80)
            logger.info("LLM REQUEST")
            logger.info(f"Query: {query}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Context chunks: {len(context_chunks)}")
            logger.info(f"Temperature: {temperature}")
            logger.info("="*80)

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract answer
            answer = response.content[0].text

            # Log the response in human-readable format
            logger.info("="*80)
            logger.info("LLM RESPONSE")
            logger.info(f"Model: {response.model}")
            logger.info(f"Input tokens: {response.usage.input_tokens}")
            logger.info(f"Output tokens: {response.usage.output_tokens}")
            logger.info(f"Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")
            logger.info(f"Stop reason: {response.stop_reason}")
            logger.info(f"Answer preview: {answer[:200]}...")
            logger.info("="*80)

            return {
                "answer": answer,
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "sources": sources,
            }

        except Exception as e:
            logger.error("="*80)
            logger.error(f"LLM ERROR: {e}")
            logger.error("="*80)
            raise

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build a prompt for Claude with context.

        Args:
            query: User's question
            context: Retrieved context chunks

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful assistant that answers questions based on provided document context.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context below
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Include specific references to sources (e.g., "According to Source 1...")
4. Be concise but thorough
5. If you find contradictory information in different sources, mention this
6. Do not make up or infer information that isn't in the context

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

Please provide a clear, accurate answer based on the context above:"""

        return prompt

    def simple_query(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Send a simple query to Claude without RAG context.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature

        Returns:
            Claude's response text

        Raises:
            Exception: If API call fails
        """
        try:
            # Log the request in human-readable format
            logger.info("="*80)
            logger.info("SIMPLE LLM REQUEST")
            logger.info(f"Model: {self.model}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Prompt preview: {prompt[:200]}...")
            logger.info("="*80)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.content[0].text

            # Log the response in human-readable format
            logger.info("="*80)
            logger.info("SIMPLE LLM RESPONSE")
            logger.info(f"Model: {response.model}")
            logger.info(f"Input tokens: {response.usage.input_tokens}")
            logger.info(f"Output tokens: {response.usage.output_tokens}")
            logger.info(f"Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")
            logger.info(f"Stop reason: {response.stop_reason}")
            logger.info(f"Answer preview: {answer[:200]}...")
            logger.info("="*80)

            return answer

        except Exception as e:
            logger.error("="*80)
            logger.error(f"SIMPLE LLM ERROR: {e}")
            logger.error("="*80)
            raise
