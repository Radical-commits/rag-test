"""Streamlit web UI for the RAG system."""

# CRITICAL: Disable SSL verification BEFORE any other imports
import os
import ssl

# Set environment variables before any library loads
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"  # Hugging Face Hub specific
os.environ["HF_HUB_DISABLE_XET"] = "1"  # Disable Hugging Face XET mode

# Disable SSL verification for Python's default HTTPS context
ssl._create_default_https_context = ssl._create_unverified_context

# Now import other libraries
import streamlit as st
import logging
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import urllib3
import requests

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification globally
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Also patch the base requests methods
original_get = requests.get
original_post = requests.post
def patched_get(url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_get(url, **kwargs)
def patched_post(url, **kwargs):
    kwargs.setdefault('verify', False)
    return original_post(url, **kwargs)
requests.get = patched_get
requests.post = patched_post

# Load environment variables
load_dotenv()

# Import after SSL is disabled
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="RAG Document Q&A System - Christmas Edition", page_icon="🎄", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chunk-box {
        background-color: #e8eaf0;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_pipeline():
    """Initialize the RAG pipeline and store in session state."""
    if "pipeline" not in st.session_state:
        try:
            with st.spinner("Loading models... This may take a minute on first run."):
                # Import here to avoid circular import
                from src.chunk_validator import ChunkValidationConfig

                # Configure chunk quality validation
                validation_config = ChunkValidationConfig(
                    min_length=100,               # Minimum 100 characters (filters short boundary chunks)
                    min_meaningful_ratio=0.3,     # At least 30% alphanumeric
                    max_special_char_ratio=0.7,   # Max 70% special characters
                )
                logger.info(
                    f"Chunk validation: min_length={validation_config.min_length}, "
                    f"min_meaningful={validation_config.min_meaningful_ratio:.0%}, "
                    f"max_special={validation_config.max_special_char_ratio:.0%}"
                )

                st.session_state.pipeline = RAGPipeline(
                    chunk_size=1000,
                    chunk_overlap=100,  # Reduced from 200 to minimize near-duplicate chunks
                    embedding_model="./models/bge-small-en-v1.5",
                    chroma_db_path="./chroma_db",
                    chunk_validation_config=validation_config
                )
            st.success("System initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            logger.error(f"Pipeline initialization error: {e}", exc_info=True)
            st.stop()


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to temporary directory."""
    temp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def display_stats():
    """Display system statistics in sidebar."""
    try:
        stats = st.session_state.pipeline.get_stats()

        st.sidebar.markdown("### System Statistics")
        st.sidebar.metric("Documents Indexed", stats["document_count"])
        st.sidebar.metric("Embedding Model", stats["embedding_model"])
        st.sidebar.metric("LLM Model", stats["llm_model"])
        st.sidebar.metric("Embedding Dimension", stats["embedding_dimension"])

        # Chunk Filtering Statistics
        if "chunk_filtering" in stats:
            st.sidebar.markdown("### Chunk Filtering")
            filter_stats = stats["chunk_filtering"]

            # Check if this is first run (no stats yet)
            if filter_stats["total_chunks_created"] == 0:
                st.sidebar.info(
                    "No filtering stats yet. Stats will appear after indexing documents."
                )
            else:
                # Main metrics
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Total Chunks", filter_stats["total_chunks_created"])
                with col2:
                    st.metric("Filtered", filter_stats["chunks_filtered"])

                # Filter rate
                st.sidebar.metric(
                    "Filter Rate",
                    f"{filter_stats['filter_rate']:.1%}",
                    help="Percentage of chunks filtered due to low quality"
                )

                # Filter reasons breakdown
                if filter_stats["filter_reasons"]:
                    with st.sidebar.expander("Filter Reasons"):
                        for reason, count in sorted(
                            filter_stats["filter_reasons"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            st.text(f"{reason}: {count}")

    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")


def document_upload_section():
    """Handle document upload and indexing."""
    st.markdown('<h3 class="section-header">📄 Document Upload</h3>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF documents to index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to add to the knowledge base",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        index_button = st.button(
            "Index Documents", type="primary", disabled=not uploaded_files
        )
    with col2:
        clear_button = st.button("Clear Index", type="secondary")

    if clear_button:
        if st.warning("Are you sure? This will delete all indexed documents."):
            try:
                st.session_state.pipeline.clear_index()
                st.success("Index cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing index: {e}")

    if index_button and uploaded_files:
        try:
            # Save uploaded files
            pdf_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                pdf_paths.append(file_path)

            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, message):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Progress: {current}/{total} - {message}")

            # Index documents
            results = st.session_state.pipeline.index_documents(
                pdf_paths=pdf_paths,
                skip_duplicates=True,
                progress_callback=progress_callback,
            )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display results
            if results["new_chunks"] > 0:
                st.success(
                    f"Successfully indexed {results['total_files']} files "
                    f"({results['new_chunks']} new chunks)"
                )

            if results["skipped_files"] > 0:
                st.info(f"Skipped {results['skipped_files']} already indexed files")

            if results["errors"]:
                st.error("Some files failed to process:")
                for error in results["errors"]:
                    st.text(f"- {error}")

            # Refresh stats
            st.rerun()

        except Exception as e:
            st.error(f"Error indexing documents: {str(e)}")
            logger.error(f"Indexing error: {e}", exc_info=True)


def query_section():
    """Handle user queries."""
    st.markdown('<h3 class="section-header">🔍 Ask Questions</h3>', unsafe_allow_html=True)

    # Check if there are documents to query
    stats = st.session_state.pipeline.get_stats()
    if stats["document_count"] == 0:
        st.warning("No documents indexed yet. Please upload and index documents first.")
        return

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="What would you like to know about the documents?",
        height=100,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary", disabled=not query)
    with col2:
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)

    # Advanced options
    with st.expander("Advanced Retrieval Options"):
        use_mmr = st.checkbox(
            "Use MMR (Maximal Marginal Relevance)",
            value=True,
            help="MMR increases diversity in retrieved results by balancing relevance and dissimilarity"
        )
        mmr_lambda = st.slider(
            "MMR Balance (λ)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = maximum diversity, 1 = maximum relevance",
            disabled=not use_mmr
        )

    if ask_button and query:
        try:
            with st.spinner("Searching documents and generating answer..."):
                response = st.session_state.pipeline.query(
                    question=query,
                    top_k=top_k,
                    temperature=0.0,
                    use_mmr=use_mmr,
                    mmr_lambda=mmr_lambda
                )

            # Display answer
            st.markdown("### Answer")
            st.markdown(response["answer"])

            # Display token usage
            usage = response["usage"]
            st.caption(
                f"Model: {response['model']} | "
                f"Input tokens: {usage['input_tokens']} | "
                f"Output tokens: {usage['output_tokens']}"
            )

            # Display sources
            st.markdown("### Sources Used")
            for i, source in enumerate(response["sources"], 1):
                with st.expander(
                    f"Source {i}: {source['source']} (Page {source['page']})"
                ):
                    chunk_info = response["retrieved_chunks"][i - 1]
                    st.markdown(
                        f"**Relevance Score:** {1 - chunk_info['distance']:.3f}"
                    )
                    st.markdown("**Text:**")
                    st.text(
                        chunk_info["text"][:500]
                        + ("..." if len(chunk_info["text"]) > 500 else "")
                    )

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query error: {e}", exc_info=True)


def main():
    """Main application entry point."""
    # Header
    st.markdown(
        '<h1 class="main-header">🎄 RAG Document Q&A System - Christmas Edition</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload PDF documents and ask questions using AI-powered search</p>',
        unsafe_allow_html=True,
    )

    # Check for API key (LiteLLM or direct Anthropic)
    litellm_key = os.getenv("LITELLM_API_KEY")
    litellm_url = os.getenv("LITELLM_BASE_URL")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not (litellm_key or anthropic_key):
        st.error(
            "⚠️ No API configuration found. "
            "Please set up your .env file with either LiteLLM or Anthropic credentials."
        )
        st.info(
            "Create a .env file in the project root with:\n\n"
            "**Option 1 (LiteLLM Proxy):**\n"
            "```\n"
            "LITELLM_API_KEY=sk-your-litellm-token\n"
            "LITELLM_BASE_URL=https://your-proxy.com/anthropic\n"
            "CLAUDE_MODEL_NAME=bedrock-claude-4.5-sonnet\n"
            "```\n\n"
            "**Option 2 (Direct Anthropic):**\n"
            "```\n"
            "ANTHROPIC_API_KEY=sk-ant-api03-your-key\n"
            "```"
        )
        st.stop()

    # Display which configuration is being used
    if litellm_key and litellm_url:
        st.sidebar.success("✓ Using LiteLLM Proxy")
    elif anthropic_key:
        st.sidebar.success("✓ Using Direct Anthropic API")

    # Initialize pipeline
    initialize_pipeline()

    # Sidebar with stats
    display_stats()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This RAG system uses:\n"
        "- **Docling** for PDF processing\n"
        "- **BGE** embeddings\n"
        "- **ChromaDB** for vector storage\n"
        "- **Claude** for question answering"
    )

    # Main sections
    document_upload_section()
    st.markdown("---")
    query_section()


if __name__ == "__main__":
    main()
