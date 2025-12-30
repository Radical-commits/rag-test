# RAG Document Q&A System

A complete Retrieval-Augmented Generation (RAG) system for internal document question-answering. Upload PDF documents and ask questions using AI-powered semantic search and Claude's language models.

## Documentation

- **README.md** (this file) - Complete documentation and usage guide
- **SETUP.md** - Detailed setup instructions for both Anthropic and LiteLLM configurations
- **ARCHITECTURE.md** - Technical deep-dive into system design and components

## Overview

This system allows you to:
- Upload and index PDF documents
- Ask natural language questions about your documents
- Get accurate answers with source citations
- See which parts of documents were used to generate answers

## Architecture

The system consists of several key components:

1. **Document Processing** (`src/document_processor.py`)
   - Uses Docling to parse PDF files with OCR and table detection
   - Extracts text from both standard and scanned PDFs
   - Preserves document structure including tables
   - Stores metadata (filename, page numbers, chunk index)

2. **Embedding Generation** (`src/embeddings.py`)
   - Uses BAAI/bge-small-en-v1.5 model to create vector embeddings
   - Converts text chunks into 384-dimensional vectors
   - Enables semantic similarity search

3. **Vector Storage** (`src/vector_store.py`)
   - Uses ChromaDB for efficient vector storage and retrieval
   - Persists data locally for reuse across sessions
   - Performs semantic search to find relevant chunks

4. **LLM Integration** (`src/llm_client.py`)
   - Connects to Claude API (Sonnet 3.5 by default)
   - Formats retrieved chunks as context
   - Generates accurate, context-based answers

5. **RAG Pipeline** (`src/rag_pipeline.py`)
   - Orchestrates all components
   - Manages indexing workflow
   - Handles query processing end-to-end

6. **Web Interface** (`app.py`)
   - Streamlit-based UI for easy interaction
   - File upload and indexing interface
   - Query input and results display
   - Source citations and relevance scores

## Quick Start

For detailed setup instructions, see **SETUP.md**. Quick overview:

### Prerequisites

- Python 3.12 or higher
- UV package manager (install from https://docs.astral.sh/uv/)
- Anthropic API key OR LiteLLM proxy credentials

### Installation

1. **Clone or download this repository**

2. **Install dependencies using UV**
   ```bash
   uv sync
   ```

3. **Download the embedding model manually** (required for corporate SSL environments)

   Due to SSL certificate issues in corporate networks, use this one-line Python command to download the model:

   ```bash
   uv run python -c "import os,ssl,requests,urllib3;from pathlib import Path;from tqdm import tqdm;os.environ.update({'CURL_CA_BUNDLE':'','REQUESTS_CA_BUNDLE':'','SSL_CERT_FILE':'','PYTHONHTTPSVERIFY':'0','HF_HUB_DISABLE_SSL_VERIFY':'1'});ssl._create_default_https_context=ssl._create_unverified_context;urllib3.disable_warnings();og=requests.get;requests.get=lambda u,**k:og(u,**{**k,'verify':False});BASE='https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main';FILES=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','sentence_bert_config.json','modules.json','config_sentence_transformers.json','README.md','1_Pooling/config.json'];[[(print(f'Downloading {f}...'),p:=Path(f'models/bge-small-en-v1.5/{f}'),p.parent.mkdir(parents=True,exist_ok=True),r:=requests.get(f'{BASE}/{f}',stream=True,verify=False),r.raise_for_status(),[(fp:=open(p,'wb')),(bar:=tqdm(total=int(r.headers.get('content-length',0)),unit='B',unit_scale=True))]+[fp.write(chunk) or bar.update(len(chunk)) for chunk in r.iter_content(8192) if chunk]+[fp.close(),bar.close()],print(f'✓ {f}'))] for f in FILES];print('✓ All files downloaded!')"
   ```

   Verify: `ls -lh ./models/bge-small-en-v1.5/model.safetensors` (should be ~127 MB)

   **Alternative:** See `SETUP.md` for alternative download methods.

4. **Set up environment variables**

   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and configure your API access:

   **Option A: Direct Anthropic API**
   ```
   ANTHROPIC_API_KEY=sk-ant-api03-your-key
   ```

   **Option B: Corporate LiteLLM Proxy**
   ```
   LITELLM_API_KEY=sk-your-litellm-token
   LITELLM_BASE_URL=https://your-proxy.company.com/anthropic
   CLAUDE_MODEL_NAME=bedrock-claude-4.5-sonnet
   ```

   See **SETUP.md** for complete configuration instructions.

5. **Run the application**
   ```bash
   uv run streamlit run app.py
   ```

   The application will open in your browser at http://localhost:8501

## Usage Guide

### Indexing Documents

1. Click "Browse files" in the Document Upload section
2. Select one or more PDF files
3. Click "Index Documents" button
4. Wait for processing to complete (progress will be shown)
5. You'll see a success message with the number of chunks indexed

**Note:** The system automatically skips documents that have already been indexed to avoid duplicates.

### Asking Questions

1. After indexing documents, scroll to the "Ask Questions" section
2. Type your question in the text area
3. Adjust the number of sources to retrieve (default: 5)
4. (Optional) Configure Advanced Retrieval Options:
   - **MMR (Maximal Marginal Relevance)**: Enabled by default for diverse results
   - **MMR Balance (λ)**: Adjust from 0 (max diversity) to 1 (max relevance)
   - Default: λ=0.5 provides balanced results
5. Click "Ask" button
6. View the answer and source citations

### Understanding Results

- **Answer**: Claude's response based on retrieved context
- **Sources Used**: Expandable sections showing which document chunks were used
- **Relevance Score**: How similar each chunk is to your query (0-1 scale, higher = more relevant)
- **Token Usage**: Shows input/output tokens for API cost tracking

### Retrieval Quality Features

The system uses multiple techniques to ensure high-quality results:

#### 1. Chunk Quality Filtering (During Indexing)
Automatically filters out low-quality chunks before storage:
- **Table artifacts**: Removes formatting characters from PDF tables (e.g., `| |`)
- **Short fragments**: Filters chunks under 100 characters (section headers, page breaks)
- **Low content ratio**: Removes chunks with less than 30% alphanumeric characters
- **High special character ratio**: Filters chunks with more than 70% special characters
- **Whitelisted patterns**: Preserves code blocks and technical content

**Benefits:**
- Cleaner vector database with only meaningful content
- Better semantic search accuracy
- No garbage text sent to Claude

**Monitoring:**
Check the sidebar "Chunk Filtering" section to see:
- Total chunks created vs. filtered
- Filter rate percentage
- Breakdown of filter reasons (too_short, table_artifact, etc.)

#### 2. MMR (Maximal Marginal Relevance) - During Retrieval
Balances relevance and diversity in search results:
- **How it works**: Fetches 3x candidate chunks, then iteratively selects documents that are both relevant to your query AND different from already-selected chunks
- **λ parameter**: Controls balance between relevance (1.0) and diversity (0.0)
- **Default**: λ=0.5 provides balanced results
- **Use cases**:
  - λ=0.7-1.0: Prioritize relevance (similar chunks acceptable)
  - λ=0.3-0.5: Prioritize diversity (broader coverage of topic)
  - λ=0.0-0.2: Maximum diversity (different perspectives)

**Benefits:**
- Prevents 5 nearly identical chunks from same section
- Provides broader coverage of topic across different document sections
- Better answers when information is spread across document

**Configuration:**
- Enable/disable in "Advanced Retrieval Options" expander
- Adjust λ slider to tune behavior
- Enabled by default for best results

#### 3. Deduplication - During Retrieval
Removes near-duplicate chunks from results:
- **Similarity threshold**: 80% text similarity
- **Algorithm**: Uses SequenceMatcher for fast text comparison
- **Applied after**: MMR reranking (filters MMR output)

**Benefits:**
- Removes chunks that repeat the same content (e.g., from page overlaps)
- Ensures each chunk adds unique information
- Better use of Claude's context window

**Always enabled** - no configuration needed.

### Managing the Index

- **Clear Index**: Removes all indexed documents (cannot be undone)
- **System Statistics**: View in sidebar - shows document count, models used, etc.

## Technical Details

### Why Models Are Loaded Once

The embedding model (~100MB) is loaded when the application starts and kept in memory. This is intentional because:
- Model loading takes several seconds
- Loading on every query would make the system very slow
- Memory usage is acceptable for internal tooling

### Chunking Strategy

Documents are split into chunks of ~1000 characters with 100-character overlap:
- **Overlap**: Maintains context at chunk boundaries while minimizing duplicates
- **Smart splitting**: Breaks at paragraph or sentence boundaries when possible
- **Metadata**: Each chunk knows its source, page, and position
- **Quality filtering**: Filters out low-quality chunks (table artifacts, short fragments, excessive special characters)

### Search Process

1. Your question is converted to a vector embedding
2. ChromaDB finds similar document chunks using cosine similarity
3. **MMR (Maximal Marginal Relevance)** reranking (optional):
   - Fetches 3x candidate chunks for diversity
   - Balances relevance to query with diversity in results
   - Prevents repetitive or near-duplicate content
   - Controlled by λ parameter (0=max diversity, 1=max relevance)
4. **Deduplication** removes near-duplicate chunks:
   - Uses text similarity comparison (80% threshold)
   - Ensures varied perspectives in context
5. Top-k chunks are formatted and sent to Claude as context
6. Claude generates an answer based only on the provided context

## Project Structure

```
rag-test/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── document_processor.py    # PDF processing and chunking
│   ├── embeddings.py            # Embedding generation
│   ├── vector_store.py          # ChromaDB integration
│   ├── llm_client.py            # Claude API client
│   └── rag_pipeline.py          # Main orchestration
├── models/                      # Downloaded embedding models (gitignored)
│   └── bge-small-en-v1.5/      # BGE embedding model
├── chroma_db/                   # Vector database storage (gitignored)
├── app.py                       # Streamlit web UI
├── pyproject.toml               # Dependencies and project config
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
├── CORPORATE_SSL_FIX.md        # SSL configuration notes
└── README.md                    # This file
```

## Dependencies

Key packages (see `pyproject.toml` for complete list):
- `docling` - PDF processing
- `sentence-transformers` - BGE embedding model
- `chromadb` - Vector database
- `anthropic` - Claude API client
- `streamlit` - Web interface
- `python-dotenv` - Environment variable management

## Troubleshooting

### API Configuration Issues

**Error:** "No API configuration found"
- Make sure you created a `.env` file in the project root
- Add either ANTHROPIC_API_KEY or LITELLM_API_KEY + LITELLM_BASE_URL
- Check that keys are correctly pasted (no extra spaces or quotes)
- Verify the file is named exactly `.env` (not `.env.txt`)
- Restart the application after editing `.env`

See **SETUP.md** for detailed configuration and troubleshooting.

### Model Loading Issues

**Error:** "Failed to load embedding model" or SSL certificate errors
- The model is now loaded from the local `./models/` directory
- If you haven't downloaded it yet, follow step 3 in Setup Instructions
- Make sure the model directory exists at `./models/bge-small-en-v1.5/`
- Ensure you have enough disk space (~500MB for models)

**Error:** "Can't load the model" after manual download
- Verify the model was cloned completely: `ls ./models/bge-small-en-v1.5/`
- You should see files like `config.json`, `pytorch_model.bin`, etc.
- If incomplete, delete the folder and re-clone with `GIT_SSL_NO_VERIFY=true`

### PDF Processing Errors

**Error:** "Failed to process PDF"
- Make sure the file is a valid PDF (not corrupted)
- Some PDFs with complex formatting may fail - try a simpler version
- Check that the file isn't password-protected

**Error:** "CAS service error" or connection timeouts during indexing
- This is fixed by disabling Hugging Face XET mode
- The fix is already in `app.py`: `HF_HUB_DISABLE_XET=1`
- Restart the app if you still see this error
- OCR and table detection now work properly

### Memory Issues

If the application crashes with memory errors:
- Reduce `chunk_size` in `app.py` (line 22)
- Process fewer documents at once
- Reduce `top_k` when querying (default: 5)

## Configuration Options

You can customize the system by modifying parameters in `app.py`:

```python
RAGPipeline(
    chunk_size=1000,              # Characters per chunk
    chunk_overlap=100,            # Overlap between chunks (reduced to minimize duplicates)
    embedding_model="./models/bge-small-en-v1.5",  # Local embedding model path
    claude_model="claude-3-5-sonnet-20241022",     # Claude model
    chroma_db_path="./chroma_db", # Vector DB location
    chunk_validation_config=ChunkValidationConfig(
        min_length=100,           # Minimum chunk length in characters
        min_meaningful_ratio=0.3, # Minimum ratio of alphanumeric content
        max_special_char_ratio=0.7 # Maximum ratio of special characters
    )
)
```

**Chunk Validation Parameters:**
- `min_length`: Minimum characters required (default: 100). Filters short fragments.
- `min_meaningful_ratio`: Minimum alphanumeric content ratio (default: 0.3). Filters non-text artifacts.
- `max_special_char_ratio`: Maximum special character ratio (default: 0.7). Filters formatting artifacts.

**Retrieval Parameters** (configured in UI):
- `use_mmr`: Enable MMR for diverse results (default: True)
- `mmr_lambda`: Balance relevance vs. diversity (default: 0.5, range: 0.0-1.0)
- `deduplicate`: Enable deduplication (default: True, always on)

**Model Options:**

For direct Anthropic API:
- Sonnet 3.5: Best quality, slower, more expensive
- Haiku 3.5: Faster, cheaper, good quality
- Set in `.env`: `CLAUDE_MODEL_NAME=claude-3-5-haiku-20241022`

For LiteLLM proxy:
- Use model names configured by your admin
- Common: `bedrock-claude-4.5-sonnet`, `anthropic/claude-3-5-sonnet`
- Set in `.env`: `CLAUDE_MODEL_NAME=bedrock-claude-4.5-sonnet`

## LLM Interaction Logging

All interactions with Claude are automatically logged to `./logs/llm_interactions.log` in a simple, human-readable format for debugging and cost tracking.

**What's logged:**
- Query text
- Model name
- Number of context chunks used
- Temperature setting
- Token usage (input, output, total)
- Stop reason
- Answer preview (first 200 characters)
- Timestamps for each request/response
- Error messages if API calls fail

**Example log entry:**
```
================================================================================
2025-12-29 16:30:15 - src.llm_client - INFO - LLM REQUEST
2025-12-29 16:30:15 - src.llm_client - INFO - Query: What are the key features?
2025-12-29 16:30:15 - src.llm_client - INFO - Model: bedrock-claude-4.5-sonnet
2025-12-29 16:30:15 - src.llm_client - INFO - Context chunks: 5
2025-12-29 16:30:15 - src.llm_client - INFO - Temperature: 0.0
================================================================================
2025-12-29 16:30:18 - src.llm_client - INFO - LLM RESPONSE
2025-12-29 16:30:18 - src.llm_client - INFO - Model: bedrock-claude-4.5-sonnet
2025-12-29 16:30:18 - src.llm_client - INFO - Input tokens: 2341
2025-12-29 16:30:18 - src.llm_client - INFO - Output tokens: 156
2025-12-29 16:30:18 - src.llm_client - INFO - Total tokens: 2497
2025-12-29 16:30:18 - src.llm_client - INFO - Stop reason: end_turn
2025-12-29 16:30:18 - src.llm_client - INFO - Answer preview: Based on the provided context, the key features include: 1. Document processing with Docling for PDF extraction 2. Semantic search using BGE embeddings 3. Vector storage with ChromaDB 4. Q...
================================================================================
```

**View logs in real-time:**
```bash
tail -f ./logs/llm_interactions.log
```

**Find specific queries:**
```bash
grep "Query:" ./logs/llm_interactions.log
```

**Track token usage:**
```bash
grep "Total tokens:" ./logs/llm_interactions.log
```

The log file uses UTF-8 encoding and includes visual separators (80 `=` characters) between each request/response pair for easy readability.

## Performance Notes

- **First run**: Fast model loading from local directory (~5-10 seconds)
- **Document indexing**: ~5-10 seconds per PDF page
- **Queries**: ~2-5 seconds depending on Claude model
- **Storage**: ~1KB per chunk in ChromaDB
- **Model size**: BGE model uses ~100MB in memory once loaded

## Security Notes

- Never commit your `.env` file (it's in `.gitignore`)
- API keys are loaded from environment variables only
- ChromaDB data is stored locally (not shared)
- Uploaded PDFs are temporarily stored and can be deleted

## Future Enhancements

Possible improvements for later:
- Support for more document formats (DOCX, TXT, HTML)
- Advanced filtering by document metadata
- Conversation history and follow-up questions
- Export results to PDF or markdown
- Custom prompt templates
- Multi-language support

## License

Internal tool for organizational use.

## Support

For issues or questions, contact the development team or check the logs in the terminal where you ran the application.