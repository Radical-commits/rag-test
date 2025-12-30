<objective>
Build a complete RAG (Retrieval-Augmented Generation) system in Python for internal tooling use. The system should process PDF documents using Docling, create embeddings with BAAI/bge-small-en-v1.5, store vectors in ChromaDB, and provide a simple Streamlit web UI for querying the indexed data using Claude's Sonnet or Haiku models.

This will be used as an internal tool for document question-answering, allowing team members to query PDF documents through a natural language interface.
</objective>

<context>
- Project management: UV (for Python package management)
- Document processing: Docling (for PDF parsing and chunking)
- Embeddings: BAAI/bge-small-en-v1.5 model
- Vector database: ChromaDB
- LLM: Claude Sonnet or Haiku via Anthropic API
- UI framework: Streamlit
- This is for internal tooling, so moderate polish is acceptable

Before starting, read @CLAUDE.md to understand project conventions and coding style.
</context>

<requirements>
1. **Document Processing Pipeline**:
   - Use Docling to parse PDF files and extract text
   - Implement intelligent chunking strategy (consider document structure, sections, paragraphs)
   - Handle multiple PDFs and store metadata (filename, chunk index, page numbers)

2. **Embedding and Vector Storage**:
   - Load and use BAAI/bge-small-en-v1.5 model for generating embeddings
   - Store embeddings in ChromaDB with associated metadata
   - Implement efficient batch processing for large documents

3. **Retrieval System**:
   - Implement semantic search using ChromaDB to find relevant chunks
   - Return top-k most relevant chunks with metadata
   - Include relevance scores in results

4. **LLM Integration**:
   - Use Anthropic API to query Claude (Sonnet or Haiku)
   - Pass retrieved chunks as context to the LLM
   - Format prompts to encourage accurate, context-based answers
   - Handle API errors gracefully

5. **Streamlit Web UI**:
   - File upload interface for PDFs (single or batch)
   - Display indexing progress and status
   - Query input field with submit button
   - Display results with source citations (which document/page)
   - Show retrieved chunks that were used for the answer
   - Simple, clean layout suitable for internal use

6. **Configuration and Setup**:
   - Use UV for dependency management (pyproject.toml)
   - Store API keys in environment variables (never hardcode)
   - Include clear setup instructions in README
   - Organize code into logical modules (document processing, embedding, retrieval, UI)
</requirements>

<implementation>
**Project Structure**:
- Create a modular architecture separating concerns:
  - Document processing module
  - Embedding generation module
  - Vector storage/retrieval module
  - LLM query module
  - Streamlit UI module

**Best Practices**:
- Use type hints throughout for clarity
- Implement error handling for file operations, API calls, and model loading
- Add progress indicators for long-running operations (document processing, embedding generation)
- Cache embeddings to avoid reprocessing the same documents
- Use session state in Streamlit to persist data across interactions

**What to Avoid**:
- Don't hardcode API keys - use environment variables and provide clear instructions on where to set them
- Don't load models on every query - initialize once and reuse (explain WHY: model loading is expensive and slows down queries)
- Don't process the same document multiple times - implement deduplication logic
- Don't skip error messages - user needs to know if something fails and why
</implementation>

<output>
Create the following files with relative paths:

- `./pyproject.toml` - UV project configuration with all dependencies
- `./src/document_processor.py` - Docling integration for PDF processing and chunking
- `./src/embeddings.py` - BGE model loading and embedding generation
- `./src/vector_store.py` - ChromaDB integration for storage and retrieval
- `./src/llm_client.py` - Anthropic API client for Claude queries
- `./src/rag_pipeline.py` - Main RAG pipeline orchestration
- `./app.py` - Streamlit application
- `./.env.example` - Example environment variables file
- `./README.md` - Setup instructions, usage guide, and architecture overview
- `./.gitignore` - Python and environment-specific ignores

Ensure all imports are correct and the application can run with: `uv run streamlit run app.py`
</output>

<verification>
Before declaring complete, verify your work:

1. **Dependencies Check**:
   - Run `!uv sync` to ensure all dependencies install correctly
   - Verify pyproject.toml includes all required packages

2. **Code Quality**:
   - Check that all modules have proper imports
   - Ensure type hints are present on functions
   - Verify error handling exists for external API calls

3. **Configuration**:
   - Confirm .env.example contains all required environment variables
   - Verify .gitignore excludes sensitive files (.env, __pycache__, etc.)

4. **Documentation**:
   - README includes step-by-step setup instructions
   - Architecture overview explains how components interact
   - Usage examples are clear and actionable

5. **Integration Points**:
   - Document processor outputs format compatible with embedding module
   - Vector store can handle the metadata structure from document processor
   - LLM client properly formats retrieved chunks as context
</verification>

<success_criteria>
- UV project is properly configured with all dependencies
- PDF documents can be uploaded and processed through Docling
- Embeddings are generated using BGE model and stored in ChromaDB
- Semantic search retrieves relevant chunks based on queries
- Claude API is called with proper context and returns answers
- Streamlit UI is functional with file upload, query input, and results display
- Source citations link answers back to original documents
- Code is modular, well-organized, and follows Python best practices
- README provides clear setup and usage instructions
- Application runs without errors using `uv run streamlit run app.py`
</success_criteria>
