# Architecture and Design Decisions

This document explains the technical architecture and key design decisions for the RAG Document Q&A system.

## System Overview

The RAG system follows a classic pipeline architecture:

```
PDF Upload → Document Processing → Embedding Generation → Vector Storage
                                                                ↓
                                                          Index Created
                                                                ↓
User Query → Query Embedding → Semantic Search → Context Retrieval → LLM Query → Answer
```

## Component Architecture

### 1. Document Processor (`src/document_processor.py`)

**Purpose**: Transform PDF documents into searchable text chunks

**Key Design Decisions**:

- **Docling Integration**: Uses Docling for robust PDF parsing
  - Why: Better handling of complex PDFs than PyPDF2
  - Extracts text while preserving structure
  - Converts to markdown for consistent formatting

- **Chunking Strategy**:
  - Default: 1000 characters per chunk with 100-character overlap
  - Why overlap: Maintains context at boundaries while minimizing duplicates
  - Smart splitting: Breaks at paragraph/sentence boundaries
  - Preserves metadata: filename, page, position

- **Chunk Quality Filtering** (`ChunkValidator`):
  - Filters low-quality chunks before storage
  - Minimum length: 100 characters (removes short fragments)
  - Minimum meaningful ratio: 30% alphanumeric content
  - Maximum special characters: 70% of content
  - Whitelisted patterns: Code blocks, function definitions
  - Tracks filtering statistics for monitoring

- **Data Structure** (`DocumentChunk`):
  ```python
  @dataclass
  class DocumentChunk:
      text: str           # The actual text content
      source: str         # Original filename
      page: int          # Estimated page number
      chunk_index: int   # Sequential index
      metadata: Dict     # Additional context
  ```

### 2. Embedding Generator (`src/embeddings.py`)

**Purpose**: Convert text into vector representations for similarity search

**Key Design Decisions**:

- **Model Choice**: BAAI/bge-small-en-v1.5
  - Why: Good balance of quality and speed
  - 384 dimensions (smaller = faster search)
  - Strong performance on semantic similarity tasks
  - Reasonable model size (~100MB)

- **Model Loading**:
  - Loaded once at initialization
  - Cached in memory for reuse
  - Why: Model loading takes 5-10 seconds
  - Trade-off: Higher memory usage for better performance

- **Batch Processing**:
  - Processes multiple texts together
  - Configurable batch size (default: 32)
  - Progress bar for large batches
  - Why: 10-50x faster than sequential processing

### 3. Vector Store (`src/vector_store.py`)

**Purpose**: Store and retrieve document embeddings efficiently

**Key Design Decisions**:

- **ChromaDB Choice**:
  - Why: Simple, fast, no server required
  - Local persistence (data survives restarts)
  - Built-in similarity search
  - Good Python integration

- **Storage Structure**:
  - Collections: Logical grouping of documents
  - Documents: Text chunks with metadata
  - Embeddings: Vector representations
  - IDs: Auto-generated unique identifiers

- **Search Method**:
  - Cosine similarity (default in ChromaDB)
  - Returns top-k most similar chunks
  - Includes distance scores (lower = more similar)
  - Metadata filtering support

- **Retrieval Quality Enhancements**:

  **MMR (Maximal Marginal Relevance)**:
  - Algorithm that balances relevance and diversity
  - Formula: `MMR = λ * relevance - (1-λ) * max_similarity_to_selected`
  - Fetches 3x candidate chunks for reranking
  - Iteratively selects chunks that are:
    1. Relevant to the query (high similarity)
    2. Different from already-selected chunks (low similarity)
  - λ parameter controls balance:
    - λ=1.0: Pure relevance (like standard search)
    - λ=0.5: Balanced (default)
    - λ=0.0: Pure diversity
  - Implementation uses vectorized numpy operations for efficiency
  - Why: Prevents retrieving 5 nearly-identical chunks from same section

  **Deduplication**:
  - Uses SequenceMatcher for text similarity comparison
  - Threshold: 80% similarity = duplicate
  - Sequential comparison: each candidate vs. already-selected docs
  - Applied after MMR reranking
  - Always enabled for better results
  - Why: Removes chunks with repeated content from page overlaps

### 4. LLM Client (`src/llm_client.py`)

**Purpose**: Query Claude with retrieved context

**Key Design Decisions**:

- **Model Selection**:
  - Default: Claude 3.5 Sonnet
  - Alternative: Claude 3.5 Haiku (faster, cheaper)
  - Why Sonnet: Best quality for question-answering

- **Prompt Engineering**:
  - Explicit instructions to use only provided context
  - Source citations included in context
  - Structured format for consistency
  - Temperature 0 for deterministic answers

- **Context Formatting**:
  ```
  [Source 1: filename.pdf, Page 3, Chunk 5]
  <chunk text>
  ---
  [Source 2: ...]
  ```

- **Error Handling**:
  - API errors caught and logged
  - Rate limiting respected
  - Timeouts configured appropriately

### 5. RAG Pipeline (`src/rag_pipeline.py`)

**Purpose**: Orchestrate all components into a cohesive system

**Key Design Decisions**:

- **Component Initialization**:
  - All components loaded at startup
  - Single pipeline instance per session
  - Why: Avoid repeated model loading

- **Document Deduplication**:
  - Hash-based tracking of processed documents
  - Prevents reprocessing same file
  - Based on filename + size + modified time

- **Indexing Workflow**:
  1. Process PDFs → Extract chunks
  2. Generate embeddings (batch)
  3. Store in vector database
  4. Update processed document registry

- **Query Workflow**:
  1. Generate query embedding
  2. Search vector store for similar chunks (3x if MMR enabled)
  3. Apply MMR reranking (optional, default: enabled)
     - Balances relevance and diversity
     - Returns top-k from candidates
  4. Apply deduplication (always enabled)
     - Removes near-duplicate chunks
     - Ensures variety in context
  5. Format chunks as context
  6. Query Claude with context
  7. Return answer with sources

### 6. Web Interface (`app.py`)

**Purpose**: Provide user-friendly access to RAG functionality

**Key Design Decisions**:

- **Streamlit Framework**:
  - Why: Rapid development for internal tools
  - No frontend code needed
  - Built-in components for common UI patterns
  - Session state for persistence

- **Session Management**:
  - Pipeline initialized once per session
  - Stored in `st.session_state`
  - Survives page interactions

- **Progress Feedback**:
  - Progress bars for long operations
  - Status messages for user awareness
  - Error messages with actionable info

- **Result Display**:
  - Answer prominently displayed
  - Sources in expandable sections
  - Relevance scores shown
  - Token usage for cost tracking

## Data Flow

### Indexing Flow

```
User uploads PDF
    ↓
Save to temp storage
    ↓
Docling parses PDF → Extract text
    ↓
Split into chunks (with 100-char overlap)
    ↓
Apply chunk quality filtering
    ├── Filter: too_short (< 100 chars)
    ├── Filter: table_artifact (| | ?)
    ├── Filter: low_meaningful_ratio (< 30% alphanumeric)
    └── Filter: high_special_char_ratio (> 70% special)
    ↓
Generate embeddings (batch, only valid chunks)
    ↓
Store in ChromaDB
    ↓
Update UI with success message + filtering stats
```

### Query Flow

```
User enters question
    ↓
Generate query embedding
    ↓
Search ChromaDB (cosine similarity)
    ├── If MMR enabled: fetch 3x candidates (e.g., 15 for top_k=5)
    └── If MMR disabled: fetch top_k
    ↓
Apply MMR reranking (if enabled)
    ├── Calculate relevance scores (similarity to query)
    ├── Calculate diversity scores (dissimilarity to selected)
    ├── Iteratively select: MMR = λ*relevance - (1-λ)*max_similarity
    └── Return top_k with best MMR scores
    ↓
Apply deduplication (always enabled)
    ├── Compare each chunk with selected chunks
    ├── Filter if 80%+ text similarity
    └── Return unique chunks only
    ↓
Format as context with source citations
    ↓
Send to Claude API with system prompt
    ↓
Receive answer
    ↓
Display with source citations + relevance scores
```

## Performance Optimizations

### 1. Model Caching
- Embedding model loaded once
- Reused across all operations
- Trades memory for speed

### 2. Batch Processing
- Embeddings generated in batches
- Significantly faster than sequential
- Configurable batch size

### 3. Vector Search
- ChromaDB optimized for similarity search
- Fast retrieval even with many documents
- Approximate nearest neighbor search

### 4. Smart Chunking
- Reduced overlap (100 chars) minimizes duplicates
- Boundary-aware splitting
- Metadata preserved for tracking
- Quality filtering removes low-value chunks

### 5. Retrieval Quality
- MMR uses vectorized numpy operations for speed
- Deduplication uses fast SequenceMatcher algorithm
- Always-on deduplication has minimal overhead
- MMR candidate fetching (3x) amortized by quality gain

## Configuration Parameters

### Document Processing
```python
chunk_size = 1000        # Characters per chunk
chunk_overlap = 100      # Overlap between chunks (reduced from 200)

# Chunk quality filtering thresholds
min_length = 100                 # Minimum chunk length in characters
min_meaningful_ratio = 0.3       # Minimum alphanumeric content ratio
max_special_char_ratio = 0.7     # Maximum special character ratio
```

**Tuning Advice**:
- Larger chunks: More context, fewer chunks, slower search
- Smaller chunks: More precise, more chunks, faster search
- Overlap: 10-20% of chunk_size is typical

### Embedding Generation
```python
embedding_model = "BAAI/bge-small-en-v1.5"
batch_size = 32
```

**Tuning Advice**:
- Larger batch: Faster processing, more memory
- Smaller batch: Slower processing, less memory

### Vector Search
```python
top_k = 5                # Number of chunks to retrieve
use_mmr = True           # Enable MMR for diverse results (default)
mmr_lambda = 0.5         # Balance: 0=diversity, 1=relevance
deduplicate = True       # Always enabled, removes near-duplicates
similarity_threshold = 0.8  # Deduplication threshold (80% similarity)
```

**Tuning Advice**:
- More chunks: Better context, slower, more tokens
- Fewer chunks: Faster, cheaper, might miss context
- MMR lambda tuning:
  - λ=0.7-1.0: Prioritize relevance (accept similar chunks)
  - λ=0.3-0.5: Balanced (default: 0.5)
  - λ=0.0-0.2: Prioritize diversity (maximum variety)
- Deduplication threshold:
  - 0.8 (default): Removes very similar chunks
  - 0.9: More aggressive, only removes near-identical
  - 0.7: Less aggressive, removes moderately similar

### LLM Configuration
```python
model = "claude-3-5-sonnet-20241022"
max_tokens = 2048
temperature = 0.0
```

**Tuning Advice**:
- Sonnet: Best quality, slower, expensive
- Haiku: Good quality, faster, cheaper
- Temperature 0: Deterministic, factual
- Temperature 0.7: More creative, varied

## Retrieval Quality Algorithms

### Chunk Quality Filtering

**Purpose**: Remove low-quality chunks before storage to improve search accuracy

**Implementation** (`src/chunk_validator.py`):
```python
class ChunkValidator:
    def is_valid_chunk(self, text: str) -> Tuple[bool, str]:
        # 1. Check whitelisted patterns (code blocks, functions)
        if matches_whitelist(text):
            return True, "whitelisted"

        # 2. Check minimum length (filters short fragments)
        if len(text) < min_length:
            return False, f"too_short({len(text)})"

        # 3. Check table artifacts (e.g., "| | ?")
        if re.match(r'^[\|\s\?\-]+$', text):
            return False, "table_artifact"

        # 4. Check meaningful content ratio
        alphanumeric_ratio = count_alnum(text) / len(text)
        if alphanumeric_ratio < min_meaningful_ratio:
            return False, f"low_meaningful_ratio({ratio})"

        # 5. Check special character ratio
        special_ratio = count_special(text) / len(text)
        if special_ratio > max_special_char_ratio:
            return False, f"high_special_char_ratio({ratio})"

        return True, "valid"
```

**Statistics Tracking**:
- Total chunks created
- Chunks filtered (by reason)
- Filter rate percentage
- Displayed in UI sidebar for monitoring

### MMR (Maximal Marginal Relevance)

**Purpose**: Balance relevance and diversity to prevent repetitive results

**Algorithm**:
```python
def apply_mmr(query_embedding, documents, embeddings, lambda_param):
    # 1. Fetch 3x candidate chunks from vector store
    candidates = search(query_embedding, n_results=top_k * 3)

    # 2. Normalize all vectors for cosine similarity
    query_vec = normalize(query_embedding)
    doc_vecs = normalize(embeddings)

    # 3. Calculate relevance scores (1 - distance)
    relevance_scores = 1 - distances

    # 4. Iteratively select top_k documents
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        mmr_scores = []
        for idx in remaining:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component
            if selected:
                # Max similarity to already-selected docs
                similarities = dot(doc_vecs[selected], doc_vecs[idx])
                max_sim = max(similarities)
            else:
                max_sim = 0

            # MMR formula
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr_score))

        # Select document with highest MMR score
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return reorder_results(documents, selected)
```

**Key Points**:
- λ=1.0: Pure relevance (standard semantic search)
- λ=0.5: Balanced (default)
- λ=0.0: Pure diversity (maximum variety)
- Uses vectorized numpy operations for efficiency
- Iterative greedy algorithm (optimal for λ=0.5)

**Trade-offs**:
- Fetches 3x chunks: Slightly more compute
- Reranking overhead: ~50-100ms for 15 chunks
- Quality improvement: Significant (prevents near-duplicates)
- Net impact: Positive (better answers worth the cost)

### Deduplication

**Purpose**: Remove near-duplicate chunks from retrieval results

**Algorithm**:
```python
def filter_near_duplicates(documents, threshold=0.8):
    unique_docs = []

    for doc in documents:
        is_duplicate = False

        # Compare with already-selected unique documents
        for unique_doc in unique_docs:
            # Calculate text similarity using SequenceMatcher
            similarity = SequenceMatcher(None, doc, unique_doc).ratio()

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(doc)

    return unique_docs
```

**Key Points**:
- Uses `difflib.SequenceMatcher` for text comparison
- Threshold: 0.8 (80% similarity = duplicate)
- Sequential comparison: O(n²) worst case
- In practice: Fast for small n (typically 5-10 chunks)
- Always enabled (no configuration needed)

**Why 80% Threshold**:
- Catches near-duplicates from page overlaps
- Allows minor variations (different examples, context)
- Tested on production data (good balance)

### Integration Flow

**During Indexing**:
```
PDF → Extract text → Create chunks → Validate each chunk → Store valid chunks
                                          ↓
                              Filter: too_short, table_artifact,
                                      low_content, high_special_chars
```

**During Retrieval**:
```
Query → Generate embedding → Search (3x if MMR) → Apply MMR → Apply dedup → Return
                                                      ↓              ↓
                                              Rerank for      Remove 80%+
                                              diversity       similar
```

## Security Considerations

### API Keys
- Stored in environment variables only
- Never committed to version control
- `.env` in `.gitignore`

### File Handling
- PDFs stored temporarily
- Automatic cleanup on errors
- No permanent storage of uploads

### Data Privacy
- All data stored locally
- ChromaDB not shared externally
- No telemetry by default

## Error Handling

### Graceful Degradation
- Component failures don't crash system
- Error messages displayed to user
- Logging for debugging

### Recovery Strategies
- API rate limiting handled
- Network errors retried
- Partial results returned when possible

## Extension Points

### Adding New Document Types
1. Create parser in `document_processor.py`
2. Ensure output matches `DocumentChunk` format
3. Update UI to accept new file types

### Changing Embedding Models
1. Update model name in `embeddings.py`
2. Ensure compatible with ChromaDB
3. Clear and rebuild index

### Adding New LLM Providers
1. Create client in `llm_client.py`
2. Match interface of existing client
3. Update pipeline configuration

### Custom Metadata
1. Extend `DocumentChunk` dataclass
2. Update metadata extraction in processor
3. Use in vector store filtering

## Testing Recommendations

### Unit Tests
- Test each component independently
- Mock external dependencies
- Focus on edge cases

### Integration Tests
- Test component interactions
- Use small test documents
- Verify end-to-end flow

### Performance Tests
- Benchmark chunking speed
- Measure embedding generation time
- Monitor query latency

## Known Limitations

1. **PDF Parsing**: Complex layouts may not parse perfectly
2. **Language**: Optimized for English text
3. **Context Window**: Limited by Claude's context size
4. **Scalability**: Not optimized for millions of documents
5. **Concurrency**: Single-user session model

## Future Improvements

### Short Term
- Add conversation history
- Support more file formats
- Improve error messages

### Medium Term
- Multi-user support
- Advanced metadata filtering
- Custom prompt templates

### Long Term
- Distributed vector store
- Multi-language support
- Advanced chunking strategies

## References

- [Docling Documentation](https://github.com/DS4SD/docling)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [Claude API Docs](https://docs.anthropic.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Maintenance Notes

### Regular Tasks
- Monitor API usage and costs
- Update dependencies quarterly
- Review and update prompts
- Check for model updates

### Troubleshooting
- Check logs first
- Verify API key validity
- Ensure disk space available
- Monitor memory usage

### Performance Monitoring
- Track query response times
- Monitor indexing speed
- Watch memory usage
- Log error rates
