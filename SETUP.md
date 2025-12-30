# RAG System Setup Guide

Complete setup instructions for the RAG Document Q&A system with support for both direct Anthropic API and corporate LiteLLM proxy.

## Prerequisites

Before starting, ensure you have:
- **Python 3.12 or higher**
  ```bash
  python --version  # Should show 3.12.x or higher
  ```
- **UV package manager**
  ```bash
  uv --version  # Install from https://docs.astral.sh/uv/
  ```
- **API Access** - One of:
  - Anthropic API key from https://console.anthropic.com/
  - OR LiteLLM proxy credentials from your IT team

## Quick Setup (5 Minutes)

### Step 1: Install Dependencies

```bash
uv sync
```

This installs all required Python packages (~500MB with models).

### Step 2: Download Embedding Model

Due to corporate SSL certificate issues, the BGE embedding model must be downloaded manually.

Run this one-line Python command:

```bash
uv run python -c "import os,ssl,requests,urllib3;from pathlib import Path;from tqdm import tqdm;os.environ.update({'CURL_CA_BUNDLE':'','REQUESTS_CA_BUNDLE':'','SSL_CERT_FILE':'','PYTHONHTTPSVERIFY':'0','HF_HUB_DISABLE_SSL_VERIFY':'1'});ssl._create_default_https_context=ssl._create_unverified_context;urllib3.disable_warnings();og=requests.get;requests.get=lambda u,**k:og(u,**{**k,'verify':False});BASE='https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main';FILES=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','sentence_bert_config.json','modules.json','config_sentence_transformers.json','README.md','1_Pooling/config.json'];[[(print(f'Downloading {f}...'),p:=Path(f'models/bge-small-en-v1.5/{f}'),p.parent.mkdir(parents=True,exist_ok=True),r:=requests.get(f'{BASE}/{f}',stream=True,verify=False),r.raise_for_status(),[(fp:=open(p,'wb')),(bar:=tqdm(total=int(r.headers.get('content-length',0)),unit='B',unit_scale=True))]+[fp.write(chunk) or bar.update(len(chunk)) for chunk in r.iter_content(8192) if chunk]+[fp.close(),bar.close()],print(f'✓ {f}'))] for f in FILES];print('✓ All files downloaded!')"
```

**Verify the download:**
```bash
ls -lh ./models/bge-small-en-v1.5/model.safetensors
```
The file should be ~127 MB. If smaller or missing, re-run the download command.

**Why manual download?** Corporate SSL inspection uses self-signed certificates that Python's SSL libraries don't trust. This bypass command disables SSL verification to download the model files.

### Step 3: Configure API Access

Create your environment configuration file:

```bash
cp .env.example .env
```

Edit `.env` and choose ONE of the following configurations:

#### Option A: Direct Anthropic API

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

Get your API key from: https://console.anthropic.com/

#### Option B: Corporate LiteLLM Proxy

```bash
LITELLM_API_KEY=sk-your-litellm-token
LITELLM_BASE_URL=https://your-proxy.company.com/anthropic
CLAUDE_MODEL_NAME=bedrock-claude-4.5-sonnet
```

**Notes:**
- Contact your IT/DevOps team for your LiteLLM credentials
- Base URL should end with `/anthropic`
- Model name must match your LiteLLM configuration (ask your admin)
- No quotes needed around values

### Step 4: Test Your Configuration

Run the verification script:

```bash
uv run python verify_setup.py
```

Expected output:
```
✓ All imports successful
✓ Environment file exists
✓ Configuration looks good
```

### Step 5: Launch the Application

```bash
uv run streamlit run app.py
```

The application will open in your browser at http://localhost:8501

## First Use

Once the application is running:

1. **Upload a PDF**
   - Click "Browse files" in the Document Upload section
   - Select a small PDF (1-3 pages recommended for first test)
   - Click "Index Documents"

2. **Wait for Processing**
   - Progress bar shows indexing status
   - Takes ~5-10 seconds per page
   - Success message shows number of chunks indexed

3. **Ask a Question**
   - Scroll to "Ask Questions" section
   - Type a question about your document
   - Click "Ask" button
   - View answer with source citations

## Verification Checklist

Use this checklist to verify your setup:

### Installation
- [ ] Python 3.12+ installed
- [ ] UV package manager installed
- [ ] Dependencies installed (`uv sync` completed)
- [ ] Embedding model downloaded (~127 MB file exists)
- [ ] `.env` file created and configured
- [ ] Verify script passes all checks

### First Run
- [ ] Application starts without errors
- [ ] "System initialized successfully!" appears
- [ ] System statistics shown in sidebar
- [ ] No import or configuration errors

### Functionality Test
- [ ] PDF upload works (1-2 page test document)
- [ ] Document indexes successfully
- [ ] Sidebar shows document count > 0
- [ ] Query returns answer
- [ ] Sources displayed with citations
- [ ] Relevance scores shown

## Troubleshooting

### API Configuration Issues

**Error: "No API configuration found"**
- Check `.env` file exists in project root (not `.env.txt`)
- Verify you added either ANTHROPIC_API_KEY or both LITELLM_API_KEY + LITELLM_BASE_URL
- No quotes or extra spaces around values
- Restart the application after editing `.env`

**Error: "Authentication error" or "Invalid API key"**
- For Anthropic: Verify key starts with `sk-ant-api03-`
- For LiteLLM: Check token with IT team, may be expired
- Ensure no typos when copying the key
- Check for accidental spaces or newlines

### Model Download Issues

**Error: "Failed to load embedding model" or model not found**
- Verify model directory exists: `ls ./models/bge-small-en-v1.5/`
- Check model file size: `ls -lh ./models/bge-small-en-v1.5/model.safetensors`
- Should be ~127 MB - if smaller, download is incomplete
- Delete `./models/` directory and re-run download command

**SSL certificate errors during download**
- This is expected and the command handles it automatically
- If download fails, check internet connection
- Try the download command again (it's safe to re-run)

### PDF Processing Issues

**Error: "Failed to process PDF"**
- Ensure PDF is not corrupted (can open in PDF reader)
- Check PDF is not password-protected
- Try a simpler PDF first
- Some complex PDFs with unusual formatting may fail

**Error: "CAS service error" or connection timeouts**
- This is fixed in the code with `HF_HUB_DISABLE_XET=1`
- Restart the application if you see this
- Should not occur in current version

**ChromaDB batch size error (10,000+ chunks)**
- This is fixed - the system automatically batches large documents
- If you see this error, you may be running old code
- Check `src/vector_store.py` has batch_size parameter

### Application Startup Issues

**Import errors or missing modules**
```bash
# Reinstall dependencies
uv sync

# Verify installation
uv run python verify_setup.py
```

**Port already in use (8501)**
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
uv run streamlit run app.py --server.port 8502
```

**Memory errors**
- Reduce document chunk size in `src/rag_pipeline.py`
- Process fewer documents at once
- Ensure at least 2GB RAM available

### LiteLLM Proxy Specific Issues

**Error: "Connection error" or timeout**
- Verify `LITELLM_BASE_URL` is correct
- Check if VPN connection is required
- Confirm URL ends with `/anthropic`
- Test connectivity: `curl -v https://your-proxy.company.com`

**Error: "Model not found"**
- Model name in `CLAUDE_MODEL_NAME` must match your LiteLLM config
- Common names: `bedrock-claude-4.5-sonnet`, `anthropic/claude-3-5-sonnet`
- Ask your IT/DevOps team for the exact model identifier

**Test script fails but manual curl works**
- Python SSL/TLS configuration issue
- Already handled by SSL bypass in code
- If persists, check: `python -c "import ssl; print(ssl.OPENSSL_VERSION)"`

## Configuration Options

### Environment Variables

Complete list of supported variables in `.env`:

```bash
# Direct Anthropic API (Option 1)
ANTHROPIC_API_KEY=sk-ant-api03-your-key

# LiteLLM Proxy (Option 2)
LITELLM_API_KEY=sk-your-litellm-token
LITELLM_BASE_URL=https://your-proxy.company.com/anthropic
CLAUDE_MODEL_NAME=bedrock-claude-4.5-sonnet

# Optional: Override model for direct Anthropic
CLAUDE_MODEL_NAME=claude-3-5-haiku-20241022  # Faster, cheaper alternative
```

### Performance Tuning

Adjust these parameters in `src/rag_pipeline.py` initialization:

```python
RAGPipeline(
    chunk_size=1000,              # Characters per chunk (500-2000)
    chunk_overlap=200,            # Overlap between chunks (100-500)
    embedding_model="./models/bge-small-en-v1.5",
    chroma_db_path="./chroma_db"
)
```

**Tuning advice:**
- **Larger chunks** (1500-2000): More context, fewer chunks, slower search
- **Smaller chunks** (500-800): More precise, faster search, may lose context
- **Overlap**: 10-20% of chunk_size prevents losing context at boundaries

### Query Configuration

Adjust in the Streamlit UI:
- **Number of sources**: Default 5, increase for more context (slower, more tokens)
- Shown in "Ask Questions" section

### Model Selection

**For direct Anthropic API:**
- Sonnet 3.5: Best quality, ~$3 per million tokens
- Haiku 3.5: Good quality, faster, ~$0.25 per million tokens

Set in `.env`:
```bash
CLAUDE_MODEL_NAME=claude-3-5-haiku-20241022
```

**For LiteLLM proxy:**
- Use model names configured by your admin
- Ask IT/DevOps for available models

## Performance Benchmarks

Expected performance on typical hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| First launch | 10-20s | Model loading |
| Document indexing | 5-10s per page | Depends on complexity |
| Query response | 2-5s | Depends on model choice |
| Memory usage | 500MB-1GB | Includes loaded models |

## Advanced: Manual Model Download (Alternative)

If the Python one-liner fails, use git:

```bash
mkdir -p models
cd models
GIT_SSL_NO_VERIFY=true git clone https://huggingface.co/BAAI/bge-small-en-v1.5
cd ..
```

Verify: `ls ./models/bge-small-en-v1.5/`

## Security Notes

- API keys stored in `.env` only (git-ignored)
- Never commit `.env` to version control
- All data stored locally in `./chroma_db/`
- No telemetry or external data sharing
- SSL verification disabled for corporate proxies (by design)

## Getting Help

If you encounter issues:

1. Check this troubleshooting section first
2. Review error messages in terminal
3. Run `uv run python verify_setup.py` for diagnostics
4. Check the logs: `tail -f ./logs/llm_interactions.log`
5. Consult README.md for detailed documentation
6. Contact development team with:
   - Error message (full text)
   - Steps to reproduce
   - Output from verify_setup.py

## Next Steps

After successful setup:
- Read README.md for usage details
- Check ARCHITECTURE.md for technical deep-dive
- Upload your documents and start asking questions
- Monitor token usage in the UI (for cost tracking)
- Review LLM interaction logs in `./logs/llm_interactions.log`

## Quick Command Reference

```bash
# Start the application
uv run streamlit run app.py

# Verify installation
uv run python verify_setup.py

# Install/update dependencies
uv sync

# Stop the application
# Press Ctrl+C in terminal

# Clear the vector database (start fresh)
# Use "Clear Index" button in the UI

# View LLM interaction logs
tail -f ./logs/llm_interactions.log
```
