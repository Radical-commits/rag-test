"""Quick verification script to check if all components can be imported."""

import sys
from pathlib import Path

def verify_imports():
    """Test that all modules can be imported successfully."""
    print("Verifying RAG system setup...\n")

    checks = []

    # Check 1: Document Processor
    try:
        from src.document_processor import DocumentProcessor, DocumentChunk
        checks.append(("Document Processor", True, "OK"))
    except Exception as e:
        checks.append(("Document Processor", False, str(e)))

    # Check 2: Embeddings
    try:
        from src.embeddings import EmbeddingGenerator
        checks.append(("Embedding Generator", True, "OK"))
    except Exception as e:
        checks.append(("Embedding Generator", False, str(e)))

    # Check 3: Vector Store
    try:
        from src.vector_store import VectorStore
        checks.append(("Vector Store", True, "OK"))
    except Exception as e:
        checks.append(("Vector Store", False, str(e)))

    # Check 4: LLM Client
    try:
        from src.llm_client import LLMClient
        checks.append(("LLM Client", True, "OK"))
    except Exception as e:
        checks.append(("LLM Client", False, str(e)))

    # Check 5: RAG Pipeline
    try:
        from src.rag_pipeline import RAGPipeline
        checks.append(("RAG Pipeline", True, "OK"))
    except Exception as e:
        checks.append(("RAG Pipeline", False, str(e)))

    # Check 6: Environment file
    env_file = Path(".env")
    if env_file.exists():
        checks.append((".env file", True, "Found"))
    else:
        checks.append((".env file", False, "Not found - copy .env.example to .env"))

    # Print results
    print("=" * 60)
    print("IMPORT VERIFICATION RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed, message in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {name:25s} | {message}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Make sure you have set ANTHROPIC_API_KEY in your .env file")
        print("2. Run: uv run streamlit run app.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nTry running: uv sync")
        return 1

if __name__ == "__main__":
    sys.exit(verify_imports())
