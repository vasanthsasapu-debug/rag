"""
Embedding Pipeline Runner
Builds ChromaDB vector stores for all embedding model × chunking strategy combinations.

Usage:
    python run_embeddings.py            # Build all 3 local models × 3 strategies (9 collections)
    python run_embeddings.py --vertex   # Include Vertex AI embedding (4 models × 3 = 12 collections)
    python run_embeddings.py --quick    # Quick test: MiniLM × recursive only
    python run_embeddings.py --force    # Rebuild everything from scratch
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.embeddings import run_embedding_pipeline

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
CHROMA_DIR = str(DATA_DIR / "chroma_db")


def main():
    force = "--force" in sys.argv
    quick = "--quick" in sys.argv
    include_vertex = "--vertex" in sys.argv

    if not CHUNKS_DIR.exists() or not list(CHUNKS_DIR.glob("*.json")):
        print("❌ No chunk files found.")
        print("   Run the ingestion pipeline first:")
        print("   python run_ingestion.py")
        return

    if quick:
        models = [{"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"}]
        strategies = ["recursive"]
        print("⚡ Quick mode: MiniLM × recursive only\n")
    elif include_vertex:
        models = [
            # {"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"},
            # {"name": "all-mpnet-base-v2", "type": "sentence-transformers"},
            {"name": "BAAI/bge-large-en-v1.5", "type": "sentence-transformers"},
            # {"name": "text-embedding-005", "type": "vertex_ai"},
        ]
        strategies = [
            # "fixed", "recursive", 
                       "semantic"
                       ]
        print("🚀 Full mode with Vertex AI embedding (4 models × 3 strategies = 12 collections)\n")
    else:
        models = [
            # {"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"},
            # {"name": "all-mpnet-base-v2", "type": "sentence-transformers"},
            {"name": "BAAI/bge-large-en-v1.5", "type": "sentence-transformers"},
        ]
        strategies = [
            # "fixed", "recursive", 
                       "semantic"
                       ]

    run_embedding_pipeline(
        chunks_dir=CHUNKS_DIR,
        persist_dir=CHROMA_DIR,
        models=models,
        strategies=strategies,
        force_rebuild=force,
    )

    print("\n✅ Embedding pipeline complete!")
    print("   Next step: python run_rag.py")


if __name__ == "__main__":
    main()