"""
Ingestion Pipeline Runner
Runs the full ingestion pipeline: PDF extraction → chunking (3 strategies)

Usage:
    python run_ingestion.py                  # Run full pipeline
    python run_ingestion.py --extract-only   # Only extract text from PDFs
    python run_ingestion.py --chunk-only     # Only run chunking (needs extracted docs)
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_extractor import process_all_pdfs
from src.ingestion.chunker import chunk_documents

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
METADATA_FILE = DATA_DIR / "paper_metadata.json"
EXTRACTED_FILE = DATA_DIR / "processed" / "extracted_documents.json"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"


def run_extraction():
    """Step 1: Extract text from PDFs."""
    print("=" * 60)
    print("STEP 1: PDF TEXT EXTRACTION")
    print("=" * 60)

    meta_path = METADATA_FILE if METADATA_FILE.exists() else None
    documents = process_all_pdfs(PDF_DIR, meta_path, EXTRACTED_FILE)

    if not documents:
        print("\n❌ No documents extracted. Check that PDFs are in data/raw_pdfs/")
        return None

    return documents


def run_chunking():
    """Step 2: Chunk documents with all 3 strategies."""
    print("\n" + "=" * 60)
    print("STEP 2: TEXT CHUNKING (3 STRATEGIES)")
    print("=" * 60)

    # Load extracted documents
    if not EXTRACTED_FILE.exists():
        print(f"❌ No extracted documents found at {EXTRACTED_FILE}")
        print("   Run extraction first: python run_ingestion.py --extract-only")
        return None

    with open(EXTRACTED_FILE) as f:
        documents = json.load(f)

    print(f"📄 Loaded {len(documents)} extracted documents")

    # Run all chunking strategies
    results = chunk_documents(documents, output_dir=CHUNKS_DIR)
    return results


def main():
    args = sys.argv[1:]

    if "--extract-only" in args:
        run_extraction()
    elif "--chunk-only" in args:
        run_chunking()
    else:
        # Full pipeline
        print("🚀 RAG INGESTION PIPELINE")
        print("=" * 60)
        print(f"   PDF directory:  {PDF_DIR}")
        print(f"   Output:         {CHUNKS_DIR}")
        print()

        documents = run_extraction()
        if documents:
            run_chunking()

        print("\n✅ Ingestion pipeline complete!")
        print("   Next step: python run_embeddings.py")


if __name__ == "__main__":
    main()
