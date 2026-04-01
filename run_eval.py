"""
RAGAS Evaluation Runner
Runs evaluation pipeline on the RAG system.

Usage:
    python run_eval.py --test             # 1 query, 1 config — verify pipeline works
    python run_eval.py --generate-only    # Generate answers + cache (no RAGAS eval, saves credits)
    python run_eval.py --eval-only        # Run RAGAS on cached samples (no LLM generation)
    python run_eval.py                    # Default: quick eval (10 queries, 1 config)
    python run_eval.py --quick            # Retrieval-only eval (no RAGAS LLM needed)
    python run_eval.py --full             # Full RAGAS eval on all 4 retrieval strategies
    python run_eval.py --all              # Full experiment matrix (all models × strategies)
    python run_eval.py --regenerate       # Clear cached samples and regenerate
    python run_eval.py --generate-queries # Generate test queries JSON only

Recommended workflow:
    1. python run_eval.py --test               # Verify pipeline end-to-end
    2. python run_eval.py --all --generate-only # Generate all answers (uses Groq, free)
    3. python run_eval.py --all --eval-only     # Score with RAGAS (uses Vertex AI credits)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.evaluation.evaluator import (
    run_evaluation,
    quick_evaluate,
    generate_test_queries,
)

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
CHROMA_DIR = str(DATA_DIR / "chroma_db")
RESULTS_DIR = DATA_DIR / "results"
TEST_QUERIES_FILE = DATA_DIR / "eval" / "test_queries.json"


def main():
    args = sys.argv[1:]

    # Check prerequisites
    if not CHUNKS_DIR.exists() or not list(CHUNKS_DIR.glob("*.json")):
        print("❌ No chunk files found.")
        print("   Run the pipeline first:")
        print("   1. python run_ingestion.py")
        print("   2. python run_embeddings.py")
        return

    # Parse mode flags
    generate_only = "--generate-only" in args
    eval_only = "--eval-only" in args

    # Handle --regenerate: clear cache first
    if "--regenerate" in args:
        import shutil
        cache_dir = RESULTS_DIR / "samples_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("🗑️  Cleared samples cache\n")

    if "--generate-queries" in args:
        generate_test_queries(TEST_QUERIES_FILE)
        return

    if "--test" in args:
        # Minimal test: 1 query, 1 config — just verify the pipeline works
        print("🧪 Test mode: 1 query, 1 config — verifying pipeline works\n")
        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            embedding_models=["all-MiniLM-L6-v2"],
            chunk_strategies=["recursive"],
            retrieval_strategies=["hybrid_rerank"],
            max_queries=1,
            generate_only=generate_only,
            eval_only=eval_only,
        )
        return

    if "--quick" in args:
        # Retrieval-only evaluation — no LLM needed for eval
        print("⚡ Quick mode: retrieval metrics only (no RAGAS LLM needed)\n")
        quick_evaluate(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
        )

    elif "--all" in args:
        # Full experiment matrix
        mode_label = "generate-only" if generate_only else ("eval-only" if eval_only else "full")
        print(f"🔬 Full matrix: 4 embeddings × 3 chunking × 4 retrieval ({mode_label})\n")
        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            embedding_models=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "BAAI/bge-large-en-v1.5",
                "text-embedding-005",
            ],
            chunk_strategies=[
                "fixed", "recursive", 
                # "semantic"
                ],
            retrieval_strategies=["dense_only", "bm25_only", "hybrid", "hybrid_rerank"],
            generate_only=generate_only,
            eval_only=eval_only,
        )

    elif "--full" in args:
        # Full RAGAS on all retrieval strategies (single embedding + chunking)
        print("📊 Full RAGAS eval: 4 retrieval strategies\n")
        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            embedding_models=["all-MiniLM-L6-v2"],
            chunk_strategies=["recursive"],
            retrieval_strategies=["dense_only", "bm25_only", "hybrid", "hybrid_rerank"],
            generate_only=generate_only,
            eval_only=eval_only,
        )

    else:
        # Default: quick RAGAS eval (10 queries, best config only)
        print("📋 Default eval: 10 queries, hybrid_rerank only\n")
        print("   Use --test for 1-query pipeline test")
        print("   Use --full for all 4 retrieval strategies")
        print("   Use --all for full experiment matrix")
        print("   Add --generate-only to cache answers without RAGAS scoring")
        print("   Add --eval-only to score cached answers without regenerating\n")

        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            max_queries=10,
            generate_only=generate_only,
            eval_only=eval_only,
        )

    print("\n✅ Evaluation complete!")
    print("   View results: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()