"""
RAG Pipeline Runner — Interactive Query Interface
Test the full RAG pipeline with different configurations.

Usage:
    python run_rag.py                          # Interactive mode
    python run_rag.py --query "What is RAG?"   # Single query
    python run_rag.py --compare                # Compare all strategies on sample queries
"""

import sys
import json
from pathlib import Path
from dataclasses import asdict

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.retriever import Retriever
from src.generation.llm import RAGGenerator

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
CHROMA_DIR = str(DATA_DIR / "chroma_db")


def run_single_query(
    query: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_strategy: str = "recursive",
    retrieval_strategy: str = "hybrid_rerank",
):
    """Run a single query through the full RAG pipeline."""
    print(f"\n{'='*60}")
    print(f"📝 Query: {query}")
    print(f"   Config: {embedding_model} × {chunk_strategy} × {retrieval_strategy}")
    print(f"{'='*60}")

    # Initialize retriever
    retriever = Retriever(
        chroma_dir=CHROMA_DIR,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
        chunks_dir=CHUNKS_DIR,
    )

    # Retrieve
    print("\n🔍 Retrieving relevant chunks...")
    retrieval_output = retriever.retrieve(
        query=query,
        strategy=retrieval_strategy,
        top_k=10,
        rerank_top_k=5,
    )

    print(f"   Found {retrieval_output.num_results} results:")
    for r in retrieval_output.results[:5]:
        print(f"   [{r.rank}] (score: {r.score:.4f}) {r.doc_title[:50]}...")
        print(f"       Section: {r.section_heading}")

    # Generate — Groq primary, Vertex AI fallback
    print("\n🤖 Generating answer...")
    generator = RAGGenerator(primary="groq", fallback="vertex_ai")
    result = generator.generate(
        query=query,
        retrieved_chunks=retrieval_output.results,
    )

    print(f"\n💬 Answer ({result.provider}/{result.model}):")
    print(f"{'─'*60}")
    print(result.answer)
    print(f"{'─'*60}")
    if result.latency_ms:
        print(f"⏱️  Latency: {result.latency_ms:.0f}ms")

    return retrieval_output, result


def run_comparison(queries: list[str] = None):
    """
    Compare retrieval strategies on sample queries.
    This produces the comparison data for your README tables.
    """
    if queries is None:
        queries = [
            "What is retrieval augmented generation and how does it work?",
            "How does LoRA fine-tuning reduce memory requirements?",
            "What are the differences between dense and sparse retrieval?",
            "Explain the attention mechanism in transformers",
            "What is chain-of-thought prompting?",
            "How does BM25 scoring work?",
            "What is RAGAS and how does it evaluate RAG systems?",
            "Compare different quantization techniques for LLMs",
        ]

    strategies = ["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]
    embedding_model = "all-MiniLM-L6-v2"
    chunk_strategy = "recursive"

    print("🔬 RETRIEVAL STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Embedding: {embedding_model}")
    print(f"Chunking:  {chunk_strategy}")
    print(f"Queries:   {len(queries)}")
    print(f"Strategies: {strategies}")
    print()

    retriever = Retriever(
        chroma_dir=CHROMA_DIR,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
        chunks_dir=CHUNKS_DIR,
    )

    results_log = []

    for query in queries:
        print(f"\n📝 {query[:60]}...")
        query_results = {}

        for strategy in strategies:
            output = retriever.retrieve(query=query, strategy=strategy)
            top_result = output.results[0] if output.results else None
            query_results[strategy] = {
                "num_results": output.num_results,
                "top_score": top_result.score if top_result else 0,
                "top_doc": top_result.doc_title[:40] if top_result else "N/A",
            }
            print(f"  {strategy:18s}: {output.num_results} results, "
                  f"top score={top_result.score:.4f}" if top_result else "")

        results_log.append({"query": query, "results": query_results})

    # Save comparison results
    output_file = DATA_DIR / "results" / "strategy_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")
    return results_log


def interactive_mode():
    """Interactive query loop."""
    print("🚀 RAG System — Interactive Mode")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Type 'quit' to exit, 'config' to change settings.\n")

    embedding_model = "all-MiniLM-L6-v2"
    chunk_strategy = "recursive"
    retrieval_strategy = "hybrid_rerank"

    while True:
        try:
            query = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "config":
            print(f"  Embedding:  {embedding_model}")
            print(f"  Chunking:   {chunk_strategy}")
            print(f"  Retrieval:  {retrieval_strategy}")
            continue

        run_single_query(
            query=query,
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            retrieval_strategy=retrieval_strategy,
        )


def main():
    if "--query" in sys.argv:
        idx = sys.argv.index("--query")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "What is RAG?"
        run_single_query(query)
    elif "--compare" in sys.argv:
        run_comparison()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
