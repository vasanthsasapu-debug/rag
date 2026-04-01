"""
arXiv Paper Downloader for RAG System
Downloads 50-100 ML/AI papers across key topics with metadata.

Supports two modes:
  1. API download (default) — fetches from arXiv API
  2. Manual mode — generates download links you can open in your browser
"""

import arxiv
import json
import os
import ssl
import time
from pathlib import Path
from datetime import datetime


# --- Configuration ---
SEARCH_QUERIES = [
    ("retrieval augmented generation", 15),
    ("large language model fine-tuning", 12),
    ("transformer architecture", 10),
    ("prompt engineering techniques", 8),
    ("vector database embeddings", 8),
    ("LLM evaluation benchmarks", 8),
    ("attention mechanism efficient", 7),
    ("instruction tuning RLHF", 7),
    ("mixture of experts LLM", 5),
    ("chain of thought reasoning", 5),
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
METADATA_FILE = DATA_DIR / "paper_metadata.json"

# --- Curated paper list (for manual download / SSL bypass) ---
# These are real, high-quality papers across our target topics
CURATED_PAPERS = [
    # RAG & Retrieval
    {"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "topic": "RAG"},
    {"arxiv_id": "2312.10997", "title": "Retrieval-Augmented Generation for Large Language Models: A Survey", "topic": "RAG"},
    {"arxiv_id": "2401.15884", "title": "A Survey on RAG Meeting LLMs", "topic": "RAG"},
    {"arxiv_id": "2402.19473", "title": "Corrective Retrieval Augmented Generation", "topic": "RAG"},
    {"arxiv_id": "2310.11511", "title": "Self-RAG: Learning to Retrieve, Generate, and Critique", "topic": "RAG"},
    {"arxiv_id": "2404.10981", "title": "From Local to Global: A Graph RAG Approach", "topic": "RAG"},
    {"arxiv_id": "2305.14283", "title": "Active Retrieval Augmented Generation", "topic": "RAG"},
    {"arxiv_id": "2407.21059", "title": "Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks", "topic": "RAG"},
    {"arxiv_id": "2307.03172", "title": "Lost in the Middle: How Language Models Use Long Contexts", "topic": "RAG"},
    {"arxiv_id": "2212.10496", "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)", "topic": "RAG"},
    {"arxiv_id": "2309.01431", "title": "Benchmarking Large Language Models in Retrieval-Augmented Generation", "topic": "RAG"},
    {"arxiv_id": "2403.14403", "title": "RAFT: Adapting Language Model to Domain Specific RAG", "topic": "RAG"},
    {"arxiv_id": "2310.06839", "title": "Seven Failure Points When Engineering a RAG System", "topic": "RAG"},

    # Transformers & Architecture
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "topic": "transformer"},
    {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners (GPT-3)", "topic": "transformer"},
    {"arxiv_id": "2302.13971", "title": "LLaMA: Open and Efficient Foundation Language Models", "topic": "transformer"},
    {"arxiv_id": "2307.09288", "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "topic": "transformer"},
    {"arxiv_id": "2310.06825", "title": "Mistral 7B", "topic": "transformer"},
    {"arxiv_id": "2401.04088", "title": "Mixtral of Experts", "topic": "transformer"},
    {"arxiv_id": "2205.01068", "title": "OPT: Open Pre-trained Transformer Language Models", "topic": "transformer"},
    {"arxiv_id": "2108.12409", "title": "Efficiently Modeling Long Sequences with Structured State Spaces", "topic": "transformer"},

    # Fine-tuning & PEFT
    {"arxiv_id": "2106.09685", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "topic": "fine-tuning"},
    {"arxiv_id": "2305.14314", "title": "QLoRA: Efficient Finetuning of Quantized Language Models", "topic": "fine-tuning"},
    {"arxiv_id": "2203.02155", "title": "Training language models to follow instructions (InstructGPT)", "topic": "fine-tuning"},
    {"arxiv_id": "2304.01933", "title": "LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning", "topic": "fine-tuning"},
    {"arxiv_id": "2110.07602", "title": "Multitask Prompted Training Enables Zero-Shot Task Generalization", "topic": "fine-tuning"},
    {"arxiv_id": "2402.12354", "title": "DoRA: Weight-Decomposed Low-Rank Adaptation", "topic": "fine-tuning"},
    {"arxiv_id": "2305.18290", "title": "Direct Preference Optimization (DPO)", "topic": "fine-tuning"},

    # Prompt Engineering & Reasoning
    {"arxiv_id": "2201.11903", "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs", "topic": "prompting"},
    {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models", "topic": "prompting"},
    {"arxiv_id": "2305.10601", "title": "Tree of Thoughts: Deliberate Problem Solving with LLMs", "topic": "prompting"},
    {"arxiv_id": "2309.03409", "title": "Textbooks Are All You Need II: phi-1.5 technical report", "topic": "prompting"},
    {"arxiv_id": "2205.11916", "title": "Large Language Models are Zero-Shot Reasoners", "topic": "prompting"},

    # Embeddings & Vector Search
    {"arxiv_id": "1908.10084", "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "topic": "embeddings"},
    {"arxiv_id": "2212.03533", "title": "E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training", "topic": "embeddings"},
    {"arxiv_id": "2401.00368", "title": "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity", "topic": "embeddings"},
    {"arxiv_id": "2104.08663", "title": "The Faiss Library", "topic": "embeddings"},
    {"arxiv_id": "2310.07554", "title": "Retrieve Anything To Augment LLMs", "topic": "embeddings"},

    # Evaluation & Benchmarks
    {"arxiv_id": "2309.15217", "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation", "topic": "evaluation"},
    {"arxiv_id": "2306.05685", "title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "topic": "evaluation"},
    {"arxiv_id": "2311.12022", "title": "ARES: An Automated Evaluation Framework for RAG Systems", "topic": "evaluation"},
    {"arxiv_id": "2305.14625", "title": "HELM: Holistic Evaluation of Language Models", "topic": "evaluation"},

    # Agents & Tools
    {"arxiv_id": "2308.08155", "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", "topic": "agents"},
    {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models", "topic": "agents"},
    {"arxiv_id": "2303.17580", "title": "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace", "topic": "agents"},
    {"arxiv_id": "2305.17126", "title": "Voyager: An Open-Ended Embodied Agent with Large Language Models", "topic": "agents"},
    {"arxiv_id": "2402.18679", "title": "An Introduction to LLM Agents", "topic": "agents"},

    # Quantization & Efficiency
    {"arxiv_id": "2208.07339", "title": "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", "topic": "efficiency"},
    {"arxiv_id": "2306.00978", "title": "AWQ: Activation-aware Weight Quantization for LLM Compression", "topic": "efficiency"},
    {"arxiv_id": "2210.17323", "title": "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", "topic": "efficiency"},
    {"arxiv_id": "2310.16944", "title": "Efficient Streaming Language Models with Attention Sinks", "topic": "efficiency"},
]


def download_papers(
    queries: list[tuple[str, int]] = SEARCH_QUERIES,
    output_dir: Path = RAW_PDF_DIR,
    metadata_file: Path = METADATA_FILE,
    max_total: int = 100,
):
    """Download arXiv papers via API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    metadata = []
    total_downloaded = 0

    client = arxiv.Client(
        page_size=20,
        delay_seconds=3.0,
        num_retries=3,
    )

    for query, count in queries:
        if total_downloaded >= max_total:
            break

        print(f"\n🔍 Searching: '{query}' (target: {count} papers)")

        search = arxiv.Search(
            query=query,
            max_results=count + 5,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        query_downloaded = 0
        for result in client.results(search):
            if total_downloaded >= max_total or query_downloaded >= count:
                break

            paper_id = result.entry_id.split("/")[-1]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            safe_name = f"{paper_id.replace('.', '_')}.pdf"
            pdf_path = output_dir / safe_name

            if pdf_path.exists():
                print(f"  ⏭️  Already exists: {safe_name}")
            else:
                try:
                    result.download_pdf(dirpath=str(output_dir), filename=safe_name)
                    print(f"  ✅ Downloaded: {result.title[:70]}...")
                except Exception as e:
                    print(f"  ❌ Failed: {result.title[:50]}... — {e}")
                    continue

            metadata.append({
                "arxiv_id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors[:5]],
                "abstract": result.summary,
                "categories": result.categories,
                "published": result.published.isoformat(),
                "pdf_filename": safe_name,
                "search_query": query,
            })

            total_downloaded += 1
            query_downloaded += 1
            time.sleep(0.5)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Done! Downloaded {total_downloaded} papers to {output_dir}")
    print(f"📋 Metadata saved to {metadata_file}")
    print(f"{'='*60}")
    return metadata


def generate_download_links(output_file: Path = DATA_DIR / "download_links.txt"):
    """
    Generate browser-friendly download links for all curated papers.
    Use this if the API is blocked by corporate firewall/SSL.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("arXiv Paper Download Links — RAG System Dataset")
    lines.append("=" * 70)
    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("1. Open each link in your browser (Ctrl+Click to open in new tab)")
    lines.append("2. Save the PDF to: data/raw_pdfs/")
    lines.append(f"3. Total papers: {len(CURATED_PAPERS)}")
    lines.append("")
    lines.append("TIP: You can also paste all URLs into a download manager")
    lines.append("     (like Free Download Manager) to batch-download them.")
    lines.append("")

    current_topic = None
    for paper in CURATED_PAPERS:
        if paper["topic"] != current_topic:
            current_topic = paper["topic"]
            lines.append("")
            lines.append(f"--- {current_topic.upper()} ---")

        arxiv_id = paper["arxiv_id"]
        lines.append(f"  {paper['title']}")
        lines.append(f"  https://arxiv.org/pdf/{arxiv_id}.pdf")
        lines.append("")

    content = "\n".join(lines)

    with open(output_file, "w") as f:
        f.write(content)

    print(f"✅ Generated {len(CURATED_PAPERS)} download links")
    print(f"📄 Saved to: {output_file}")
    print(f"\nOpen the file and click/copy the links to download manually.")
    return output_file


def build_metadata_from_folder(
    pdf_dir: Path = RAW_PDF_DIR,
    metadata_file: Path = METADATA_FILE,
):
    """
    Build metadata.json from whatever PDFs are already in the folder.
    Maps filenames back to curated paper info where possible.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup from curated list
    curated_lookup = {}
    for paper in CURATED_PAPERS:
        aid = paper["arxiv_id"]
        safe = f"{aid.replace('.', '_')}.pdf"
        curated_lookup[safe] = paper

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in {pdf_dir}")
        print("   Download papers first (API or manual links).")
        return []

    metadata = []
    for pdf_path in pdfs:
        fname = pdf_path.name
        if fname in curated_lookup:
            paper = curated_lookup[fname]
            metadata.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": [],
                "abstract": "",
                "categories": [],
                "published": "",
                "pdf_filename": fname,
                "search_query": paper["topic"],
            })
        else:
            # Unknown PDF — still include it
            metadata.append({
                "arxiv_id": fname.replace("_", ".").replace(".pdf", ""),
                "title": fname.replace("_", " ").replace(".pdf", ""),
                "authors": [],
                "abstract": "",
                "categories": [],
                "published": "",
                "pdf_filename": fname,
                "search_query": "unknown",
            })

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Built metadata for {len(metadata)} papers")
    print(f"📋 Saved to: {metadata_file}")
    return metadata


def show_stats(metadata_file: Path = METADATA_FILE):
    """Print summary statistics of downloaded papers."""
    if not metadata_file.exists():
        print("No metadata file found. Run download first.")
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    print(f"\n📊 Dataset Summary")
    print(f"{'='*40}")
    print(f"Total papers: {len(metadata)}")

    from collections import Counter
    query_counts = Counter(p["search_query"] for p in metadata)
    print(f"\nPapers by topic:")
    for query, count in query_counts.most_common():
        print(f"  • {query}: {count}")


if __name__ == "__main__":
    import sys

    print("🚀 arXiv Paper Downloader for RAG System\n")

    if "--links" in sys.argv:
        # Corporate firewall? Generate browser links instead
        generate_download_links()
    elif "--metadata" in sys.argv:
        # Already have PDFs? Just build metadata
        build_metadata_from_folder()
    else:
        # Default: try API download
        try:
            metadata = download_papers()
            show_stats()
        except Exception as e:
            if "SSL" in str(e) or "Certificate" in str(e):
                print(f"\n❌ SSL Error — likely a corporate firewall blocking the connection.")
                print(f"   Error: {str(e)[:100]}...\n")
                print("🔄 Generating manual download links instead...\n")
                generate_download_links()
                print("\n💡 After downloading the PDFs manually, run:")
                print("   python src/ingestion/download_arxiv.py --metadata")
            else:
                raise