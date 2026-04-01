"""
Text Chunking Strategies Module
Implements 3 chunking approaches for systematic comparison:
  1. Fixed-size chunking (baseline)
  2. Recursive character text splitting (LangChain standard)
  3. Semantic chunking (embedding-based)

Each strategy produces chunks with metadata for traceability.
"""

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


@dataclass
class Chunk:
    """A single chunk of text with full traceability metadata."""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    section_heading: str
    chunk_index: int           # Position within the document
    total_chunks: int          # Total chunks from this document
    char_count: int
    strategy: str              # "fixed", "recursive", or "semantic"
    metadata: dict = field(default_factory=dict)


def _generate_chunk_id(doc_id: str, strategy: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{doc_id}_{strategy}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _assign_section(chunk_text: str, sections: list[dict]) -> str:
    """Find which section a chunk most likely belongs to."""
    if not sections:
        return "Unknown"

    # Simple heuristic: check which section's content overlaps most
    best_section = sections[0]["heading"]
    best_overlap = 0

    for section in sections:
        # Count how many words from the chunk appear in this section
        chunk_words = set(chunk_text.lower().split()[:50])  # First 50 words
        section_words = set(section["content"].lower().split()[:200])
        overlap = len(chunk_words & section_words)

        if overlap > best_overlap:
            best_overlap = overlap
            best_section = section["heading"]

    return best_section


# ============================================================
# Strategy 1: Fixed-Size Chunking (Baseline)
# ============================================================

def chunk_fixed_size(
    text: str,
    doc_id: str,
    doc_title: str,
    sections: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Simple fixed-size chunking by token count.
    This is the baseline — fast but ignores document structure.

    Args:
        chunk_size: Number of tokens per chunk
        chunk_overlap: Token overlap between consecutive chunks
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",  # GPT-4 / text-embedding-3 tokenizer
    )

    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=_generate_chunk_id(doc_id, "fixed", i),
            text=chunk_text.strip(),
            doc_id=doc_id,
            doc_title=doc_title,
            section_heading=_assign_section(chunk_text, sections),
            chunk_index=i,
            total_chunks=total,
            char_count=len(chunk_text),
            strategy="fixed",
            metadata=metadata or {},
        ))

    return chunks


# ============================================================
# Strategy 2: Recursive Character Text Splitting
# ============================================================

def chunk_recursive(
    text: str,
    doc_id: str,
    doc_title: str,
    sections: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Recursive splitting — tries to split on paragraphs first,
    then sentences, then words. Preserves natural boundaries.

    This is the most commonly used strategy in production RAG.

    Args:
        chunk_size: Max characters per chunk
        chunk_overlap: Character overlap between chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(Chunk(
            chunk_id=_generate_chunk_id(doc_id, "recursive", i),
            text=chunk_text.strip(),
            doc_id=doc_id,
            doc_title=doc_title,
            section_heading=_assign_section(chunk_text, sections),
            chunk_index=i,
            total_chunks=total,
            char_count=len(chunk_text),
            strategy="recursive",
            metadata=metadata or {},
        ))

    return chunks


# ============================================================
# Strategy 3: Semantic Chunking
# ============================================================

def chunk_semantic(
    text: str,
    doc_id: str,
    doc_title: str,
    sections: list[dict],
    max_chunk_size: int = 1500,
    similarity_threshold: float = 0.75,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Semantic chunking — groups consecutive sentences that are
    semantically similar using embedding cosine similarity.

    Sentences with similar meaning stay together. When similarity
    drops below the threshold, a new chunk starts.

    This captures topical boundaries that character-based
    methods miss completely.

    Args:
        max_chunk_size: Maximum characters per chunk (hard cap)
        similarity_threshold: Cosine similarity threshold for grouping
    """
    import numpy as np

    # Split into sentences first
    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(
            chunk_id=_generate_chunk_id(doc_id, "semantic", 0),
            text=text.strip(),
            doc_id=doc_id,
            doc_title=doc_title,
            section_heading=sections[0]["heading"] if sections else "Unknown",
            chunk_index=0,
            total_chunks=1,
            char_count=len(text),
            strategy="semantic",
            metadata=metadata or {},
        )]

    # Compute sentence embeddings using a lightweight model
    # We use a small model here just for chunking — retrieval uses the full models
    embeddings = _get_sentence_embeddings(sentences)

    # Group sentences by semantic similarity
    groups = []
    current_group = [0]  # Start with first sentence

    for i in range(1, len(sentences)):
        # Compare current sentence with the average of the current group
        group_embedding = np.mean(
            [embeddings[j] for j in current_group], axis=0
        )
        similarity = _cosine_similarity(embeddings[i], group_embedding)

        current_text = " ".join(sentences[j] for j in current_group)

        # Start new group if similarity drops or chunk gets too large
        if similarity < similarity_threshold or len(current_text) > max_chunk_size:
            groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    # Build chunks from sentence groups
    total = len(groups)
    chunks = []

    for i, group in enumerate(groups):
        chunk_text = " ".join(sentences[j] for j in group)
        chunks.append(Chunk(
            chunk_id=_generate_chunk_id(doc_id, "semantic", i),
            text=chunk_text.strip(),
            doc_id=doc_id,
            doc_title=doc_title,
            section_heading=_assign_section(chunk_text, sections),
            chunk_index=i,
            total_chunks=total,
            char_count=len(chunk_text),
            strategy="semantic",
            metadata=metadata or {},
        ))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (avoids heavy NLP dependencies)."""
    import re

    # Split on period, question mark, or exclamation followed by space + capital
    # Handles abbreviations like "et al.", "Fig.", "Eq." reasonably well
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])',
        text
    )

    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _get_sentence_embeddings(sentences: list[str]) -> list:
    """
    Get embeddings for sentences using a lightweight model.
    Uses all-MiniLM-L6-v2 for chunking (fast, small, good enough for grouping).
    """
    from sentence_transformers import SentenceTransformer

    # Cache the model in a module-level variable for reuse
    if not hasattr(_get_sentence_embeddings, "_model"):
        print("  📦 Loading sentence embedding model for semantic chunking...")
        _get_sentence_embeddings._model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",
        )

    model = _get_sentence_embeddings._model
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
    return embeddings


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


# ============================================================
# Main Pipeline: Process Documents with All 3 Strategies
# ============================================================

def chunk_documents(
    documents: list[dict],
    strategies: list[str] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, list[Chunk]]:
    """
    Apply chunking strategies to all documents.

    Args:
        documents: List of ExtractedDocument dicts (from pdf_extractor)
        strategies: Which strategies to run. Default: all three.
        output_dir: Optional directory to save chunked output

    Returns:
        Dict mapping strategy name to list of all chunks
    """
    if strategies is None:
        strategies = ["fixed", "recursive", "semantic"]

    strategy_map = {
        "fixed": chunk_fixed_size,
        "recursive": chunk_recursive,
        "semantic": chunk_semantic,
    }

    results = {s: [] for s in strategies}

    print(f"\n🔪 Chunking {len(documents)} documents with strategies: {strategies}\n")

    for doc in documents:
        doc_id = doc["doc_id"]
        doc_title = doc["title"]
        full_text = doc["full_text"]
        sections = doc.get("sections", [])
        meta = doc.get("metadata", {})

        title_display = doc_title[:50] + "..." if len(doc_title) > 50 else doc_title
        print(f"  📄 {title_display}")

        for strategy_name in strategies:
            chunk_fn = strategy_map[strategy_name]
            chunks = chunk_fn(
                text=full_text,
                doc_id=doc_id,
                doc_title=doc_title,
                sections=sections,
                metadata=meta,
            )
            results[strategy_name].extend(chunks)
            print(f"     → {strategy_name}: {len(chunks)} chunks")

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Chunking Summary:")
    for strategy_name, chunks in results.items():
        avg_len = sum(c.char_count for c in chunks) / max(len(chunks), 1)
        print(f"  • {strategy_name:12s}: {len(chunks):5d} chunks "
              f"(avg {avg_len:.0f} chars)")
    print(f"{'='*60}")

    # Save to disk
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for strategy_name, chunks in results.items():
            out_path = output_dir / f"chunks_{strategy_name}.json"
            with open(out_path, "w") as f:
                json.dump([asdict(c) for c in chunks], f, indent=2)
            print(f"💾 Saved {strategy_name} chunks to {out_path}")

    return results


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    EXTRACTED_FILE = DATA_DIR / "processed" / "extracted_documents.json"
    CHUNKS_DIR = DATA_DIR / "processed" / "chunks"

    # Load extracted documents
    if not EXTRACTED_FILE.exists():
        print("❌ No extracted documents found.")
        print("   Run pdf_extractor.py first:")
        print("   python src/ingestion/pdf_extractor.py")
        exit(1)

    with open(EXTRACTED_FILE) as f:
        documents = json.load(f)

    print(f"📄 Loaded {len(documents)} extracted documents\n")

    # Run all 3 chunking strategies
    results = chunk_documents(documents, output_dir=CHUNKS_DIR)
