"""
Retrieval Strategies Module
Implements 4 retrieval approaches for systematic comparison:
  1. Dense only — pure vector similarity
  2. BM25 only — pure sparse keyword retrieval
  3. Hybrid — dense + BM25 with Reciprocal Rank Fusion (RRF)
  4. Hybrid + Re-ranking — hybrid + cross-encoder re-ranking

All strategies return results in a common format for fair evaluation.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    """A single retrieved chunk with scores."""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    section_heading: str
    score: float
    rank: int
    retrieval_strategy: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalOutput:
    """Complete retrieval output for a single query."""
    query: str
    strategy: str
    embedding_model: str
    chunk_strategy: str
    results: list[RetrievalResult]
    num_results: int


# ============================================================
# BM25 Index (Sparse Retrieval)
# ============================================================

class BM25Index:
    """BM25 sparse retrieval index built from chunk documents."""

    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts with 'text' and 'chunk_id' fields
        """
        self.chunks = chunks
        self.corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """
        Search the BM25 index.

        Returns:
            List of (chunk_dict, bm25_score) tuples, sorted by score desc
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.chunks[idx], float(scores[idx])))

        return results


# ============================================================
# Cross-Encoder Re-ranker
# ============================================================

class CrossEncoderReranker:
    """Re-ranks retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        print(f"  📦 Loading re-ranker: {model_name}")
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Re-rank results using cross-encoder scores.

        The cross-encoder sees (query, document) pairs together,
        which gives much better relevance scoring than bi-encoder
        similarity alone — but it's slower (can't pre-compute).
        """
        if not results:
            return results

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, r.text) for r in results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        # Update ranks and return top_k
        for i, result in enumerate(reranked[:top_k]):
            result.rank = i + 1
            result.retrieval_strategy += "+rerank"

        return reranked[:top_k]


# ============================================================
# Retriever: Unified Interface
# ============================================================

class Retriever:
    """
    Unified retrieval interface supporting all 4 strategies.
    Works with any (embedding_model, chunk_strategy) combination.
    """

    def __init__(
        self,
        chroma_dir: str,
        embedding_model: str,
        chunk_strategy: str,
        chunks_dir: Optional[Path] = None,
    ):
        """
        Initialize retriever for a specific experiment configuration.

        Args:
            chroma_dir: Path to ChromaDB persistence directory
            embedding_model: Which embedding model's collection to use
            chunk_strategy: Which chunking strategy's collection to use
            chunks_dir: Path to chunks directory (needed for BM25)
        """
        import chromadb
        from chromadb.config import Settings

        self.embedding_model = embedding_model
        self.chunk_strategy = chunk_strategy

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get the right collection
        col_name = f"{embedding_model}__{chunk_strategy}"
        col_name = col_name.replace("/", "_").replace("-", "_").replace(".", "_")[:63]

        try:
            self.collection = self.chroma_client.get_collection(col_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{col_name}' not found. "
                f"Run the embedding pipeline first.\n  Error: {e}"
            )

        # Load embedder for query embedding
        from src.retrieval.embeddings import get_embedder
        embedding_type = self._infer_embedding_type(embedding_model)
        self.embedder = get_embedder(embedding_model, embedding_type)

        # Build BM25 index (needed for BM25 and hybrid strategies)
        self.bm25_index = None
        if chunks_dir:
            chunks_file = chunks_dir / f"chunks_{chunk_strategy}.json"
            if chunks_file.exists():
                with open(chunks_file) as f:
                    chunks = json.load(f)
                self.bm25_index = BM25Index(chunks)

        # Lazy-load re-ranker (only when needed)
        self._reranker = None

    @staticmethod
    def _infer_embedding_type(model_name: str) -> str:
        """Infer embedding provider type from model name."""
        vertex_ai_models = {"text-embedding-004", "text-embedding-005", "gemini-embedding-001"}
        if model_name in vertex_ai_models or model_name.startswith("text-embedding-"):
            return "vertex_ai"
        google_models = {"models/text-embedding-004", "models/embedding-001"}
        if model_name in google_models:
            return "google"
        return "sentence-transformers"

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid_rerank",
        top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> RetrievalOutput:
        """
        Retrieve relevant chunks using the specified strategy.

        Args:
            query: User query
            strategy: One of "dense_only", "bm25_only", "hybrid", "hybrid_rerank"
            top_k: Number of results to retrieve
            rerank_top_k: Number of results after re-ranking (for hybrid_rerank)

        Returns:
            RetrievalOutput with ranked results
        """
        if strategy == "dense_only":
            results = self._dense_search(query, top_k)
        elif strategy == "bm25_only":
            results = self._bm25_search(query, top_k)
        elif strategy == "hybrid":
            results = self._hybrid_search(query, top_k)
        elif strategy == "hybrid_rerank":
            results = self._hybrid_search(query, top_k)
            results = self.reranker.rerank(query, results, rerank_top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return RetrievalOutput(
            query=query,
            strategy=strategy,
            embedding_model=self.embedding_model,
            chunk_strategy=self.chunk_strategy,
            results=results,
            num_results=len(results),
        )

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Pure vector similarity search."""
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances (lower = closer); convert to similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance  # For cosine distance

            retrieval_results.append(RetrievalResult(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                doc_id=results["metadatas"][0][i].get("doc_id", ""),
                doc_title=results["metadatas"][0][i].get("doc_title", ""),
                section_heading=results["metadatas"][0][i].get("section_heading", ""),
                score=similarity,
                rank=i + 1,
                retrieval_strategy="dense_only",
                metadata=results["metadatas"][0][i],
            ))

        return retrieval_results

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Pure BM25 sparse retrieval."""
        if self.bm25_index is None:
            raise ValueError(
                "BM25 index not available. Provide chunks_dir when initializing."
            )

        bm25_results = self.bm25_index.search(query, top_k)

        retrieval_results = []
        for rank, (chunk, score) in enumerate(bm25_results):
            retrieval_results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                doc_id=chunk["doc_id"],
                doc_title=chunk["doc_title"],
                section_heading=chunk["section_heading"],
                score=score,
                rank=rank + 1,
                retrieval_strategy="bm25_only",
                metadata={},
            ))

        return retrieval_results

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[RetrievalResult]:
        """
        Hybrid search: Dense + BM25 combined with Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum( 1 / (k + rank) ) for each retriever
        where k=60 is a constant that prevents high-ranked items from
        dominating too much. This is the same fusion method used by
        Elasticsearch and major search engines.
        """
        # Get results from both retrievers
        dense_results = self._dense_search(query, top_k * 2)  # Fetch more for fusion
        bm25_results = self._bm25_search(query, top_k * 2)

        # Reciprocal Rank Fusion
        k = 60  # Standard RRF constant
        rrf_scores = {}
        chunk_lookup = {}

        # Score from dense retrieval
        for result in dense_results:
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0)
            rrf_scores[result.chunk_id] += dense_weight * (1.0 / (k + result.rank))
            chunk_lookup[result.chunk_id] = result

        # Score from BM25 retrieval
        for result in bm25_results:
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0)
            rrf_scores[result.chunk_id] += sparse_weight * (1.0 / (k + result.rank))
            if result.chunk_id not in chunk_lookup:
                chunk_lookup[result.chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
        retrieval_results = []
        for rank, chunk_id in enumerate(sorted_ids[:top_k]):
            base_result = chunk_lookup[chunk_id]
            retrieval_results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=base_result.text,
                doc_id=base_result.doc_id,
                doc_title=base_result.doc_title,
                section_heading=base_result.section_heading,
                score=rrf_scores[chunk_id],
                rank=rank + 1,
                retrieval_strategy="hybrid",
                metadata=base_result.metadata,
            ))

        return retrieval_results 