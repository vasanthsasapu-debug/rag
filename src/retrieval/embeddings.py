"""
Embedding Pipeline
Creates vector stores using multiple embedding models for comparison.
Supports:
  - sentence-transformers (local/free)
  - Google Gemini embedding (free API)
  - Google Vertex AI embedding ($300 GCP trial credit)

Each embedding model gets its own ChromaDB collection, enabling
side-by-side retrieval quality comparison.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os

import chromadb
from chromadb.config import Settings


# ============================================================
# Embedding Model Wrappers
# ============================================================

class SentenceTransformerEmbedder:
    """Local embedding using sentence-transformers (free, no API)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        print(f"  📦 Loading {model_name}...")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


class GoogleEmbedder:
    """Google Gemini embedding via free API (text-embedding-004)."""

    def __init__(self, model_name: str = "models/text-embedding-004"):
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment.\n"
                "Get a free key at: https://aistudio.google.com/apikey\n"
                "Then add to .env: GOOGLE_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def embed(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Embed texts in batches to respect rate limits."""
        import google.generativeai as genai

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            all_embeddings.extend(result["embedding"])

            # Rate limit: 15 RPM on free tier, so be gentle
            if i + batch_size < len(texts):
                time.sleep(2)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        import google.generativeai as genai

        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]


class VertexAIEmbedder:
    """
    Google embedding via Vertex AI (GCP trial: $300 credit).

    Uses the new google-genai SDK with vertexai=True.
    Much higher rate limits than the free API.

    Supports:
      - "text-embedding-004"   (768 dims, legacy but solid)
      - "gemini-embedding-001" (3072 dims default, supports MRL down to 256)

    Setup:
        1. pip install google-genai google-auth
        2. Set env vars in .env:
           GOOGLE_CLOUD_PROJECT=your-project-id
           GOOGLE_CLOUD_LOCATION=us-central1
           GOOGLE_APPLICATION_CREDENTIALS=path/to/api_key.json
    """

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        output_dimensionality: Optional[int] = 768,
    ):
        from pathlib import Path
        from google import genai

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT not found.\n"
                "Set in .env:\n"
                "  GOOGLE_CLOUD_PROJECT=your-project-id\n"
                "  GOOGLE_CLOUD_LOCATION=us-central1\n"
                "  GOOGLE_APPLICATION_CREDENTIALS=path/to/api_key.json"
            )

        # Load service account credentials explicitly (Windows-safe)
        creds = None
        if creds_path and Path(creds_path).exists():
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
        )
        self.model_name = model_name
        self.output_dimensionality = output_dimensionality
        # text-embedding-004/005 support up to 20,000 tokens (~60,000 chars)
        # Truncate long texts to stay safely under the limit
        self.max_chars = 50000

    def _truncate(self, text: str) -> str:
        """Truncate text to stay within the model's token limit."""
        if len(text) > self.max_chars:
            return text[:self.max_chars]
        return text

    def embed(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        Embed texts in batches via Vertex AI.
        Batch size is kept small (20) because the API's token limit applies
        to the TOTAL tokens across all texts in a single request, not per text.
        Includes retry with exponential backoff for rate limit (429) errors.
        """
        from google.genai.types import EmbedContentConfig

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [self._truncate(t) for t in texts[i : i + batch_size]]

            config = EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            )
            if self.output_dimensionality:
                config = EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.output_dimensionality,
                )

            # Retry with exponential backoff for rate limits
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    result = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=config,
                    )
                    break  # Success — exit retry loop
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                        print(f"    ⏳ Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait)
                        if attempt == max_retries - 1:
                            raise  # Give up after max retries
                    else:
                        raise  # Non-rate-limit error — fail immediately

            # Extract embedding vectors from response
            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)

            # Rate limiting: ~1.5s between batches to stay under quota
            if i + batch_size < len(texts):
                time.sleep(1.5)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with RETRIEVAL_QUERY task type."""
        from google.genai.types import EmbedContentConfig

        text = self._truncate(text)

        config = EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        )
        if self.output_dimensionality:
            config = EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality,
            )

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=config,
        )

        return result.embeddings[0].values


def get_embedder(model_name: str, model_type: str):
    """Factory function to create the right embedder."""
    if model_type == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    elif model_type == "google":
        return GoogleEmbedder(model_name)
    elif model_type == "vertex_ai":
        return VertexAIEmbedder(model_name)
    else:
        raise ValueError(f"Unknown embedding type: {model_type}")


# ============================================================
# ChromaDB Vector Store Management
# ============================================================

class VectorStoreManager:
    """
    Manages ChromaDB collections — one per (embedding_model, chunking_strategy) pair.
    This enables systematic comparison across the experiment matrix.
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection_name(self, embedding_model: str, chunk_strategy: str) -> str:
        """Generate a clean collection name from model + strategy."""
        # ChromaDB collection names: 3-63 chars, alphanumeric + underscores
        name = f"{embedding_model}__{chunk_strategy}"
        # Clean: replace slashes, hyphens, dots
        name = name.replace("/", "_").replace("-", "_").replace(".", "_")
        # Truncate if needed
        return name[:63]

    def get_or_create_collection(
        self, embedding_model: str, chunk_strategy: str
    ) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a specific experiment config."""
        name = self._collection_name(embedding_model, chunk_strategy)
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={
                "embedding_model": embedding_model,
                "chunk_strategy": chunk_strategy,
            },
        )
        return collection

    def collection_exists(
        self, embedding_model: str, chunk_strategy: str
    ) -> bool:
        """Check if a collection already has data."""
        name = self._collection_name(embedding_model, chunk_strategy)
        try:
            col = self.client.get_collection(name)
            return col.count() > 0
        except Exception:
            return False

    def list_collections(self) -> list[dict]:
        """List all collections with their stats."""
        collections = self.client.list_collections()
        stats = []
        for col in collections:
            collection = self.client.get_collection(col.name)
            stats.append({
                "name": col.name,
                "count": collection.count(),
                "metadata": col.metadata,
            })
        return stats

    def delete_collection(self, embedding_model: str, chunk_strategy: str):
        """Delete a specific collection."""
        name = self._collection_name(embedding_model, chunk_strategy)
        try:
            self.client.delete_collection(name)
            print(f"  🗑️  Deleted collection: {name}")
        except Exception:
            pass


# ============================================================
# Embedding Pipeline
# ============================================================

def build_vector_store(
    chunks_file: Path,
    embedding_model: str,
    embedding_type: str,
    chunk_strategy: str,
    store_manager: VectorStoreManager,
    force_rebuild: bool = False,
    batch_size: int = 100,
) -> int:
    """
    Build a ChromaDB collection for one (embedding_model, chunk_strategy) pair.

    Args:
        chunks_file: Path to chunks JSON file
        embedding_model: Model name (e.g. "all-MiniLM-L6-v2")
        embedding_type: Model type ("sentence-transformers", "google", or "vertex_ai")
        chunk_strategy: Strategy name ("fixed", "recursive", "semantic")
        store_manager: VectorStoreManager instance
        force_rebuild: If True, delete and rebuild existing collection
        batch_size: Number of chunks to embed at once

    Returns:
        Number of chunks indexed
    """
    # Vertex AI embedding models have a total token limit per request (~20K tokens).
    # With average chunks of ~1000 chars (~300 tokens), batch_size=20 keeps us safe.
    if embedding_type == "vertex_ai":
        batch_size = 20

    # Check if already built
    if not force_rebuild and store_manager.collection_exists(embedding_model, chunk_strategy):
        col = store_manager.get_or_create_collection(embedding_model, chunk_strategy)
        print(f"  ⏭️  Already exists: {embedding_model} × {chunk_strategy} "
              f"({col.count()} chunks)")
        return col.count()

    if force_rebuild:
        store_manager.delete_collection(embedding_model, chunk_strategy)

    # Load chunks
    with open(chunks_file) as f:
        chunks = json.load(f)

    if not chunks:
        print(f"  ❌ No chunks in {chunks_file}")
        return 0

    print(f"\n  🔢 Embedding {len(chunks)} chunks with {embedding_model} (batch_size={batch_size})...")

    # Initialize embedder
    embedder = get_embedder(embedding_model, embedding_type)

    # Get or create collection
    collection = store_manager.get_or_create_collection(embedding_model, chunk_strategy)

    # Process in batches
    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "doc_title": c["doc_title"],
                "section_heading": c["section_heading"],
                "chunk_index": c["chunk_index"],
                "char_count": c["char_count"],
                "strategy": c["strategy"],
            }
            for c in batch
        ]

        # Generate embeddings — pass all texts, let the embedder handle
        # its own internal batching (already set appropriately per provider)
        embeddings = embedder.embed(texts)

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_indexed += len(batch)
        print(f"    → Indexed {total_indexed}/{len(chunks)} chunks")

    print(f"  ✅ {embedding_model} × {chunk_strategy}: {total_indexed} chunks indexed")
    return total_indexed


def run_embedding_pipeline(
    chunks_dir: Path,
    persist_dir: str = "./data/chroma_db",
    models: Optional[list[dict]] = None,
    strategies: Optional[list[str]] = None,
    force_rebuild: bool = False,
) -> dict:
    """
    Build vector stores for all (embedding_model, chunk_strategy) combinations.

    Args:
        chunks_dir: Directory containing chunks_*.json files
        persist_dir: ChromaDB persistence directory
        models: List of embedding model configs. Default: all free models.
        strategies: Which chunk strategies to index. Default: all.
        force_rebuild: If True, rebuild all collections from scratch.

    Returns:
        Dict of results: {(model, strategy): num_chunks}
    """
    # Default: 3 free sentence-transformer models
    if models is None:
        models = [
            {"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"},
            {"name": "all-mpnet-base-v2", "type": "sentence-transformers"},
            {"name": "BAAI/bge-large-en-v1.5", "type": "sentence-transformers"},
        ]

    if strategies is None:
        strategies = ["fixed", "recursive", "semantic"]

    store_manager = VectorStoreManager(persist_dir)

    print("🔢 EMBEDDING PIPELINE")
    print("=" * 60)
    print(f"   Models:     {[m['name'] for m in models]}")
    print(f"   Strategies: {strategies}")
    total_combos = len(models) * len(strategies)
    print(f"   Total combinations: {total_combos}")
    print()

    results = {}

    for model_config in models:
        model_name = model_config["name"]
        model_type = model_config["type"]

        print(f"\n{'─'*60}")
        print(f"📦 Model: {model_name}")
        print(f"{'─'*60}")

        for strategy in strategies:
            chunks_file = chunks_dir / f"chunks_{strategy}.json"
            if not chunks_file.exists():
                print(f"  ⚠️  Chunks file not found: {chunks_file}")
                continue

            count = build_vector_store(
                chunks_file=chunks_file,
                embedding_model=model_name,
                embedding_type=model_type,
                chunk_strategy=strategy,
                store_manager=store_manager,
                force_rebuild=force_rebuild,
            )
            results[(model_name, strategy)] = count

    # Summary
    print(f"\n{'='*60}")
    print("📊 Embedding Pipeline Summary")
    print(f"{'='*60}")

    collections = store_manager.list_collections()
    for col in collections:
        print(f"  • {col['name']}: {col['count']} chunks")

    print(f"\nTotal collections: {len(collections)}")
    print(f"ChromaDB location: {persist_dir}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import sys

    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
    CHROMA_DIR = str(DATA_DIR / "chroma_db")

    force = "--force" in sys.argv
    quick = "--quick" in sys.argv
    include_vertex = "--vertex" in sys.argv

    if quick:
        # Quick mode: only MiniLM + recursive (for testing)
        models = [{"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"}]
        strategies = ["recursive"]
        print("⚡ Quick mode: MiniLM × recursive only\n")
    elif include_vertex:
        # Full mode including Vertex AI embedding
        models = [
            {"name": "all-MiniLM-L6-v2", "type": "sentence-transformers"},
            {"name": "all-mpnet-base-v2", "type": "sentence-transformers"},
            {"name": "BAAI/bge-large-en-v1.5", "type": "sentence-transformers"},
            {"name": "text-embedding-004", "type": "vertex_ai"},
        ]
        strategies = None   # All 3 strategies
        print("🚀 Full mode with Vertex AI embedding\n")
    else:
        models = None      # Use defaults (all 3 free models)
        strategies = None   # Use defaults (all 3 strategies)

    run_embedding_pipeline(
        chunks_dir=CHUNKS_DIR,
        persist_dir=CHROMA_DIR,
        models=models,
        strategies=strategies,
        force_rebuild=force,
    )