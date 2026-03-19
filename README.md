# RAG — Retrieval-Augmented Generation System

A production-ready **Retrieval-Augmented Generation (RAG)** system built with Python, FastAPI, LangChain, ChromaDB, and OpenAI.

RAG combines semantic search over your own documents with the generative power of large language models. Instead of relying solely on the LLM's training data, the system first retrieves the most relevant passages from your document store and feeds them as context to the model — giving you accurate, grounded answers with traceable sources.

---

## Architecture

```
User Query
    │
    ▼
DocumentIngester ──► VectorStore (ChromaDB)
                          │
                     Retriever (similarity search)
                          │
                     Generator (OpenAI GPT)
                          │
                     Answer + Sources
```

| Component | Technology |
|-----------|-----------|
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | ChromaDB (persistent) |
| LLM | OpenAI GPT (configurable, default `gpt-4o-mini`) |
| API | FastAPI |
| Document loading | LangChain loaders (TXT, Markdown, PDF) |

---

## Prerequisites

- Python 3.11+
- An OpenAI API key

---

## Installation

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file and fill in your OpenAI key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
TEMPERATURE=0.0
```

---

## Running the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Usage

### Health check

```bash
curl http://localhost:8000/
```

### Ingest raw text

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "LangChain is a framework for building LLM applications.", "metadata": {"source": "manual"}}'
```

### Ingest a file

```bash
curl -X POST http://localhost:8000/ingest/file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

Supported formats: `.txt`, `.md`, `.pdf`

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain?"}'
```

Response:

```json
{
  "answer": "LangChain is a framework for building LLM applications.",
  "sources": [
    {"content": "LangChain is a framework...", "metadata": {"source": "manual"}}
  ],
  "num_sources": 1
}
```

### Clear the collection

```bash
curl -X DELETE http://localhost:8000/collection
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
.
├── app.py                  # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and tool config
├── .env.example            # Environment variable template
└── src/
    └── rag/
        ├── __init__.py     # Package exports
        ├── config.py       # Pydantic settings
        ├── ingestion.py    # Document loading and chunking
        ├── embeddings.py   # OpenAI embedding wrapper
        ├── vectorstore.py  # ChromaDB wrapper
        ├── retriever.py    # Similarity search retrieval
        ├── generator.py    # LLM answer generation
        └── pipeline.py     # End-to-end RAG pipeline
```
