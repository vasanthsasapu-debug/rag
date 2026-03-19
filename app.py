from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

app = FastAPI(title="RAG API", version="0.1.0")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class IngestTextRequest(BaseModel):
    text: str
    metadata: dict | None = None


class IngestFileRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "service": "RAG API"}


@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest) -> dict:
    try:
        chunk_count = get_pipeline().ingest_text(request.text, request.metadata)
        return {"message": f"Successfully ingested {chunk_count} chunk(s).", "chunks_ingested": chunk_count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest/file")
def ingest_file(request: IngestFileRequest) -> dict:
    try:
        chunk_count = get_pipeline().ingest_file(request.file_path)
        return {"message": f"Successfully ingested {chunk_count} chunk(s).", "chunks_ingested": chunk_count}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        return get_pipeline().query(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/collection")
def delete_collection() -> dict:
    try:
        get_pipeline().clear()
        return {"message": "Collection cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
