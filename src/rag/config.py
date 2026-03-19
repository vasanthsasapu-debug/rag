from __future__ import annotations

import functools

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-ada-002"
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    temperature: float = 0.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
