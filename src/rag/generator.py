from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag.config import Settings, get_settings

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based strictly on the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say so clearly. Do not make up information."
)

_NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the provided documents to answer that question."
)


class Generator:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=self._settings.temperature,
            openai_api_key=self._settings.openai_api_key,
        )

    def generate(self, query: str, context_documents: list[Document]) -> str:
        if not context_documents:
            return _NO_CONTEXT_RESPONSE

        context = "\n\n---\n\n".join(
            doc.page_content for doc in context_documents
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = self._llm.invoke(messages)
        return str(response.content)
