"""
LLM Generation Module
Provider-agnostic LLM interface with automatic fallback.
Primary: Groq (Llama-3.1 70B, free)
Fallback: Google Gemini via Vertex AI ($300 trial credit) or free API

Keeps the generation layer constant while we vary everything else.
"""

import os
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """LLM generation output with metadata."""
    answer: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


# ============================================================
# LLM Provider Wrappers
# ============================================================

class GroqLLM:
    """Groq API — Llama-3.1 70B (free tier: 30 RPM, 14,400 RPD)."""

    def __init__(self, model: str = "llama-3.1-70b-versatile"):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
                "Get a free key at: https://console.groq.com/keys\n"
                "Then add to .env: GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.provider = "groq"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000

        return GenerationResult(
            answer=response.choices[0].message.content,
            provider=self.provider,
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency,
        )


class GeminiLLM:
    """Google Gemini Flash (free API tier: 15 RPM, 500 RPD)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found.\n"
                "Get a free key at: https://aistudio.google.com/apikey\n"
                "Then add to .env: GOOGLE_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.provider = "google"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        start = time.time()
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        latency = (time.time() - start) * 1000

        return GenerationResult(
            answer=response.text,
            provider=self.provider,
            model=self.model_name,
            latency_ms=latency,
        )


class VertexAILLM:
    """
    Google Gemini via Vertex AI (GCP trial: $300 credit).

    Uses the new google-genai SDK (recommended over the deprecated vertexai SDK).
    Higher rate limits than the free Gemini API, same models.

    Setup:
        1. pip install google-genai google-auth
        2. Set env vars in .env:
           GOOGLE_CLOUD_PROJECT=your-project-id
           GOOGLE_CLOUD_LOCATION=us-central1
           GOOGLE_APPLICATION_CREDENTIALS=path/to/api_key.json
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
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
        self.model_name = model
        self.provider = "vertex_ai"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        from google.genai import types

        start = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency = (time.time() - start) * 1000

        # Extract token counts if available
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        return GenerationResult(
            answer=response.text,
            provider=self.provider,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
        )


# ============================================================
# RAG Generation Chain
# ============================================================

# System prompt for RAG generation
RAG_SYSTEM_PROMPT = """You are a helpful research assistant that answers questions about machine learning and AI papers.

INSTRUCTIONS:
- Answer the question using ONLY the provided context from retrieved papers.
- If the context doesn't contain enough information, say so clearly.
- Cite the source paper titles when making claims.
- Be concise but thorough.
- If multiple papers discuss the topic, synthesize their findings.

CONTEXT FROM RETRIEVED PAPERS:
{context}
"""


class RAGGenerator:
    """
    RAG generation chain: takes retrieved chunks + query → generates answer.
    Uses Groq as primary, Vertex AI / Gemini as fallback.

    Provider priority (configurable):
        "groq"      → Free Llama-3.1 70B via Groq
        "vertex_ai" → Gemini via Vertex AI (GCP $300 trial credit)
        "google"    → Gemini via free API (low rate limits)
    """

    def __init__(
        self,
        primary: str = "groq",
        fallback: str = "vertex_ai",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize providers
        self.providers = {}
        self.provider_order = []

        for provider_name in [primary, fallback]:
            if provider_name in self.providers:
                continue  # Skip duplicates
            try:
                if provider_name == "groq":
                    self.providers["groq"] = GroqLLM()
                    self.provider_order.append("groq")
                    print(f"  ✅ Groq (Llama-3.1 70B) — connected")
                elif provider_name == "vertex_ai":
                    self.providers["vertex_ai"] = VertexAILLM()
                    self.provider_order.append("vertex_ai")
                    print(f"  ✅ Vertex AI (Gemini Flash) — connected")
                elif provider_name == "google":
                    self.providers["google"] = GeminiLLM()
                    self.provider_order.append("google")
                    print(f"  ✅ Gemini Flash (free API) — connected")
            except ValueError as e:
                print(f"  ⚠️  {provider_name}: {e}")

        if not self.providers:
            raise ValueError(
                "No LLM providers available. Set at least one of:\n"
                "  GROQ_API_KEY           — https://console.groq.com/keys\n"
                "  GOOGLE_CLOUD_PROJECT   — GCP console (Vertex AI)\n"
                "  GOOGLE_API_KEY         — https://aistudio.google.com/apikey"
            )

    def generate(
        self,
        query: str,
        retrieved_chunks: list,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate an answer using retrieved context.

        Args:
            query: User's question
            retrieved_chunks: List of RetrievalResult objects
            system_prompt: Custom system prompt (uses default RAG prompt if None)

        Returns:
            GenerationResult with the answer and metadata
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.doc_title if hasattr(chunk, "doc_title") else "Unknown"
            section = chunk.section_heading if hasattr(chunk, "section_heading") else ""
            context_parts.append(
                f"[Source {i+1}: {source} — {section}]\n{chunk.text}\n"
            )

        context = "\n".join(context_parts)

        # Build prompt
        if system_prompt is None:
            system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
            user_prompt = f"Question: {query}"
        else:
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Try providers in order (primary → fallback)
        last_error = None
        for provider_name in self.provider_order:
            try:
                result = self.providers[provider_name].generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return result

            except Exception as e:
                last_error = e
                print(f"  ⚠️  {provider_name} failed: {e}")
                print(f"  🔄 Trying fallback...")
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )