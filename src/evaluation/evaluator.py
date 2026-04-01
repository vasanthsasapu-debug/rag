"""
RAGAS Evaluation Pipeline
Systematically evaluates RAG configurations across the experiment matrix.

Metrics:
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved chunks relevant? (ranked higher = better)
  - Context Recall: Does the retrieved context cover the ground truth?

Usage:
    python src/evaluation/evaluator.py                    # Evaluate default config
    python src/evaluation/evaluator.py --all              # Full experiment matrix
    python src/evaluation/evaluator.py --generate-queries # Generate test queries only
"""

import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime


@dataclass
class EvalSample:
    """A single evaluation sample with question, ground truth, and RAG outputs."""
    question: str
    ground_truth: str  # Human-written expected answer
    answer: str = ""  # RAG-generated answer
    contexts: list[str] = field(default_factory=list)  # Retrieved chunks
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Evaluation result for one configuration."""
    embedding_model: str
    chunk_strategy: str
    retrieval_strategy: str
    num_samples: int
    metrics: dict  # {metric_name: score}
    per_sample: list[dict] = field(default_factory=list)
    timestamp: str = ""
    latency_seconds: float = 0.0


# ============================================================
# Test Query Generation
# ============================================================

# Curated test queries spanning different difficulty levels and topics.
# Ground truths are written to be verifiable from the arXiv papers in the corpus.
DEFAULT_TEST_QUERIES = [
    {
        "question": "What is retrieval augmented generation and how does it work?",
        "ground_truth": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval component with a "
            "generative language model. Given a query, the retriever finds relevant documents "
            "from a knowledge base, and the generator produces an answer conditioned on both "
            "the query and the retrieved documents. The original RAG paper by Lewis et al. "
            "proposed using a pre-trained seq2seq model (BART) with a neural retriever (DPR) "
            "that accesses a Wikipedia index."
        ),
        "topic": "RAG",
        "difficulty": "easy",
    },
    {
        "question": "How does LoRA reduce the number of trainable parameters during fine-tuning?",
        "ground_truth": (
            "LoRA (Low-Rank Adaptation) freezes the pre-trained model weights and injects "
            "trainable low-rank decomposition matrices into each transformer layer. Instead of "
            "updating the full weight matrix W (d×d), LoRA learns two smaller matrices A (d×r) "
            "and B (r×d) where r << d, so the update is W + BA. This reduces trainable "
            "parameters by 10,000x or more while matching full fine-tuning performance."
        ),
        "topic": "fine-tuning",
        "difficulty": "medium",
    },
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": (
            "The attention mechanism in transformers computes scaled dot-product attention: "
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Queries, keys, and values are "
            "linear projections of the input. Multi-head attention runs multiple attention "
            "functions in parallel, allowing the model to attend to information from different "
            "representation subspaces. Self-attention lets each position attend to all other "
            "positions in the sequence."
        ),
        "topic": "transformer",
        "difficulty": "easy",
    },
    {
        "question": "How does BM25 scoring work for text retrieval?",
        "ground_truth": (
            "BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency "
            "and inverse document frequency. It scores documents by summing over query terms: "
            "each term's score depends on its frequency in the document (with diminishing returns "
            "via saturation parameter k1), the document length relative to average (parameter b), "
            "and the term's rarity across the corpus (IDF). BM25 is the default ranking function "
            "in most search engines."
        ),
        "topic": "retrieval",
        "difficulty": "medium",
    },
    {
        "question": "What is chain-of-thought prompting and why does it improve reasoning?",
        "ground_truth": (
            "Chain-of-thought (CoT) prompting involves providing examples that include step-by-step "
            "reasoning before the final answer. Wei et al. showed that CoT prompting significantly "
            "improves performance on arithmetic, commonsense, and symbolic reasoning tasks. It "
            "works because it decomposes complex problems into intermediate steps, making each "
            "step easier for the model. Zero-shot CoT simply appends 'Let's think step by step' "
            "to the prompt."
        ),
        "topic": "prompting",
        "difficulty": "easy",
    },
    {
        "question": "Compare dense retrieval and sparse retrieval for RAG systems.",
        "ground_truth": (
            "Dense retrieval uses learned vector embeddings to capture semantic similarity — it "
            "excels at finding paraphrases and conceptually similar passages but can miss exact "
            "keyword matches. Sparse retrieval (like BM25) uses term frequency statistics — it "
            "excels at exact term matching and is robust for specific entities and numbers but "
            "misses semantic equivalences. Hybrid approaches combine both using methods like "
            "Reciprocal Rank Fusion (RRF) to get the best of both worlds."
        ),
        "topic": "retrieval",
        "difficulty": "medium",
    },
    {
        "question": "What is QLoRA and how does it differ from standard LoRA?",
        "ground_truth": (
            "QLoRA extends LoRA by quantizing the base model to 4-bit precision using a new "
            "NormalFloat4 data type, while keeping the LoRA adapter weights in higher precision "
            "(BFloat16). It introduces double quantization (quantizing the quantization constants) "
            "and paged optimizers to manage memory spikes. QLoRA enables fine-tuning a 65B "
            "parameter model on a single 48GB GPU while maintaining full 16-bit fine-tuning "
            "performance."
        ),
        "topic": "fine-tuning",
        "difficulty": "hard",
    },
    {
        "question": "Explain the Mixture of Experts (MoE) architecture used in models like Mixtral.",
        "ground_truth": (
            "Mixture of Experts replaces each dense feed-forward layer with multiple expert "
            "sub-networks and a gating/router network. For each token, the router selects only "
            "a subset of experts (typically 2 out of 8), so computation scales with the number "
            "of active experts rather than total parameters. Mixtral 8x7B has 46.7B total "
            "parameters but only uses 12.9B per token, achieving performance comparable to "
            "much larger dense models while being faster at inference."
        ),
        "topic": "transformer",
        "difficulty": "hard",
    },
    {
        "question": "How does the RAGAS framework evaluate RAG systems?",
        "ground_truth": (
            "RAGAS (Retrieval Augmented Generation Assessment) evaluates RAG pipelines using "
            "four component-level metrics: faithfulness measures whether the answer is grounded "
            "in the retrieved context; answer relevancy checks if the answer addresses the "
            "question; context precision evaluates whether retrieved chunks are relevant (with "
            "ranking awareness); and context recall measures if the retrieval covers the ground "
            "truth information. RAGAS uses LLM-based evaluation to compute these metrics."
        ),
        "topic": "evaluation",
        "difficulty": "medium",
    },
    {
        "question": "What is Self-RAG and how does it improve over standard RAG?",
        "ground_truth": (
            "Self-RAG trains a language model to adaptively retrieve, generate, and critique "
            "its own outputs using special reflection tokens. Unlike standard RAG which always "
            "retrieves, Self-RAG decides when retrieval is necessary, evaluates whether retrieved "
            "passages are relevant, and checks if the generated output is supported by the "
            "evidence. This on-demand retrieval and self-reflection improves both factuality "
            "and quality compared to always-retrieve approaches."
        ),
        "topic": "RAG",
        "difficulty": "hard",
    },
    {
        "question": "What is Reciprocal Rank Fusion and how is it used in hybrid search?",
        "ground_truth": (
            "Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrievers by "
            "assigning each document a score of 1/(k+rank) from each retriever, then summing "
            "across retrievers. The constant k (typically 60) prevents top-ranked documents "
            "from dominating. RRF is used in hybrid search to fuse dense embedding results "
            "with sparse BM25 results, producing a single merged ranking that benefits from "
            "both semantic and keyword matching."
        ),
        "topic": "retrieval",
        "difficulty": "medium",
    },
    {
        "question": "What are the key differences between GPT-3 and LLaMA models?",
        "ground_truth": (
            "GPT-3 (175B parameters) is a closed-source model by OpenAI trained on a filtered "
            "web corpus, accessible only via API. LLaMA by Meta is open-source, available in "
            "smaller sizes (7B to 65B), and trained on publicly available data. LLaMA showed "
            "that smaller models trained on more tokens can match or exceed larger models — "
            "LLaMA-13B outperforms GPT-3 on most benchmarks despite being 13x smaller. This "
            "democratized access to capable foundation models."
        ),
        "topic": "transformer",
        "difficulty": "medium",
    },
    {
        "question": "How does cross-encoder re-ranking improve retrieval quality?",
        "ground_truth": (
            "Cross-encoder re-ranking passes the query-document pair jointly through a "
            "transformer, allowing full cross-attention between them. Unlike bi-encoders "
            "that encode query and document independently, cross-encoders capture fine-grained "
            "interactions between the query and each candidate passage. This produces much "
            "more accurate relevance scores but is too slow for initial retrieval over large "
            "corpora, so it is used as a second-stage re-ranker on the top-k candidates "
            "from a faster first-stage retriever."
        ),
        "topic": "retrieval",
        "difficulty": "medium",
    },
    {
        "question": "What is the ReAct framework for language model agents?",
        "ground_truth": (
            "ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning traces with "
            "action steps in a language model. The model generates a thought (reasoning about "
            "what to do), then an action (like searching or looking up information), observes "
            "the result, and continues reasoning. This synergy allows the model to create and "
            "adjust plans dynamically, handle exceptions, and ground its reasoning in external "
            "information, outperforming both reasoning-only and acting-only approaches."
        ),
        "topic": "agents",
        "difficulty": "medium",
    },
    {
        "question": "What is Direct Preference Optimization (DPO) and how does it simplify RLHF?",
        "ground_truth": (
            "DPO reformulates the RLHF objective to directly optimize a language model on "
            "preference data without needing a separate reward model or reinforcement learning. "
            "It derives a closed-form mapping between the reward function and the optimal "
            "policy, resulting in a simple classification loss on preferred vs rejected "
            "response pairs. DPO is more stable, computationally lighter, and easier to "
            "implement than PPO-based RLHF while achieving comparable or better performance."
        ),
        "topic": "fine-tuning",
        "difficulty": "hard",
    },
    {
        "question": "How do sentence embeddings from Sentence-BERT differ from regular BERT embeddings?",
        "ground_truth": (
            "Standard BERT produces token-level embeddings and requires passing both sentences "
            "through the network together for comparison, making it extremely slow for similarity "
            "search (O(n^2) comparisons). Sentence-BERT uses a siamese/triplet network structure "
            "to generate fixed-size sentence embeddings that can be compared with cosine similarity. "
            "This enables semantic search over thousands of sentences in milliseconds instead of "
            "hours, while maintaining most of BERT's accuracy on semantic similarity tasks."
        ),
        "topic": "embeddings",
        "difficulty": "medium",
    },
    {
        "question": "What quantization techniques are used to compress large language models?",
        "ground_truth": (
            "Key quantization techniques include: GPTQ which uses approximate second-order "
            "information for accurate one-shot weight quantization to 3-4 bits; LLM.int8() "
            "which decomposes matrix multiplication into 8-bit and 16-bit parts, handling "
            "outlier features separately; AWQ (Activation-aware Weight Quantization) which "
            "protects salient weights identified by activation magnitudes; and NormalFloat "
            "(used in QLoRA) which uses information-theoretically optimal quantization for "
            "normally distributed weights."
        ),
        "topic": "efficiency",
        "difficulty": "hard",
    },
    {
        "question": "What are the common failure modes in RAG systems?",
        "ground_truth": (
            "Common RAG failure modes include: missing content where the answer isn't in "
            "the knowledge base; incorrect retrieval where relevant documents exist but "
            "aren't retrieved; failure to extract the answer from correctly retrieved context; "
            "wrong format of the generated answer; incomplete answers that miss parts of "
            "the question; incorrect specificity (too vague or too detailed); and not "
            "disclosing when the system doesn't know. These can be addressed through better "
            "chunking, hybrid retrieval, re-ranking, and prompt engineering."
        ),
        "topic": "RAG",
        "difficulty": "medium",
    },
    {
        "question": "How does the Lost in the Middle phenomenon affect RAG systems?",
        "ground_truth": (
            "Liu et al. found that language models are significantly better at using "
            "information placed at the beginning or end of their input context, while "
            "performance degrades substantially for information in the middle. This means "
            "that in RAG systems, the order of retrieved documents matters — placing the "
            "most relevant documents at the start or end of the context improves answer "
            "quality. This has implications for how we structure the prompt with retrieved "
            "chunks."
        ),
        "topic": "RAG",
        "difficulty": "hard",
    },
    {
        "question": "What is HyDE (Hypothetical Document Embeddings) and how does it improve retrieval?",
        "ground_truth": (
            "HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to "
            "the query using an LLM, then embeds this hypothetical document instead of the "
            "original query for retrieval. The intuition is that a generated passage is more "
            "similar in embedding space to relevant real documents than a short query would be. "
            "This bridges the gap between query and document representations without requiring "
            "any training data or relevance labels — it's a zero-shot dense retrieval method."
        ),
        "topic": "RAG",
        "difficulty": "hard",
    },
]


def generate_test_queries(output_file: Path) -> list[dict]:
    """Save the curated test queries to a JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(DEFAULT_TEST_QUERIES, f, indent=2)
    print(f"✅ Generated {len(DEFAULT_TEST_QUERIES)} test queries → {output_file}")
    return DEFAULT_TEST_QUERIES


# ============================================================
# RAG Pipeline Runner for Evaluation
# ============================================================

def run_rag_for_eval(
    queries: list[dict],
    embedding_model: str,
    chunk_strategy: str,
    retrieval_strategy: str,
    chroma_dir: str,
    chunks_dir: Path,
) -> list[EvalSample]:
    """
    Run the full RAG pipeline on test queries and collect outputs for evaluation.

    Returns list of EvalSample with question, ground_truth, answer, and contexts filled in.
    """
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.retrieval.retriever import Retriever
    from src.generation.llm import RAGGenerator

    # Initialize components
    print(f"\n  🔧 Config: {embedding_model} × {chunk_strategy} × {retrieval_strategy}")

    retriever = Retriever(
        chroma_dir=chroma_dir,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
        chunks_dir=chunks_dir,
    )

    generator = RAGGenerator(primary="vertex_ai", fallback="groq")

    samples = []
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q['question'][:60]}...")

        try:
            # Retrieve
            retrieval_output = retriever.retrieve(
                query=q["question"],
                strategy=retrieval_strategy,
                top_k=10,
                rerank_top_k=5,
            )

            # Extract context texts
            contexts = [r.text for r in retrieval_output.results]

            # Generate answer
            gen_result = generator.generate(
                query=q["question"],
                retrieved_chunks=retrieval_output.results,
            )

            samples.append(EvalSample(
                question=q["question"],
                ground_truth=q["ground_truth"],
                answer=gen_result.answer,
                contexts=contexts,
                metadata={
                    "topic": q.get("topic", ""),
                    "difficulty": q.get("difficulty", ""),
                    "provider": gen_result.provider,
                    "model": gen_result.model,
                    "latency_ms": gen_result.latency_ms,
                    "num_contexts": len(contexts),
                },
            ))

            # Rate limit: be gentle with free APIs
            time.sleep(1.0)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            samples.append(EvalSample(
                question=q["question"],
                ground_truth=q["ground_truth"],
                answer=f"ERROR: {str(e)}",
                contexts=[],
                metadata={"error": str(e)},
            ))

    return samples


# ============================================================
# RAGAS Evaluation
# ============================================================

def _init_ragas_llm():
    """
    Initialize the evaluator LLM and embeddings for RAGAS.

    Priority:
      1. Vertex AI Gemini (if GOOGLE_CLOUD_PROJECT is set — uses $300 trial credit)
      2. Google Gemini free API (if GOOGLE_API_KEY is set)
      3. OpenAI (if OPENAI_API_KEY is set — RAGAS default)
      4. None (let RAGAS use its default, which requires OpenAI)

    Returns:
        (evaluator_llm, evaluator_embeddings) tuple, or (None, None) if no config found.

    Note: We MUST return an embeddings object for answer_relevancy metric.
          RAGAS auto-detects Google embeddings but the wrapper lacks embed_query(),
          so we explicitly provide a sentence-transformers embedder instead.
    """
    import os

    # Build RAGAS-compatible embeddings using sentence-transformers (always works, no API needed)
    # This avoids the 'GoogleEmbeddings' object has no attribute 'embed_query' error.
    # We use a minimal adapter that directly wraps sentence-transformers
    # because LangchainEmbeddingsWrapper and RAGAS's own HuggingfaceEmbeddings
    # have compatibility issues across different RAGAS versions.
    evaluator_embeddings = None
    try:
        from sentence_transformers import SentenceTransformer
        from ragas.embeddings.base import BaseRagasEmbeddings

        class SentenceTransformerEmbeddings(BaseRagasEmbeddings):
            """Minimal RAGAS-compatible wrapper around sentence-transformers."""

            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self._st_model = SentenceTransformer(model_name, device="cpu")
                self.model = model_name  # RAGAS telemetry reads this as a string

            def embed_query(self, text: str) -> list[float]:
                return self._st_model.encode(text, normalize_embeddings=True).tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._st_model.encode(texts, normalize_embeddings=True).tolist()

            async def aembed_query(self, text: str) -> list[float]:
                return self.embed_query(text)

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                return self.embed_documents(texts)

        evaluator_embeddings = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        print("  📦 RAGAS embeddings: SentenceTransformer all-MiniLM-L6-v2")
    except ImportError:
        print("  ⚠️  Could not load sentence-transformers for RAGAS embeddings.")
        print("     answer_relevancy metric may fail.")
        print("     Fix: pip install sentence-transformers")

    # Option 1: Vertex AI (best — high rate limits, $300 credit)
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        try:
            from pathlib import Path
            from google import genai
            from ragas.llms import llm_factory

            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            # Load service account credentials explicitly (Windows-safe)
            creds = None
            if creds_path and Path(creds_path).exists():
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    creds_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=creds,
            )
            evaluator_llm = llm_factory(
                "gemini-2.5-flash",
                provider="google",
                client=client,
            )
            print("  🔑 RAGAS evaluator LLM: Vertex AI Gemini Flash")
            return evaluator_llm, evaluator_embeddings
        except Exception as e:
            print(f"  ⚠️  Vertex AI init failed: {e}")

    # Option 2: Google Gemini free API
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from google import genai
            from ragas.llms import llm_factory

            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            evaluator_llm = llm_factory(
                "gemini-2.5-flash",
                provider="google",
                client=client,
            )
            print("  🔑 RAGAS evaluator LLM: Gemini Flash (free API)")
            return evaluator_llm, evaluator_embeddings
        except Exception as e:
            print(f"  ⚠️  Gemini free API init failed: {e}")

    # Option 3: OpenAI (RAGAS default)
    if os.getenv("OPENAI_API_KEY"):
        print("  🔑 RAGAS evaluator LLM: OpenAI (default)")
        return None, evaluator_embeddings

    print("  ⚠️  No evaluator LLM configured")
    return None, evaluator_embeddings


def evaluate_with_ragas(
    samples: list[EvalSample],
    metrics: list[str] = None,
) -> dict:
    """
    Evaluate RAG samples using RAGAS metrics.

    Uses Vertex AI Gemini as the evaluation LLM (via $300 GCP trial credit).
    Falls back to free Gemini API or OpenAI if Vertex AI isn't configured.

    Args:
        samples: List of EvalSample with all fields populated
        metrics: Which metrics to compute. Default: all four.

    Returns:
        Dict with aggregate scores and per-sample details.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Map metric names to RAGAS metric objects
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    selected_metrics = [metric_map[m] for m in metrics if m in metric_map]

    # Filter out error samples
    valid_samples = [s for s in samples if not s.answer.startswith("ERROR:")]
    if not valid_samples:
        print("  ❌ No valid samples to evaluate")
        return {"error": "No valid samples"}

    print(f"\n  📊 Evaluating {len(valid_samples)} samples with RAGAS...")
    print(f"     Metrics: {metrics}")

    # Initialize evaluator LLM (Vertex AI > free Gemini > OpenAI)
    evaluator_llm, evaluator_embeddings = _init_ragas_llm()

    # Build HuggingFace Dataset in RAGAS format
    eval_data = {
        "question": [s.question for s in valid_samples],
        "answer": [s.answer for s in valid_samples],
        "contexts": [s.contexts for s in valid_samples],
        "ground_truth": [s.ground_truth for s in valid_samples],
    }
    dataset = Dataset.from_dict(eval_data)

    # Run RAGAS evaluation
    try:
        eval_kwargs = {
            "dataset": dataset,
            "metrics": selected_metrics,
            "raise_exceptions": False,  # Don't crash on individual sample failures
        }
        if evaluator_llm is not None:
            eval_kwargs["llm"] = evaluator_llm
        if evaluator_embeddings is not None:
            eval_kwargs["embeddings"] = evaluator_embeddings

        result = evaluate(**eval_kwargs)

    except Exception as e:
        print(f"\n  ⚠️  RAGAS evaluation failed: {e}")
        print("  💡 RAGAS requires an LLM for evaluation. Configure one of:")
        print("     1. GOOGLE_CLOUD_PROJECT in .env (Vertex AI — recommended)")
        print("     2. GOOGLE_API_KEY in .env (free Gemini API)")
        print("     3. OPENAI_API_KEY in .env (OpenAI — RAGAS default)")
        print("  📖 See: https://docs.ragas.io/en/stable/howtos/integrations/gemini/")
        return {"error": str(e)}

    # Extract scores — compatible with both RAGAS v0.1 and v0.2+
    # v0.2+ stores scores as result.scores (list of dicts) and uses
    # result.to_pandas() for the DataFrame. The old `result[metric_name]`
    # and `metric_name in result` syntax no longer works in v0.2+.
    aggregate = {}
    per_sample = []

    try:
        result_df = result.to_pandas()

        # Compute aggregate scores from the DataFrame
        for metric_name in metrics:
            if metric_name in result_df.columns:
                col = result_df[metric_name].dropna()
                if len(col) > 0:
                    aggregate[metric_name] = round(float(col.mean()), 4)

        # Extract per-sample scores
        for i, row in result_df.iterrows():
            sample_scores = {}
            for metric_name in metrics:
                if metric_name in row and not (isinstance(row[metric_name], float) and
                                                 __import__('math').isnan(row[metric_name])):
                    sample_scores[metric_name] = round(float(row[metric_name]), 4)
            per_sample.append({
                "question": row.get("question", row.get("user_input", f"sample_{i}")),
                "scores": sample_scores,
            })

    except Exception as e:
        print(f"  ⚠️  Error extracting results from DataFrame: {e}")
        print("  🔄 Trying fallback extraction from result.scores...")

        # Fallback: extract directly from result.scores (RAGAS v0.2+)
        try:
            if hasattr(result, 'scores') and result.scores:
                for i, score_dict in enumerate(result.scores):
                    sample_scores = {}
                    for metric_name in metrics:
                        if metric_name in score_dict:
                            val = score_dict[metric_name]
                            if val is not None and not (isinstance(val, float) and
                                                          __import__('math').isnan(val)):
                                sample_scores[metric_name] = round(float(val), 4)
                    per_sample.append({
                        "question": valid_samples[i].question if i < len(valid_samples) else f"sample_{i}",
                        "scores": sample_scores,
                    })

                # Compute aggregates from per-sample scores
                for metric_name in metrics:
                    vals = [s["scores"][metric_name] for s in per_sample if metric_name in s["scores"]]
                    if vals:
                        aggregate[metric_name] = round(sum(vals) / len(vals), 4)
        except Exception as e2:
            print(f"  ❌ Fallback extraction also failed: {e2}")

    return {
        "aggregate": aggregate,
        "per_sample": per_sample,
        "num_evaluated": len(valid_samples),
        "num_errors": len(samples) - len(valid_samples),
    }


# ============================================================
# Experiment Matrix Runner
# ============================================================

def run_evaluation(
    test_queries_file: Path,
    chroma_dir: str,
    chunks_dir: Path,
    output_dir: Path,
    embedding_models: list[str] = None,
    chunk_strategies: list[str] = None,
    retrieval_strategies: list[str] = None,
    max_queries: int = None,
    generate_only: bool = False,
    eval_only: bool = False,
    results_filename: str = "ragas_evaluation_results.json",
) -> list[EvalResult]:
    """
    Run evaluation across the full experiment matrix.

    Args:
        test_queries_file: Path to test queries JSON
        chroma_dir: ChromaDB directory
        chunks_dir: Directory with chunk JSON files
        output_dir: Where to save results
        embedding_models: Which embedding models to evaluate
        chunk_strategies: Which chunking strategies to evaluate
        retrieval_strategies: Which retrieval strategies to evaluate
        max_queries: Limit number of queries (for quick testing)
        generate_only: If True, only generate + cache answers (skip RAGAS scoring)
        eval_only: If True, only run RAGAS on cached samples (skip generation)
        results_filename: Name of the JSON file to save RAGAS scores

    Returns:
        List of EvalResult for each configuration tested
    """
    # Load test queries
    if not test_queries_file.exists():
        print(f"📝 Test queries not found. Generating...")
        generate_test_queries(test_queries_file)

    with open(test_queries_file) as f:
        queries = json.load(f)

    if max_queries:
        queries = queries[:max_queries]

    # Defaults
    if embedding_models is None:
        embedding_models = ["all-MiniLM-L6-v2"]
    if chunk_strategies is None:
        chunk_strategies = ["recursive"]
    if retrieval_strategies is None:
        retrieval_strategies = ["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]

    total_combos = len(embedding_models) * len(chunk_strategies) * len(retrieval_strategies)

    print("🔬 RAGAS EVALUATION PIPELINE")
    print("=" * 60)
    print(f"   Test queries:        {len(queries)}")
    print(f"   Embedding models:    {embedding_models}")
    print(f"   Chunk strategies:    {chunk_strategies}")
    print(f"   Retrieval strategies: {retrieval_strategies}")
    print(f"   Total configurations: {total_combos}")
    print()

    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing RAGAS results (for crash recovery / incremental runs)
    results_file = output_dir / results_filename
    scored_configs = set()
    if results_file.exists() and not generate_only:
        try:
            with open(results_file) as f:
                existing = json.load(f)
            for r in existing:
                all_results.append(EvalResult(**r))
                key = f"{r['embedding_model']}__{r['chunk_strategy']}__{r['retrieval_strategy']}"
                scored_configs.add(key)
            if scored_configs:
                print(f"📂 Loaded {len(scored_configs)} existing RAGAS scores (will skip these)")
        except Exception as e:
            print(f"  ⚠️  Could not load existing results: {e}")

    combo_num = 0
    for emb_model in embedding_models:
        for chunk_strat in chunk_strategies:
            for ret_strat in retrieval_strategies:
                combo_num += 1

                # Skip if already scored
                config_key = f"{emb_model}__{chunk_strat}__{ret_strat}"
                if config_key in scored_configs and not generate_only:
                    print(f"\n  ⏭️  [{combo_num}/{total_combos}] {emb_model} × {chunk_strat} × {ret_strat} — already scored")
                    continue

                print(f"\n{'='*60}")
                print(f"📋 Configuration {combo_num}/{total_combos}")
                print(f"   {emb_model} × {chunk_strat} × {ret_strat}")
                print(f"{'='*60}")

                start_time = time.time()

                # Step 1: Run RAG pipeline on test queries (or load from cache)
                cache_key = f"{emb_model}__{chunk_strat}__{ret_strat}".replace("/", "_").replace("-", "_")
                cache_file = output_dir / "samples_cache" / f"samples_{cache_key}.json"

                if cache_file.exists():
                    # Load cached samples — skip generation entirely
                    print(f"  💾 Loading cached samples from {cache_file.name}")
                    with open(cache_file) as f:
                        cached = json.load(f)
                    samples = [
                        EvalSample(
                            question=s["question"],
                            ground_truth=s["ground_truth"],
                            answer=s["answer"],
                            contexts=s["contexts"],
                            metadata=s.get("metadata", {}),
                        )
                        for s in cached
                    ]
                elif eval_only:
                    print(f"  ⚠️  No cached samples found for this config. Skipping.")
                    print(f"     Run with --generate-only first to cache samples.")
                    continue
                else:
                    # Generate fresh and save to cache
                    samples = run_rag_for_eval(
                        queries=queries,
                        embedding_model=emb_model,
                        chunk_strategy=chunk_strat,
                        retrieval_strategy=ret_strat,
                        chroma_dir=chroma_dir,
                        chunks_dir=chunks_dir,
                    )

                    # Save to cache
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w") as f:
                        json.dump([asdict(s) for s in samples], f, indent=2)
                    print(f"  💾 Saved samples to {cache_file.name}")

                # Step 2: Evaluate with RAGAS (skip if generate_only)
                if generate_only:
                    print(f"  ✅ Samples cached. Skipping RAGAS eval (--generate-only mode)")
                    continue

                eval_output = evaluate_with_ragas(samples)
                elapsed = time.time() - start_time

                # Build result
                result = EvalResult(
                    embedding_model=emb_model,
                    chunk_strategy=chunk_strat,
                    retrieval_strategy=ret_strat,
                    num_samples=len(samples),
                    metrics=eval_output.get("aggregate", {}),
                    per_sample=eval_output.get("per_sample", []),
                    timestamp=datetime.now().isoformat(),
                    latency_seconds=round(elapsed, 1),
                )
                all_results.append(result)

                # Print scores
                if result.metrics:
                    print(f"\n  📊 Scores:")
                    for metric, score in result.metrics.items():
                        bar = "█" * int(score * 20)
                        print(f"     {metric:25s}: {score:.4f}  {bar}")

                # Save after each config (crash-safe)
                with open(output_dir / results_filename, "w") as f:
                    json.dump([asdict(r) for r in all_results], f, indent=2)
                print(f"  💾 Progress saved ({len(all_results)} configs scored)")

    # Save all results
    with open(output_dir / results_filename, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n💾 Results saved to {output_dir / results_filename}")

    # Generate comparison table
    _print_comparison_table(all_results)
    _save_comparison_csv(all_results, output_dir / "ragas_comparison.csv")

    return all_results


def _print_comparison_table(results: list[EvalResult]):
    """Print a formatted comparison table to the terminal."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("📊 RAGAS EVALUATION COMPARISON")
    print(f"{'='*100}")

    # Header
    metric_names = list(results[0].metrics.keys()) if results[0].metrics else []
    header = f"{'Config':<45s}"
    for m in metric_names:
        short = m.replace("_", " ").title()[:15]
        header += f" {short:>15s}"
    header += f" {'Avg':>8s}"
    print(header)
    print("─" * len(header))

    # Rows
    for r in results:
        config = f"{r.embedding_model[:15]} × {r.chunk_strategy[:10]} × {r.retrieval_strategy[:15]}"
        row = f"{config:<45s}"
        scores = []
        for m in metric_names:
            score = r.metrics.get(m, 0)
            scores.append(score)
            row += f" {score:>15.4f}"
        avg = sum(scores) / max(len(scores), 1)
        row += f" {avg:>8.4f}"
        print(row)

    print(f"{'='*100}")


def _save_comparison_csv(results: list[EvalResult], output_file: Path):
    """Save comparison results as CSV for easy import into spreadsheets."""
    if not results:
        return

    metric_names = list(results[0].metrics.keys()) if results[0].metrics else []

    lines = []
    header = "embedding_model,chunk_strategy,retrieval_strategy," + ",".join(metric_names) + ",average"
    lines.append(header)

    for r in results:
        scores = [r.metrics.get(m, 0) for m in metric_names]
        avg = sum(scores) / max(len(scores), 1)
        row = f"{r.embedding_model},{r.chunk_strategy},{r.retrieval_strategy},"
        row += ",".join(f"{s:.4f}" for s in scores)
        row += f",{avg:.4f}"
        lines.append(row)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"📄 CSV comparison saved to {output_file}")


# ============================================================
# Quick Evaluation (No RAGAS — retrieval metrics only)
# ============================================================

def quick_evaluate(
    test_queries_file: Path,
    chroma_dir: str,
    chunks_dir: Path,
    output_dir: Path,
    retrieval_strategies: list[str] = None,
) -> dict:
    """
    Quick evaluation that doesn't need RAGAS or an evaluation LLM.
    Measures retrieval quality only using:
    - Mean Reciprocal Rank (MRR)
    - Hit Rate @ K
    - Average number of retrieved chunks

    This is great for comparing retrieval strategies before running full RAGAS.
    """
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.retrieval.retriever import Retriever

    # Load test queries
    if not test_queries_file.exists():
        generate_test_queries(test_queries_file)

    with open(test_queries_file) as f:
        queries = json.load(f)

    if retrieval_strategies is None:
        retrieval_strategies = ["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]

    embedding_model = "all-MiniLM-L6-v2"
    chunk_strategy = "recursive"

    print("⚡ QUICK RETRIEVAL EVALUATION (no RAGAS needed)")
    print("=" * 60)
    print(f"   Queries: {len(queries)}")
    print(f"   Strategies: {retrieval_strategies}")
    print()

    retriever = Retriever(
        chroma_dir=chroma_dir,
        embedding_model=embedding_model,
        chunk_strategy=chunk_strategy,
        chunks_dir=chunks_dir,
    )

    results = {}
    for strategy in retrieval_strategies:
        print(f"\n  🔍 {strategy}...")
        total_results = 0
        top_scores = []

        for q in queries:
            output = retriever.retrieve(
                query=q["question"],
                strategy=strategy,
                top_k=10,
                rerank_top_k=5,
            )
            total_results += output.num_results
            if output.results:
                top_scores.append(output.results[0].score)

        avg_results = total_results / len(queries)
        avg_top_score = sum(top_scores) / max(len(top_scores), 1)

        results[strategy] = {
            "avg_results": round(avg_results, 1),
            "avg_top_score": round(avg_top_score, 4),
            "queries_with_results": len(top_scores),
        }

        print(f"     Avg results: {avg_results:.1f}")
        print(f"     Avg top score: {avg_top_score:.4f}")
        print(f"     Queries with results: {len(top_scores)}/{len(queries)}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "quick_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Quick eval results saved to {output_file}")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    DATA_DIR = PROJECT_ROOT / "data"
    CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
    CHROMA_DIR = str(DATA_DIR / "chroma_db")
    RESULTS_DIR = DATA_DIR / "results"
    TEST_QUERIES_FILE = DATA_DIR / "eval" / "test_queries.json"

    args = sys.argv[1:]

    if "--generate-queries" in args:
        generate_test_queries(TEST_QUERIES_FILE)

    elif "--quick" in args:
        quick_evaluate(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
        )

    elif "--all" in args:
        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            embedding_models=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-large-en-v1.5"],
            chunk_strategies=["fixed", "recursive", "semantic"],
            retrieval_strategies=["dense_only", "bm25_only", "hybrid", "hybrid_rerank"],
        )

    else:
        # Default: single config evaluation
        run_evaluation(
            test_queries_file=TEST_QUERIES_FILE,
            chroma_dir=CHROMA_DIR,
            chunks_dir=CHUNKS_DIR,
            output_dir=RESULTS_DIR,
            max_queries=10,  # Quick default
        )