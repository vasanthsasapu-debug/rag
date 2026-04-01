"""
RAG Chatbot — Streamlit Chat UI
Interactive chat interface with source display, confidence scores,
and retrieval strategy comparison.

Usage:
    streamlit run app/streamlit_app.py

Features:
    - Chat interface with streaming-style display
    - Source citation with paper titles and sections
    - Retrieval confidence scores and chunk viewer
    - Strategy selector (switch between dense, BM25, hybrid, hybrid+rerank)
    - Experiment comparison dashboard
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

# --- Project path setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# --- Constants ---
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
CHROMA_DIR = str(DATA_DIR / "chroma_db")
RESULTS_DIR = DATA_DIR / "results"


# ============================================================
# Page Config & Styling
# ============================================================

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Clean up default Streamlit padding */
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Chat message styling */
    .source-card {
        background-color: rgba(100, 100, 255, 0.08);
        border-left: 3px solid rgba(100, 100, 255, 0.5);
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .source-title {
        font-weight: 600;
        color: rgba(100, 100, 255, 0.9);
        margin-bottom: 0.2rem;
    }
    .source-section {
        color: rgba(150, 150, 150, 0.9);
        font-size: 0.8rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .score-high { background: rgba(0, 200, 100, 0.15); color: rgba(0, 180, 90, 1); }
    .score-mid { background: rgba(255, 180, 0, 0.15); color: rgba(220, 160, 0, 1); }
    .score-low { background: rgba(255, 80, 80, 0.15); color: rgba(220, 70, 70, 1); }

    .metric-card {
        background: rgba(100, 100, 255, 0.05);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: rgba(100, 100, 255, 0.9);
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(150, 150, 150, 0.9);
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None


# ============================================================
# Component Initialization (cached)
# ============================================================

@st.cache_resource
def init_retriever(embedding_model: str, chunk_strategy: str):
    """Initialize the retriever (cached across reruns)."""
    from src.retrieval.retriever import Retriever
    try:
        retriever = Retriever(
            chroma_dir=CHROMA_DIR,
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            chunks_dir=CHUNKS_DIR,
        )
        return retriever
    except Exception as e:
        st.error(f"❌ Failed to initialize retriever: {e}")
        return None


@st.cache_resource
def init_generator():
    """Initialize the LLM generator (cached across reruns)."""
    from src.generation.llm import RAGGenerator
    try:
        generator = RAGGenerator(primary="vertex_ai", fallback="groq")
        return generator
    except Exception as e:
        st.error(f"❌ Failed to initialize generator: {e}")
        return None


# ============================================================
# Sidebar — Configuration
# ============================================================

with st.sidebar:
    st.markdown("## 🔬 RAG Configuration")

    st.markdown("---")

    # Embedding model selector
    embedding_model = st.selectbox(
        "Embedding Model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "BAAI/bge-large-en-v1.5", "text-embedding-005"],
        index=0,
        help="Which embedding model to use for vector search",
    )

    # Chunking strategy
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        options=["recursive", "fixed", "semantic"],
        index=0,
        help="How documents were split into chunks",
    )

    # Retrieval strategy
    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid_rerank", "hybrid", "dense_only", "bm25_only"],
        index=0,
        help=(
            "dense_only = vector similarity | "
            "bm25_only = keyword matching | "
            "hybrid = dense + BM25 with RRF | "
            "hybrid_rerank = hybrid + cross-encoder"
        ),
    )

    st.markdown("---")

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        top_k = st.slider("Top-K retrieval", 3, 20, 10)
        rerank_top_k = st.slider("Re-rank Top-K", 3, 10, 5)
        show_chunks = st.checkbox("Show retrieved chunks", value=True)
        show_scores = st.checkbox("Show confidence scores", value=True)

    st.markdown("---")

    # Status indicators
    st.markdown("### 📡 System Status")

    # Check pipeline readiness
    chunks_exist = CHUNKS_DIR.exists() and list(CHUNKS_DIR.glob("*.json"))
    chroma_exists = Path(CHROMA_DIR).exists()

    if chunks_exist:
        st.success("✅ Chunks loaded")
    else:
        st.error("❌ No chunks found — run `python run_ingestion.py`")

    if chroma_exists:
        st.success("✅ Vector store ready")
    else:
        st.error("❌ No vector store — run `python run_embeddings.py`")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieval = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "Built by Vasanth Kumar Sasapu\n\n"
        "📊 Comparing 3 chunking × 3 embedding × 4 retrieval strategies"
    )


# ============================================================
# Helper Functions
# ============================================================

def score_to_badge(score: float) -> str:
    """Convert a score to a colored HTML badge."""
    if score >= 0.7:
        cls = "score-high"
    elif score >= 0.4:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score:.3f}</span>'


def render_sources(retrieval_results: list, show_scores: bool = True):
    """Render the source citations panel."""
    if not retrieval_results:
        return

    st.markdown("**📚 Sources**")

    for i, result in enumerate(retrieval_results[:5]):
        score_html = score_to_badge(result.score) if show_scores else ""

        st.markdown(
            f"""<div class="source-card">
                <div class="source-title">
                    [{i+1}] {result.doc_title[:80]} {score_html}
                </div>
                <div class="source-section">§ {result.section_heading}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_chunks_expander(retrieval_results: list):
    """Render expandable retrieved chunks for inspection."""
    if not retrieval_results:
        return

    with st.expander(f"🔍 View Retrieved Chunks ({len(retrieval_results)} results)", expanded=False):
        for i, result in enumerate(retrieval_results):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**[{result.rank}] {result.doc_title[:60]}**")
                st.markdown(f"*§ {result.section_heading}*")
                st.text(result.text[:500] + ("..." if len(result.text) > 500 else ""))
            with col2:
                st.metric("Score", f"{result.score:.4f}")
            st.markdown("---")


# ============================================================
# Main Chat Interface
# ============================================================

st.markdown("# 🔬 RAG Research Assistant")
st.markdown(
    f"Ask questions about ML/AI research papers. "
    f"Using **{retrieval_strategy}** retrieval with **{embedding_model}**."
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            render_sources(message["sources"], show_scores)

# Chat input
if prompt := st.chat_input("Ask about ML papers... (e.g., 'How does LoRA work?')"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        # Check if pipeline is ready
        if not chunks_exist or not chroma_exists:
            error_msg = (
                "⚠️ The RAG pipeline isn't ready yet. Please run:\n\n"
                "```bash\n"
                "python run_ingestion.py    # Extract and chunk PDFs\n"
                "python run_embeddings.py   # Build vector stores\n"
                "```\n\n"
                "Then restart this app."
            )
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        else:
            with st.spinner("🔍 Retrieving relevant context..."):
                # Initialize components
                retriever = init_retriever(embedding_model, chunk_strategy)
                generator = init_generator()

                if retriever is None or generator is None:
                    st.error("Failed to initialize pipeline components.")
                else:
                    try:
                        # Step 1: Retrieve
                        retrieval_output = retriever.retrieve(
                            query=prompt,
                            strategy=retrieval_strategy,
                            top_k=top_k,
                            rerank_top_k=rerank_top_k,
                        )
                        st.session_state.last_retrieval = retrieval_output

                        # Step 2: Generate
                        with st.spinner("🤖 Generating answer..."):
                            gen_result = generator.generate(
                                query=prompt,
                                retrieved_chunks=retrieval_output.results,
                            )

                        # Display answer
                        st.markdown(gen_result.answer)

                        # Display metadata
                        meta_cols = st.columns(3)
                        with meta_cols[0]:
                            st.caption(f"⏱️ {gen_result.latency_ms:.0f}ms" if gen_result.latency_ms else "")
                        with meta_cols[1]:
                            st.caption(f"🤖 {gen_result.provider}/{gen_result.model}")
                        with meta_cols[2]:
                            st.caption(f"📄 {retrieval_output.num_results} chunks retrieved")

                        # Display sources
                        render_sources(retrieval_output.results, show_scores)

                        # Display chunks if enabled
                        if show_chunks:
                            render_chunks_expander(retrieval_output.results)

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": gen_result.answer,
                            "sources": retrieval_output.results,
                            "metadata": {
                                "provider": gen_result.provider,
                                "model": gen_result.model,
                                "latency_ms": gen_result.latency_ms,
                                "strategy": retrieval_strategy,
                                "num_results": retrieval_output.num_results,
                            },
                        })

                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })


# ============================================================
# Bottom Section — Evaluation Dashboard (if results exist)
# ============================================================

results_file = RESULTS_DIR / "ragas_evaluation_results.json"
comparison_file = RESULTS_DIR / "strategy_comparison.json"

if results_file.exists() or comparison_file.exists():
    st.markdown("---")
    st.markdown("## 📊 Evaluation Dashboard")

    tab1, tab2 = st.tabs(["RAGAS Metrics", "Strategy Comparison"])

    # Tab 1: RAGAS Results
    with tab1:
        if results_file.exists():
            with open(results_file) as f:
                ragas_results = json.load(f)

            if ragas_results:
                # Show best config
                best = max(ragas_results, key=lambda r: sum(r["metrics"].values()) / max(len(r["metrics"]), 1))
                st.markdown(f"**🏆 Best configuration:** `{best['embedding_model']}` × "
                            f"`{best['chunk_strategy']}` × `{best['retrieval_strategy']}`")

                # Metric cards
                if best["metrics"]:
                    cols = st.columns(len(best["metrics"]))
                    for i, (metric, score) in enumerate(best["metrics"].items()):
                        with cols[i]:
                            st.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-value">{score:.2f}</div>
                                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                # Full comparison table
                st.markdown("### All Configurations")
                table_data = []
                for r in ragas_results:
                    row = {
                        "Embedding": r["embedding_model"],
                        "Chunking": r["chunk_strategy"],
                        "Retrieval": r["retrieval_strategy"],
                    }
                    row.update(r["metrics"])
                    if r["metrics"]:
                        row["Average"] = round(sum(r["metrics"].values()) / len(r["metrics"]), 4)
                    table_data.append(row)

                st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No RAGAS results yet. Run: `python src/evaluation/evaluator.py`")

    # Tab 2: Strategy Comparison
    with tab2:
        if comparison_file.exists():
            with open(comparison_file) as f:
                comparisons = json.load(f)

            if comparisons:
                for comp in comparisons:
                    st.markdown(f"**Q:** {comp['query']}")
                    comp_cols = st.columns(len(comp["results"]))
                    for i, (strategy, data) in enumerate(comp["results"].items()):
                        with comp_cols[i]:
                            st.metric(
                                strategy,
                                f"{data['top_score']:.4f}",
                                f"{data['num_results']} results",
                            )
                    st.markdown("---")
        else:
            st.info("No comparison data yet. Run: `python run_rag.py --compare`")


# ============================================================
# Sidebar — Example Questions
# ============================================================

with st.sidebar:
    st.markdown("### 💡 Try These Questions")
    example_questions = [
        "What is retrieval augmented generation?",
        "How does LoRA fine-tuning work?",
        "Explain the attention mechanism",
        "Compare dense vs sparse retrieval",
        "What is chain-of-thought prompting?",
        "How does Mixtral use mixture of experts?",
    ]
    for q in example_questions:
        if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
            # This sets the query for next rerun
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()