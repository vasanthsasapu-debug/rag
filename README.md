# 🔬 Production RAG System — Experiment-Driven Architecture

> A Retrieval-Augmented Generation system over 50+ arXiv ML/AI papers that systematically compares **48 configurations** across chunking strategies, embedding models, and retrieval methods — evaluated with RAGAS metrics on 20 curated test queries.

**This is not another RAG chatbot.** Most RAG projects pick one pipeline and ship it. This project treats every layer as a variable, runs a proper ablation study, and produces comparison tables that quantify which combination works best and *why*.

🏆 **Best configuration:** MiniLM-L6-v2 × recursive chunking × hybrid+rerank → **0.812 RAGAS average, 1.000 faithfulness**

---

## 📊 Key Results

### Top 10 Configurations (by RAGAS average)

| Embedding | Chunking | Retrieval | Faithfulness | Relevancy | Ctx Precision | Ctx Recall | **Average** |
|-----------|----------|-----------|:---:|:---:|:---:|:---:|:---:|
| MiniLM-L6-v2 | recursive | hybrid+rerank | 1.000 | 0.882 | 0.700 | 0.667 | **0.812** |
| text-embedding-005 | fixed | hybrid | 0.952 | 0.820 | 0.507 | 0.762 | **0.760** |
| text-embedding-005 | fixed | dense_only | 0.958 | 0.789 | 0.532 | 0.715 | **0.749** |
| BGE-large-en-v1.5 | fixed | hybrid | 0.969 | 0.832 | 0.424 | 0.708 | **0.734** |
| text-embedding-005 | fixed | hybrid+rerank | 0.954 | 0.774 | 0.445 | 0.733 | **0.727** |
| MPNet-base-v2 | fixed | bm25_only | 0.937 | 0.826 | 0.434 | 0.704 | **0.725** |
| BGE-large-en-v1.5 | fixed | bm25_only | 0.935 | 0.828 | 0.434 | 0.704 | **0.725** |
| text-embedding-005 | fixed | bm25_only | 0.872 | 0.826 | 0.434 | 0.704 | **0.709** |
| text-embedding-005 | recursive | dense_only | 0.922 | 0.814 | 0.425 | 0.673 | **0.708** |
| text-embedding-005 | recursive | hybrid | 0.901 | 0.802 | 0.421 | 0.702 | **0.706** |

### Findings

**1. Chunking is the biggest quality lever — and semantic chunking fails on academic papers**

| Strategy | Avg RAGAS | Configs |
|----------|:---------:|:-------:|
| Fixed (512 tokens) | 0.697 | 16 |
| Recursive (1000 chars) | 0.664 | 16 |
| Semantic (similarity-based) | 0.467 | 16 |

Semantic chunking scored **23 percentage points** lower than fixed chunking. Academic papers interleave concepts densely — the embedding-based topic boundaries don't align with useful retrieval boundaries.

**2. The smallest embedding model won the top spot.** MiniLM-L6-v2 (384 dims, 80MB) beat BGE-large (1024 dims, 1.2GB) and Vertex AI's text-embedding-005 when paired with hybrid+rerank. System-level optimization > component-level quality.

**3. Cross-encoder re-ranking provides the highest ceiling** but doesn't lift all configs equally. It turned a good config (MiniLM × recursive × hybrid = 0.609) into the best config (0.812) — a 33% improvement.

**4. Perfect faithfulness is achievable.** The winning config scored 1.0 — zero hallucination — through constrained system prompts, low temperature, and high-quality retrieved context.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  1. EMBED QUERY                                 │
│     Same model that indexed the chunks           │
│     (MiniLM / MPNet / BGE / Vertex AI)          │
└─────────────────────┬───────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
┌─────────┐   ┌─────────────┐   ┌───────────────┐
│ Dense   │   │   BM25      │   │   Hybrid      │
│ Search  │   │   Search    │   │   (RRF k=60)  │
│ (cosine)│   │   (TF-IDF)  │   │   0.7d + 0.3s │
└────┬────┘   └──────┬──────┘   └───────┬───────┘
     │               │                  │
     └───────────────┼──────────────────┘
                     ▼
          ┌─────────────────────┐
          │  Cross-Encoder      │
          │  Re-ranking         │
          │  (ms-marco-MiniLM)  │
          │  top-10 → top-5     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  LLM Generation     │
          │  Groq (Llama-3.1    │
          │  70B) → Gemini      │
          │  Flash fallback     │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Answer + Sources   │
          │  with citations     │
          └─────────────────────┘
```

### Experiment Matrix

| Dimension | Options | Count |
|-----------|---------|:-----:|
| Chunking | Fixed (512 tok), Recursive (1000 char), Semantic (cosine 0.75) | 3 |
| Embedding | MiniLM-L6 (384d), MPNet-base (768d), BGE-large (1024d), Vertex AI (768d) | 4 |
| Retrieval | Dense, BM25, Hybrid RRF, Hybrid+Rerank | 4 |
| **Total** | **3 × 4 × 4** | **48** |

Each configuration evaluated with **4 RAGAS metrics** on **20 curated test queries** = **3,840 individual metric scores**.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Free API key from [Groq](https://console.groq.com/keys) (30 RPM)
- Optional: [Google Gemini](https://aistudio.google.com/apikey) key for fallback

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

python -m venv ragenv
# Linux/Mac: source ragenv/bin/activate
# Windows:   ragenv\Scripts\Activate.ps1

pip install -r requirements.txt

cp .env.example .env
# Edit .env → add GROQ_API_KEY=gsk_...
```

### Run the Pipeline

```bash
# Step 1: Extract text from PDFs and chunk with 3 strategies
python run_ingestion.py

# Step 2: Build vector stores (3 models × 3 strategies = 9 collections)
python run_embeddings.py          # Full run (~30 min on CPU)
python run_embeddings.py --quick  # Quick test: MiniLM × recursive only

# Step 3: Query the system
python run_rag.py                          # Interactive mode
python run_rag.py --query "What is RAG?"   # Single query
python run_rag.py --compare                # Compare all 4 retrieval strategies

# Step 4: Run RAGAS evaluation
python run_eval.py --test     # 1 query, verify pipeline works
python run_eval.py --full     # 4 retrieval strategies, 20 queries
python run_eval.py --all      # Full 48-config experiment matrix

# Step 5: Launch the UI
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
rag-chatbot/
├── run_ingestion.py              # Step 1: PDF extraction → 3 chunking strategies
├── run_embeddings.py             # Step 2: Build ChromaDB vector stores
├── run_rag.py                    # Step 3: Interactive queries + strategy comparison
├── run_eval.py                   # Step 4: RAGAS evaluation pipeline
│
├── src/
│   ├── ingestion/
│   │   ├── download_arxiv.py     # arXiv paper downloader (API + manual fallback)
│   │   ├── pdf_extractor.py      # PyMuPDF extraction, multi-column handling
│   │   └── chunker.py            # 3 strategies: fixed, recursive, semantic
│   ├── retrieval/
│   │   ├── embeddings.py         # 4 embedding models + ChromaDB management
│   │   └── retriever.py          # 4 strategies: dense, BM25, hybrid, hybrid+rerank
│   ├── generation/
│   │   └── llm.py                # Groq → Vertex AI → Gemini fallback chain
│   └── evaluation/
│       └── evaluator.py          # RAGAS pipeline, 20 test queries, experiment matrix
│
├── app/
│   └── streamlit_app.py          # Chat UI + evaluation dashboard
│
├── configs/config.yaml           # All experiment parameters
├── data/
│   ├── raw_pdfs/                 # 50+ arXiv papers
│   ├── processed/chunks/         # 3 chunk files (fixed, recursive, semantic)
│   ├── chroma_db/                # 12 ChromaDB collections
│   └── results/                  # RAGAS scores, comparison tables
└── requirements.txt
```

**~4,500 lines** of Python across 12 files. No boilerplate, no framework lock-in.

---

## 🔧 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Direct library calls** (not LangChain/LlamaIndex) | Full control over experiment variables. No framework API churn. Better for understanding what's under the hood. |
| **ChromaDB** (not FAISS/Pinecone) | Zero infrastructure, Python-native, clean collection API. Sufficient for ~5,000 chunks. |
| **Groq primary** (Llama-3.1 70B) | 70B-parameter quality at $0. 800 tokens/sec on Groq LPU hardware. |
| **Separate judge LLM** for RAGAS | Gemini Flash judges the RAG output — different model from the generator to avoid self-evaluation bias. |
| **Temperature 0.1** | Near-deterministic for grounded generation. High enough to avoid repetitive phrasing. |
| **BM25 with simple tokenization** | No stemming/stopwords — fast, deterministic, reproducible. ML papers use consistent terminology. |
| **Cross-encoder lazy-loaded** | Only loads the 80MB re-ranker model when hybrid_rerank strategy is first used. |
| **Crash-safe evaluation** | Results saved after each config. Interrupted runs resume from last completed config. |

---

## 📈 RAGAS Metrics Explained

| Metric | What It Measures | Best Score |
|--------|-----------------|:----------:|
| **Faithfulness** | Is every claim in the answer supported by retrieved context? | 1.000 |
| **Answer Relevancy** | Does the answer address the question asked? | 0.882 |
| **Context Precision** | Are retrieved chunks relevant and ranked properly? | 0.700 |
| **Context Recall** | Does the retrieved context cover the ground truth? | 0.667 |

---

## 🔮 Known Limitations & Future Work

| Limitation | Planned Fix |
|------------|-------------|
| No query rewriting / HyDE | Add LLM-based query expansion |
| No multi-turn conversation | Add conversation history buffer + coreference resolution |
| No confidence threshold | Return "I don't know" when best chunk < 0.3 similarity |
| No streaming generation | Add token-by-token streaming via Groq/Gemini APIs |
| CPU-only embedding (~30 min) | Add GPU support for 10x faster indexing |
| PDF equation handling | Use Nougat/Marker for academic OCR |

---

## 💰 Cost

**$0.** Entire system runs on free-tier APIs and local models.

| Component | Provider | Cost |
|-----------|----------|:----:|
| Embeddings | sentence-transformers (local) | Free |
| Generation | Groq (Llama-3.1 70B) | Free |
| Fallback | Google Gemini Flash | Free |
| Evaluation judge | Vertex AI ($300 GCP trial) | Free |
| Vector store | ChromaDB (embedded) | Free |

---

## 👤 Author

**Vasanth Kumar Sasapu**
- Data Scientist | NIT Trichy
- [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
