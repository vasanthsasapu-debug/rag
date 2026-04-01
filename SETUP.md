# Environment Setup Guide

## Prerequisites
- Python 3.10 or 3.11 (recommended)
- Git
- 8GB+ RAM (for embedding models)

## Step 1: Clone & Create Virtual Environment

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** First install may take 5-10 minutes (sentence-transformers downloads ~400MB of model files on first use).

## Step 3: Set Up API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys. You need **at least one** of these:

| Provider | Cost | What it powers |
|----------|------|----------------|
| **OpenAI** | ~$0.50 for full project | Embeddings + GPT-4o-mini generation |
| **Groq** | **Free tier** | Llama-3 70B generation (fast!) |
| **HuggingFace** | **Free** | Local embeddings (no API cost) |

**Budget-friendly setup:** Use HuggingFace embeddings (free, local) + Groq generation (free tier). Total cost: **$0**.

## Step 4: Download Papers

```bash
python src/ingestion/download_arxiv.py
```

This downloads ~85 arXiv papers (~10-15 min). Papers are saved to `data/raw_pdfs/`.

## Step 5: Verify Installation

```bash
python -c "
import langchain, chromadb, streamlit, ragas
print('All core packages loaded successfully!')
print(f'  LangChain: {langchain.__version__}')
print(f'  ChromaDB:  {chromadb.__version__}')
print(f'  Streamlit: {streamlit.__version__}')
"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `pip install` fails on `unstructured` | Try: `pip install unstructured` without the `[pdf]` extra first |
| CUDA/GPU errors | This project runs fine on CPU. GPU is optional. |
| `chromadb` build errors on Windows | Install Visual C++ Build Tools, or use WSL |
| Import errors after install | Make sure your virtual environment is activated |
