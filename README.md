# Production Grade RAG Pipeline

A highly modular, advanced Retrieval-Augmented Generation (RAG) system built with Langchain, Streamlit, ChromaDB, and Groq. 

This project was built in phases to create a production-ready application that goes beyond simple vector search, utilizing a hybrid search strategy and cross-encoder reranking to ensure precise, context-aware answers from your documents.

## 🚀 Features Implemented So Far

### Phase 1: Modular Naive RAG with Memory
- **PDF Ingestion engine**: Automatically parses and recursively chunks PDFs from a `data/` folder.
- **Local Embeddings**: Uses Ollama's `nomic-embed-text` to generate embeddings securely on your machine.
- **Vector Database**: Utilizes `ChromaDB` for local, persistent storage of document vectors.
- **Conversational Memory**: Maintains chat history using Langchain's `RunnableWithMessageHistory` to allow for contextual follow-up questions.
- **Lightning Fast Inference**: Uses the Groq API (`llama3-8b-8192`) for near-instant language generation.

### Phase 1.5: Streamlit Web UI
- Replaced the terminal loop with a sleek `Streamlit` interface.
- Includes a dedicated sidebar to ingest documents dynamically.
- Maintains chat history and presents clickable sources for every answer.

### Phase 2: Hybrid Search & Reranking (Advanced RAG)
- **Sparse Retrieval (Keyword Search)**: Implemented `rank_bm25` to retain exact-match keyword search capabilities (vital for serial numbers, acronyms, etc.).
- **Hybrid Fusion**: Merges the semantic results from ChromaDB with the keyword results from BM25 using **Reciprocal Rank Fusion (RRF)** via Langchain's `EnsembleRetriever`.
- **Cross-Encoder Reranking**: Utilizes `FlashrankRerank` to take the top 20 fused results and strictly score them against the user's prompt, pruning the context window down to the absolute best 3 chunks before sending it to the LLM.

---

## 🛠️ Project Structure

```text
production_grade_RAG/
│
├── data/                       # Drop your PDF files here
├── db/                         # ChromaDB and raw text chunks are persisted here
├── src/
│   ├── config.py               # Environment variable management
│   ├── document_processor.py   # PDF loading and recursive text splitting
│   ├── vector_store.py         # Chroma DB logic and chunk persistence
│   └── rag_engine.py           # Langchain pipeline (Hybrid Retriever + Reranker + LLM)
│
├── .env                        # API keys and local URLs
├── app.py                      # Streamlit frontend application
├── main.py                     # CLI entrypoint for ingestion and terminal chat
└── requirements.txt            # Project dependencies
```

---

## 💻 Getting Started

### 1. Prerequisites
Ensure you have the following installed:
*   Python 3.10+
*   [Ollama](https://ollama.com/) running locally.

Before running the app, pull the required local models via Ollama in your terminal:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Environment Setup
Create a `.env` file in the root directory and add your Groq API key:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### 3. Installation
It is recommended to use a Conda environment or venv:
```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
```

### 4. Running the App
1. Place your target PDFs into the `data/` folder.
2. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open the provided `localhost` URL in your browser.
4. Click **"Ingest Documents"** in the sidebar.
5. Start chatting with your documents!
