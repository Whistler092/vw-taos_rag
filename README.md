# VW Taos Manuals - RAG Chat

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Azure OpenAI** that answers questions about the 2024 Volkswagen Taos Owner's Manual and Quick Start Guide.

## Overview

This application uses a vector database (Chroma) to embed and retrieve relevant sections from VW Taos documentation, then leverages Azure OpenAI to generate contextual answers. Users can ask natural language questions and receive accurate, sourced responses with highlighted context chunks.

## Features

- 🚗 **VW Taos Documentation Search** – Ask questions about the 2024 Taos Owner's Manual and Quick Start Guide
- 📚 **Context Retrieval** – View the top 5 document chunks used to generate each answer
- 💬 **Conversation Memory** – Maintains chat history for multi-turn conversations
- 🏗️ **RAG Pipeline** – Hybrid search combining embeddings and LLM reasoning
- ☁️ **Azure OpenAI Integration** – Uses Azure-hosted GPT and embedding models
- 🗃️ **Persistent Vector Store** – Chroma database caches embeddings for fast retrieval

## Requirements

- Python 3.9+
- Azure OpenAI API credentials
- Environment variables configured (see **Setup** below)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=<your-azure-openai-api-key>
MODEL=<your-gpt-model-name>                    # e.g., gpt-4-turbo or gpt-5-nano
EMBEDDING_MODEL=<your-embedding-model-name>   # e.g., text-embedding-3-small
API_URL=<your-azure-openai-endpoint>          # e.g., https://your-instance.openai.azure.com/
```

### 3. Add PDF Documents

Place the VW Taos manuals in a `PDFs/` directory:

```
PDFs/
├── VW_Taos_2024_OwnersManual.pdf
└── VW_Taos_2024_QuickStartGuide.pdf
```

On first run, the app will automatically embed and persist these documents to `./data/chroma_db/`.

### 4. Run the App

```bash
streamlit run streamlit_rag_chat.py
```

The app will be available at `http://localhost:8501`.

## Usage

1. **Open the Chat Interface** – Start the Streamlit app
2. **Ask Questions** – Type natural language questions about the VW Taos (e.g., "What are the safety features?" or "How do I reset the infotainment system?")
3. **Review Context** – Expand the "Retrieved context (top 5 chunks)" section to see which manual pages informed the answer

## Project Structure

```
vw-taos_rag/
├── streamlit_rag_chat.py      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration (add before running)
├── data/
│   └── chroma_db/             # Persistent vector database
├── PDFs/                      # Document source folder (add your PDFs here)
└── README.md                  # This file
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'dotenv'`
Ensure `requirements.txt` is installed:
```bash
pip install -r requirements.txt
```

### Missing Environment Variables
The app validates required keys on startup. Check that all four variables in `.env` are set correctly:
- `OPENAI_API_KEY`
- `MODEL`
- `EMBEDDING_MODEL`
- `API_URL`

### No Documents Indexed
Place PDF files in the `PDFs/` directory and restart the app. On first run, it will automatically embed and cache them.

### Slow First Load
The first run embeds all documents into Chroma. Subsequent queries are cached and fast. Vector database is persisted in `./data/chroma_db/`.

## Deployment

When deploying to Streamlit Cloud or other services, ensure:
1. `requirements.txt` is committed to the repository
2. Environment variables are set in the deployment platform's secrets
3. PDFs are included in the repository or accessible via a data loader

## Technologies

- **Streamlit** – Interactive web UI
- **LangChain** – RAG orchestration and prompt management
- **Azure OpenAI** – LLM and embedding models
- **Chroma** – Vector database for semantic search
- **PyPDF** – PDF document loading

## License

Project repository structure only. Content and models subject to respective licenses (Streamlit, LangChain, Azure, OpenAI terms).
