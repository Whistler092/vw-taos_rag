from __future__ import annotations

import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


API_VERSION = "2024-08-01-preview"
COLLECTION_NAME = "vw-taos-collection"
PERSIST_DIR = "./data/chroma_db"
STRUCTURED_PDF = "./PDFs/VW_Taos_2024_OwnersManual.pdf"
UNSTRUCTURED_PDF = "./PDFs/VW_Taos_2024_QuickStartGuide.pdf"


RAG_PROMPT = """You are an assistant that answers questions about the Volkswagen Taos manuals.
Use only the provided context. If the context is not enough, clearly say what is missing.
Respond in Markdown.

Conversation history:
{history}

Context:
{context}

User question:
{query}
"""


def read_split_into_docs(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    return splitter.split_documents(documents)


def format_history(messages: List[dict]) -> str:
    # Use the last 6 turns to keep prompt size controlled.
    recent = messages[-6:]
    lines: List[str] = []
    for item in recent:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def validate_env() -> Tuple[bool, str]:
    required_vars = ["OPENAI_API_KEY", "MODEL", "EMBEDDING_MODEL", "API_URL"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    return True, ""


@st.cache_resource(show_spinner=False)
def build_rag_components() -> tuple[Chroma, AzureChatOpenAI]:
    load_dotenv()

    ok, message = validate_env()
    if not ok:
        raise RuntimeError(message)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    azure_endpoint = os.getenv("API_URL")

    embeddings = AzureOpenAIEmbeddings(
        deployment=embedding_model_name,
        azure_endpoint=azure_endpoint,
        api_version=API_VERSION,
        api_key=openai_api_key,
    )

    llm = AzureChatOpenAI(
        temperature=1,
        api_key=openai_api_key,
        deployment_name=model,
        api_version=API_VERSION,
        azure_endpoint=azure_endpoint,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    current_ids = vector_store.get().get("ids", [])
    if len(current_ids) == 0:
        docs = read_split_into_docs(STRUCTURED_PDF) + read_split_into_docs(UNSTRUCTURED_PDF)
        vector_store.add_documents(docs)

    return vector_store, llm


def retrieve_context(vector_store: Chroma, query: str, k: int = 5) -> tuple[str, list]:
    retrieved = vector_store.similarity_search_with_score(query=query, k=k)
    context = "\n\n".join([doc.page_content for doc, _ in retrieved])
    return context, retrieved


def ask_rag(vector_store: Chroma, llm: AzureChatOpenAI, query: str, history: str) -> tuple[str, list]:
    context, retrieved = retrieve_context(vector_store, query)
    prompt = RAG_PROMPT.format(history=history, context=context, query=query)
    response = llm.invoke(prompt)
    return response.content, retrieved


def render_sidebar() -> None:
    st.sidebar.header("Configuration")
    st.sidebar.write("This app uses your Azure OpenAI and Chroma configuration from .env")
    st.sidebar.code(
        "\n".join(
            [
                "Model used: gpt-5-nano",
                "Embedding Used: text-embedding-3-small-2"
            ]
        )
    )
    st.sidebar.write("PDFs used: VW_Taos_2024_OwnersManual.pdf and VW_Taos_2024_QuickStartGuide.pdf")


def main() -> None:
    st.set_page_config(page_title="VW Taos RAG Chat", page_icon="🚗", layout="wide")
    st.title("VW Taos Manuals - RAG Chat")
    st.caption("Ask questions about the 2024 Volkswagen Taos Owner's Manual and Quick Start Guide.")

    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello. Ask me anything about the VW Taos manuals, for example: What are the safety features?",
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching manuals and generating answer..."):
            try:
                vector_store, llm = build_rag_components()
                history_text = format_history(st.session_state.messages)
                answer, retrieved = ask_rag(vector_store, llm, user_input, history_text)
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

        st.markdown(answer)

        with st.expander("Retrieved context (top 5 chunks)"):
            for idx, (doc, score) in enumerate(retrieved, start=1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "n/a")
                st.markdown(f"**Chunk {idx}** | score: `{score:.4f}` | source: `{source}` | page: `{page}`")
                st.write(doc.page_content[:700] + ("..." if len(doc.page_content) > 700 else ""))
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
