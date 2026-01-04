import streamlit as st
import tempfile
import sys
from pathlib import Path

# -------------------------------------------------
# Path setup
# -------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# -------------------------------------------------
# Imports
# -------------------------------------------------

from app.core.logging import setup_logging, get_logger
from app.core.exceptions import ProjectReportAnalyzerError
from app.ingestion.pdf_loader import PDFLoader
from app.chunking.page_classifier import PageClassifier
from app.chunking.hybrid_chunker import HybridChunker
from app.vectorstore.faiss_store import FAISSStore
from app.rag.rag_pipeline import RAGPipeline
from app.embeddings.embedder import Embedder


# -------------------------------------------------
# App setup
# -------------------------------------------------

setup_logging()
logger = get_logger(__name__)

st.set_page_config(
    page_title="Project Report Analyzer",
    layout="wide",
)

st.title("📄 Project Report Analyzer")
st.caption(
    "Upload project reports and ask questions. "
    "Answers are generated strictly from the documents with citations."
)

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

# -------------------------------------------------
# File upload
# -------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Project Reports (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

# -------------------------------------------------
# Ingestion & indexing
# -------------------------------------------------

if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing documents..."):
        try:
            loader = PDFLoader()
            classifier = PageClassifier()
            chunker = HybridChunker()
            embedder = Embedder()
            store = FAISSStore(embedder)


            all_documents = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = Path(tmp.name)

                elements = loader.load([uploaded_file])
                classified = classifier.classify(elements)
                documents = chunker.chunk(classified)

                # Attach source metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)

            store.build(all_documents)
            store.save()

            rag_pipeline = RAGPipeline(store)

            st.session_state.vectorstore = store
            st.session_state.rag_pipeline = rag_pipeline

            st.success("Documents indexed successfully. You can start chatting below.")

        except ProjectReportAnalyzerError as exc:
            logger.error("Document processing failed", exc_info=True)
            st.error(str(exc))

# -------------------------------------------------
# Render chat history
# -------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------
# Chat input
# -------------------------------------------------

query = st.chat_input(
    "Ask a question about the uploaded reports (e.g. Compare the timelines of these projects)"
)

# -------------------------------------------------
# RAG execution with streaming UI
# -------------------------------------------------

if query and st.session_state.rag_pipeline:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            result = st.session_state.rag_pipeline.run(query)

            # Render streamed / accumulated answer
            full_answer = result["answer"]
            placeholder.markdown(full_answer)

            # Show sources below the answer
            if result.get("citations"):
                st.markdown("**Sources:**")
                for c in result["citations"]:
                    st.markdown(
                        f"- **{c.get('source')}**, Page {c.get('page')}"
                    )

            # Store assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": full_answer}
            )

        except Exception as exc:
            logger.error("RAG execution failed", exc_info=True)
            placeholder.error("An error occurred while generating the answer.")
            st.exception(exc)
