import streamlit as st
from pathlib import Path
import tempfile
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.logging import setup_logging, get_logger
from app.core.exceptions import ProjectReportAnalyzerError
from app.ingestion.pdf_loader import PDFLoader
from app.chunking.page_classifier import PageClassifier
from app.chunking.hybrid_chunker import HybridChunker
from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FAISSStore
from app.rag.rag_pipeline import RAGPipeline

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
            loader = PDFLoader(strategy="fast")
            classifier = PageClassifier()
            chunker = HybridChunker()
            embedder = Embedder()
            store = FAISSStore(embedder)

            all_documents = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = Path(tmp.name)

                elements = loader.load(pdf_path)
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

            st.success("Documents indexed successfully.")

        except ProjectReportAnalyzerError as exc:
            logger.error("Document processing failed", exc_info=True)
            st.error(str(exc))

# -------------------------------------------------
# Query input & RAG execution
# -------------------------------------------------

query = st.text_input(
    "Ask a question about the uploaded reports:",
    placeholder="e.g. Compare the timelines of these projects",
)

if query and st.session_state.rag_pipeline:
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.rag_pipeline.run(query)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources")
            for c in result["citations"]:
                st.markdown(
                    f"- **{c.get('source')}**, Page {c.get('page')}"
                )

            with st.expander("Debug Info"):
                st.write("Retrieval Mode:", result.get("retrieval_mode"))
                st.write("Chunks Used:", len(result.get("retrieved_docs", [])))

        except Exception as exc:
            st.exception(exc)
