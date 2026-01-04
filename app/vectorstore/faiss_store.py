from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.core.logging import get_logger
from app.core.exceptions import VectorStoreError
from app.core.config import settings
from app.embeddings.embedder import Embedder

logger = get_logger(__name__)


class FAISSStore:
    """
    FAISS vector store wrapper for indexing and retrieval.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._vectorstore: FAISS | None = None

    def build(self, documents: List[Document]) -> None:
        """
        Build a FAISS index from documents.
        """

        if not documents:
            raise VectorStoreError("No documents provided for indexing")

        logger.info(
            "Building FAISS index | documents=%d",
            len(documents),
        )

        try:
            self._vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self._embedder._embedding_model,
            )
        except Exception as exc:
            logger.error(
                "FAISS index creation failed",
                exc_info=True,
            )
            raise VectorStoreError("Failed to build FAISS index") from exc

        logger.info("FAISS index built successfully")

    def save(self) -> None:
        """
        Persist the FAISS index to disk.
        """

        if not self._vectorstore:
            raise VectorStoreError("FAISS index is not initialized")

        path = Path(settings.VECTOR_STORE_DIR)
        path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving FAISS index | path=%s",
            path.resolve(),
        )

        try:
            self._vectorstore.save_local(str(path))
        except Exception as exc:
            logger.error(
                "Failed to save FAISS index",
                exc_info=True,
            )
            raise VectorStoreError("Failed to save FAISS index") from exc

    def load(self) -> None:
        """
        Load a persisted FAISS index from disk.
        """

        path = Path(settings.VECTOR_STORE_DIR)

        if not path.exists():
            raise VectorStoreError(
                f"FAISS index directory not found: {path}"
            )

        logger.info(
            "Loading FAISS index | path=%s",
            path.resolve(),
        )

        try:
            self._vectorstore = FAISS.load_local(
                str(path),
                self._embedder._embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to load FAISS index",
                exc_info=True,
            )
            raise VectorStoreError("Failed to load FAISS index") from exc

        logger.info("FAISS index loaded successfully")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search.
        """

        if not self._vectorstore:
            raise VectorStoreError("FAISS index is not initialized")

        logger.info(
            "Similarity search | k=%d | query='%s'",
            k,
            query,
        )

        try:
            return self._vectorstore.similarity_search(query, k=k)
        except Exception as exc:
            logger.error(
                "Similarity search failed",
                exc_info=True,
            )
            raise VectorStoreError("Similarity search failed") from exc

    def mmr_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """
        Perform Max Marginal Relevance (MMR) search.
        """

        if not self._vectorstore:
            raise VectorStoreError("FAISS index is not initialized")

        logger.info(
            "MMR search | k=%d | fetch_k=%d | lambda=%s",
            k,
            fetch_k,
            lambda_mult,
        )

        try:
            return self._vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        except Exception as exc:
            logger.error(
                "MMR search failed",
                exc_info=True,
            )
            raise VectorStoreError("MMR search failed") from exc
