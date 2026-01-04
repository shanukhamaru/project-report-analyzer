from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

from app.core.logging import get_logger
from app.core.exceptions import EmbeddingError
from app.core.config import settings

logger = get_logger(__name__)


class Embedder:
    """
    Wrapper around SentenceTransformer embeddings.

    This class abstracts embedding generation so the underlying
    model or provider can be swapped without affecting callers.
    """

    def __init__(self) -> None:
        logger.info(
            "Initializing embedding model | model=%s",
            settings.EMBEDDING_MODEL_NAME,
        )

        try:
            self._embedding_model = SentenceTransformerEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize embedding model",
                exc_info=True,
            )
            raise EmbeddingError(
                "Embedding model initialization failed"
            ) from exc

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Parameters
        ----------
        documents : List[Document]
            Documents to embed.

        Returns
        -------
        List[List[float]]
            Embedding vectors.
        """

        if not documents:
            raise EmbeddingError("No documents provided for embedding")

        logger.info(
            "Embedding documents | count=%d",
            len(documents),
        )

        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self._embedding_model.embed_documents(texts)
        except Exception as exc:
            logger.error(
                "Document embedding failed",
                exc_info=True,
            )
            raise EmbeddingError(
                "Failed to generate document embeddings"
            ) from exc

        logger.info(
            "Embedding completed | vectors=%d | dim=%d",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
        )

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a user query.
        """

        if not query or not query.strip():
            raise EmbeddingError("Query text is empty")

        try:
            return self._embedding_model.embed_query(query)
        except Exception as exc:
            logger.error(
                "Query embedding failed",
                exc_info=True,
            )
            raise EmbeddingError(
                "Failed to generate query embedding"
            ) from exc
