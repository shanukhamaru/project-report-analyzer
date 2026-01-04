from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger
from app.core.exceptions import ChunkingError

logger = get_logger(__name__)


class HybridChunker:
    """
    Structure-aware chunker that preserves tables and
    semantically chunks narrative text.
    """

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, classified_elements: List[Dict]) -> List[Document]:
        """
        Convert classified document elements into LangChain Documents.

        Parameters
        ----------
        classified_elements : List[Dict]
            Output of PageClassifier.classify()

        Returns
        -------
        List[Document]
            Chunked documents ready for embeddings.
        """

        if not classified_elements:
            raise ChunkingError("No classified elements provided")

        logger.info(
            "Hybrid chunking started | elements=%d",
            len(classified_elements),
        )

        documents: List[Document] = []
        pending_title: str | None = None

        for item in classified_elements:
            el = item["element"]
            el_type = item["type"]
            metadata = item.get("metadata", {}).copy()

            text = getattr(el, "text", "").strip()
            if not text:
                continue

            # --- TITLE: buffer and attach to next narrative ---
            if el_type == "TITLE":
                pending_title = text
                continue

            # --- TABLE: keep intact ---
            if el_type == "TABLE":
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            **metadata,
                            "category": "TABLE",
                        },
                    )
                )
                pending_title = None
                continue

            # --- NARRATIVE / OTHER: semantic chunking ---
            if el_type in ("NARRATIVE", "OTHER"):
                if pending_title:
                    text = f"{pending_title}\n{text}"
                    pending_title = None

                splits = self.text_splitter.split_text(text)

                for chunk in splits:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "category": "NARRATIVE",
                            },
                        )
                    )

        if not documents:
            logger.error("Hybrid chunking produced no documents")
            raise ChunkingError("Chunking resulted in zero output documents")

        logger.info(
            "Hybrid chunking completed | chunks=%d",
            len(documents),
        )

        return documents
