from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

from app.core.logging import get_logger
from app.core.exceptions import IngestionError, UnsupportedDocumentError

logger = get_logger(__name__)


class PDFLoader:
    """
    Loads and parses PDF documents using Unstructured.
    """

    def __init__(self, strategy: str = "fast") -> None:
        self.strategy = strategy

    def load(self, pdf_path: Path) -> List[Element]:
        logger.info("PDF ingestion started | file=%s", pdf_path.name)

        self._validate_pdf(pdf_path)

        try:
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=self.strategy,
                infer_table_structure=True,
            )
        except Exception as exc:
            logger.error(
                "PDF parsing failed | file=%s",
                pdf_path.name,
                exc_info=True,
            )
            raise IngestionError("Failed to parse PDF") from exc

        if not elements:
            logger.error("No elements extracted | file=%s", pdf_path.name)
            raise IngestionError("PDF parsing returned no content")

        logger.info(
            "PDF ingestion completed | file=%s | elements=%d",
            pdf_path.name,
            len(elements),
        )

        return elements

    @staticmethod
    def _validate_pdf(pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise IngestionError(f"File not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedDocumentError(
                f"Unsupported file type: {pdf_path.suffix}"
            )
