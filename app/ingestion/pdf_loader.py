from typing import List

from streamlit.runtime.uploaded_file_manager import UploadedFile
from unstructured.partition.pdf import partition_pdf

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFLoader:
    """
    Loads and parses uploaded PDF files using Unstructured.
    Designed for Streamlit UploadedFile objects.
    """

    def load(self, uploaded_files: List[UploadedFile]):
        all_elements = []

        for uploaded_file in uploaded_files:
            logger.info(
                "PDF ingestion started | file=%s",
                uploaded_file.name,
            )

            elements = partition_pdf(
                file=uploaded_file,
                strategy="fast",              # Windows-safe
                infer_table_structure=True,
            )

            # Attach source metadata early
            for el in elements:
                if hasattr(el, "metadata"):
                    el.metadata.source = uploaded_file.name

            all_elements.extend(elements)

        logger.info(
            "PDF ingestion completed | total_elements=%d",
            len(all_elements),
        )

        return all_elements
