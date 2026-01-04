from typing import List, Dict
from collections import Counter

from unstructured.documents.elements import (
    Element,
    Table,
    NarrativeText,
    Title,
)

from app.core.logging import get_logger

logger = get_logger(__name__)


class PageClassifier:
    """
    Classifies Unstructured document elements into high-level
    structural categories used by the chunking layer.
    """

    def classify(self, elements: List[Element]) -> List[Dict]:
        """
        Classify parsed document elements into structural roles.

        Parameters
        ----------
        elements : List[Element]
            Elements returned by Unstructured.

        Returns
        -------
        List[Dict]
            Classified elements with type and metadata.
        """

        logger.info(
            "Page classification started | elements=%d",
            len(elements),
        )

        classified: List[Dict] = []
        type_counter = Counter()

        for el in elements:
            element_type = self._classify_element(el)
            type_counter[element_type] += 1

            classified.append({
                "element": el,
                "type": element_type,
                "metadata": self._extract_metadata(el),
            })

        logger.info(
            "Page classification completed | distribution=%s",
            dict(type_counter),
        )

        return classified

    @staticmethod
    def _classify_element(element: Element) -> str:
        """
        Determine the structural type of a document element.
        """

        if isinstance(element, Table):
            return "TABLE"

        if isinstance(element, Title):
            return "TITLE"

        if isinstance(element, NarrativeText):
            return "NARRATIVE"

        return "OTHER"

    @staticmethod
    def _extract_metadata(element: Element) -> dict:
        """
        Extract minimal metadata required for downstream processing.
        """

        metadata = {}

        if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
            metadata["page"] = element.metadata.page_number

        return metadata
