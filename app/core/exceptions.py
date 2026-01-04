import os


class ProjectReportAnalyzerError(Exception):
    """
    Base exception for all application-specific errors.
    """
    pass


# ------------------------
# Ingestion & Parsing
# ------------------------

class IngestionError(ProjectReportAnalyzerError):
    """
    Raised when a PDF cannot be loaded or parsed correctly.
    """
    pass


class UnsupportedDocumentError(IngestionError):
    """
    Raised when the uploaded document format is invalid or unsupported.
    """
    pass


# ------------------------
# Chunking & Processing
# ------------------------

class ChunkingError(ProjectReportAnalyzerError):
    """
    Raised when semantic or structural chunking fails.
    """
    pass


# ------------------------
# Embeddings & Vector Store
# ------------------------

class EmbeddingError(ProjectReportAnalyzerError):
    """
    Raised when embedding generation fails.
    """
    pass


class VectorStoreError(ProjectReportAnalyzerError):
    """
    Raised when vector index creation, loading, or querying fails.
    """
    pass


# ------------------------
# Retrieval & RAG
# ------------------------

class RetrievalError(ProjectReportAnalyzerError):
    """
    Raised when document retrieval fails or returns invalid results.
    """
    pass


class RAGGenerationError(ProjectReportAnalyzerError):
    """
    Raised when answer generation fails or produces invalid output.
    """
    pass
