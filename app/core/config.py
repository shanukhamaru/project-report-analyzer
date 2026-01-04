import os
from typing import Optional


class Settings:
    """
    Centralized application configuration.

    All values are sourced from environment variables to support
    local development, Docker, and cloud deployments (AWS).
    """

    # Environment
    ENV: str = os.getenv("ENV", "local")

    # AWS
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")

    # Embeddings / Models
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    # LLM (LM Studio / OpenAI-compatible)
    LMSTUDIO_API_KEY: str = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    LMSTUDIO_API_BASE: str = os.getenv("LMSTUDIO_API_BASE",
                                         "http://10.204.185.198:1234/v1",
                                         )
    LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL",
    "llama-2-7b-chat",
)


    # Vector Store
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "vector_store")

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def is_local(self) -> bool:
        return self.ENV.lower() == "local"


# Singleton settings object
settings = Settings()
