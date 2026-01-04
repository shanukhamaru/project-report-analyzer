import logging
import sys
from typing import Optional


LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(name)s] "
    "- %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    stream: Optional[object] = sys.stdout,
) -> None:
    """
    Configure global logging for the application.

    This should be called once at application startup.
    """

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.StreamHandler(stream)
        ],
        force=True,  # ensures reconfiguration in notebooks & Streamlit
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with the global configuration applied.
    """
    return logging.getLogger(name)
