"""
logger.py
---------
Centralised logging factory for the trade-book project.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
    log.debug("Step reward: %.4f", reward)

Log levels:
    DEBUG   — per-step internals (disabled in prod)
    INFO    — episode summaries, training milestones
    WARNING — recovered errors, unusual market conditions
    ERROR   — unrecoverable failures
"""
import logging
import logging.handlers
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Log directory — sits at project root /logs/
# ---------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOG_DIR / "tradebook.log"

# ---------------------------------------------------------------
# Format
# ---------------------------------------------------------------
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------
# One-time root logger configuration
# Only configure once even if get_logger() is called many times
# ---------------------------------------------------------------
def _configure_root() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # Rotating file handler — DEBUG and above, 5 MB × 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger if needed."""
    _configure_root()
    return logging.getLogger(name)
