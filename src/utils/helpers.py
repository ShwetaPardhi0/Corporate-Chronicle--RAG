"""
Helpers — shared utility functions used across the project.
"""
import os
import logging

logger = logging.getLogger(__name__)


def load_documents_from_dir(docs_dir: str = "documents") -> list[dict]:
    """Scan a directory and return a list of source dicts for the pipeline.

    Supports .pdf and .txt files. Hidden files (starting with '.') are skipped.

    Args:
        docs_dir: Path to the directory containing source documents.

    Returns:
        List of source dicts with 'type', 'path', and 'metadata'.
    """
    sources = []

    if not os.path.exists(docs_dir):
        logger.warning(f"[HELPERS] Documents directory '{docs_dir}' does not exist.")
        return sources

    for filename in os.listdir(docs_dir):
        if filename.startswith("."):
            continue
        filepath = os.path.join(docs_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in (".pdf", ".txt"):
                sources.append({
                    "type": "file",
                    "path": filepath,
                    "metadata": {"source_name": filename, "file_type": ext.lstrip(".")},
                })
            else:
                logger.debug(f"[HELPERS] Skipping unsupported file type: {filename}")

    logger.info(f"[HELPERS] Found {len(sources)} document(s) in '{docs_dir}'")
    return sources


def format_sources(results: list[dict]) -> str:
    """Format retrieval results as a human-readable string for CLI output.

    Args:
        results: List of result dicts (each has 'chunk', 'score', 'rank').

    Returns:
        Formatted multi-line string.
    """
    if not results:
        return "  (no sources retrieved)"

    lines = []
    for res in results:
        chunk = res["chunk"]
        score = res["score"]
        rank = res["rank"]
        src = chunk.get("source", "unknown")
        snippet = chunk.get("text", "")[:120].replace("\n", " ")
        lines.append(f"  [{rank}] Score: {score:.4f} | Src: {src}")
        lines.append(f"      \"{snippet}...\"")

    return "\n".join(lines)


def setup_logging(log_file: str = "logs/app.log", level: int = logging.INFO) -> None:
    """Configure root logger to write to both console and a log file.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
