"""Cache for tracking processed evidence IDs to avoid redundant processing."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILE = Path("graph_creator_agent/evidence_cache.json")
EVIDENCE_CACHE: dict[str, set[str]] = {}

def load_cache():
    """Load cache from JSON file."""
    global EVIDENCE_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                # Convert lists back to sets
                EVIDENCE_CACHE = {k: set(v) for k, v in data.items()}
            logger.info(f"Loaded evidence cache from {CACHE_FILE}")
        except Exception as e:
            logger.error(f"Error loading evidence cache: {e}")
            EVIDENCE_CACHE = {}
    else:
        logger.info("No existing evidence cache found, starting fresh.")
        EVIDENCE_CACHE = {}

def save_cache():
    """Save cache to JSON file."""
    try:
        # Convert sets to lists for JSON serialization
        data = {k: list(v) for k, v in EVIDENCE_CACHE.items()}
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved evidence cache to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving evidence cache: {e}")

# Load cache on module import
load_cache()
