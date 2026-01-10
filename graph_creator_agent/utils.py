"""Utility functions for graph creator agent."""
import logging

logger = logging.getLogger(__name__)


def filter_new_evidence(evidences: list[dict], cache_key: str) -> list[dict]:
    """Filter out evidence that has already been processed.
    
    Args:
        evidences: List of evidence dicts with 'id' and 'text' keys.
        cache_key: Cache key (book_name_character_name).
        
    Returns:
        List of new evidence dicts.
    """
    from graph_creator_agent.cache import EVIDENCE_CACHE
    
    processed_ids = EVIDENCE_CACHE.get(cache_key, set())
    new_evidences = [ev for ev in evidences if ev["id"] not in processed_ids]
    
    logger.info(f"Filtered evidence: {len(new_evidences)} new out of {len(evidences)} total")
    return new_evidences
