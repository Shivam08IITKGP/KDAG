"""Graph creator agent main module."""
import logging

from graph_creator_agent.cache import EVIDENCE_CACHE, save_cache
from graph_creator_agent.graph_store import (
    load_graph,
    add_triplets,
    save_graph,
    sanitize_filename,
)
from graph_creator_agent.utils import filter_new_evidence
from graph_creator_agent.extractor import generate_triplets

logger = logging.getLogger(__name__)


def create_graph(state: dict) -> dict:
    """Create or update knowledge graph from evidence.
    
    Args:
        state: PipelineState dictionary.
        
    Returns:
        Updated state with graph_path.
    """
    logger.info("Starting graph creator agent")
    logger.info(f"Book: {state['book_name']}, Character: {state['character_name']}")
    
    book_name = state["book_name"]
    character_name = state["character_name"]
    evidences = state["evidences"]
    
    # Create cache key
    cache_key = f"{book_name}_{character_name}"
    
    # Load existing graph
    graph = load_graph(book_name, character_name)
    
    # Filter new evidence
    new_evidences = filter_new_evidence(evidences, cache_key)
    
    if not new_evidences:
        logger.info("No new evidence to process")
        safe_book = sanitize_filename(book_name)
        safe_char = sanitize_filename(character_name)
        graph_path = f"graph_creator_agent/graph/{safe_book}_{safe_char}.graphml"
        updated_state = state.copy()
        updated_state["graph_path"] = graph_path
        return updated_state
    
    logger.info(f"Processing {len(new_evidences)} new evidence items")
    
    # Initialize LLM with OpenRouter config
    from shared_config import create_llm
    llm = create_llm()
    
    # Generate triplets
    triplets = generate_triplets(new_evidences, llm, character_name)
    
    if triplets:
        # Add triplets to graph
        add_triplets(graph, triplets)
        
        # Save graph
        graph_path = save_graph(graph, book_name, character_name)
        
        # Update cache with new evidence IDs
        if cache_key not in EVIDENCE_CACHE:
            EVIDENCE_CACHE[cache_key] = set()
        
        new_evidence_ids = {ev["id"] for ev in new_evidences}
        EVIDENCE_CACHE[cache_key].update(new_evidence_ids)
        save_cache()
        
        logger.info(f"Updated cache with {len(new_evidence_ids)} new evidence IDs")
    else:
        logger.warning("No triplets generated from new evidence")
        safe_book = sanitize_filename(book_name)
        safe_char = sanitize_filename(character_name)
        graph_path = f"graph_creator_agent/graph/{safe_book}_{safe_char}.graphml"
    
    # Update state
    updated_state = state.copy()
    updated_state["graph_path"] = graph_path
    
    logger.info("Graph creator agent completed")
    return updated_state
