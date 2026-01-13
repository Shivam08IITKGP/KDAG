"""Extraction agent main module."""
import logging
from typing import TypedDict
from pydantic import BaseModel, Field

from shared_config import MAX_QUERIES, create_llm
from extraction_agent.prompts import EXTRACTION_PROMPT
from extraction_agent.character_summaries import get_character_summary
from Graphrag.pathway.retriever import retrieve_topk

logger = logging.getLogger(__name__)


class QueryList(BaseModel):
    """Structured output format for query extraction."""
    queries: list[str] = Field(description="List of search queries to retrieve evidence.")


class ExtractionOutput(TypedDict):
    """Output from extraction agent."""
    queries: list[str]
    evidences: list[dict]  # {id, text}


def get_evidence(query: str, book_name: str) -> dict[str, str]:
    # Stub implementation
    # fake_evidences = [
    #     {"id": "ev_1", "text": "Jacques Paganel is a French geographer known for his absent-mindedness."},
    #     {"id": "ev_2", "text": "Paganel has extensive knowledge of geography and travels the world."},
    #     {"id": "ev_3", "text": "He often forgets things and makes mistakes due to his absent-minded nature."},
    #     {"id": "ev_4", "text": "Paganel is a member of the Geographical Society and writes scholarly papers."},
    # ]
    fake_evidences = retrieve_topk(book_name, query, k=5)

    return {ev["id"]: ev["text"] for ev in fake_evidences}


def extract(state: dict) -> dict:
    """Extract queries and evidence.
    
    Args:
        state: PipelineState dictionary.
        
    Returns:
        Updated state with queries and evidences.
    """
    logger.info("Starting extraction agent")
    logger.info(f"Book: {state['book_name']}, Character: {state['character_name']}")
    
    # Get character summary
    character_summary = get_character_summary(state["book_name"], state["character_name"])
    if character_summary:
        logger.info(f"Using character summary for {state['character_name']}")
    else:
        logger.warning(f"No character summary found for {state['character_name']}")
        character_summary = "No canonical character information available."
    
    # Initialize LLM
    llm = create_llm()
    
    # Format prompt
    prompt = EXTRACTION_PROMPT.format(
        book_name=state["book_name"],
        character_name=state["character_name"],
        backstory=state["backstory"],
        max_queries=MAX_QUERIES,
        character_summary=character_summary,
    )
    
    # Generate queries with structured output
    logger.info("Generating queries via structured LLM")
    
    queries = []
    try:
        structured_llm = llm.with_structured_output(QueryList)
        result = structured_llm.invoke(prompt)
        queries = result.queries
        logger.info(f"Generated {len(queries)} queries")
    except Exception as e:
        logger.error(f"Structured output failed: {e}")
        # Very minimal fallback if everything fails
        queries = []

    
    # Limit to MAX_QUERIES
    if len(queries) > MAX_QUERIES:
        logger.info(f"Limiting queries from {len(queries)} to {MAX_QUERIES}")
        queries = queries[:MAX_QUERIES]
    
    # Get evidence for each query
    all_evidences = {}
    for query in queries:
        logger.info(f"Getting evidence for query: {query}")
        evidence_dict = get_evidence(query, state["book_name"])
        all_evidences.update(evidence_dict)
    
    # Convert evidence dict to list of dicts
    evidences = [{"id": ev_id, "text": ev_text} for ev_id, ev_text in all_evidences.items()]
    
    logger.info(f"Retrieved {len(evidences)} evidence items")
    
    # Create output
    output: ExtractionOutput = {
        "queries": queries,
        "evidences": evidences,
    }
    
    # Update state
    updated_state = state.copy()
    updated_state["queries"] = output["queries"]
    updated_state["evidences"] = output["evidences"]
    
    logger.info("Extraction agent completed")
    return updated_state
