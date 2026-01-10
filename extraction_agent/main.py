"""Extraction agent main module."""
import json
import logging
from typing import TypedDict

from extraction_agent.config import MAX_QUERIES
from extraction_agent.prompts import EXTRACTION_PROMPT
from shared_config import create_llm

logger = logging.getLogger(__name__)


class ExtractionOutput(TypedDict):
    """Output from extraction agent."""
    queries: list[str]
    evidences: list[dict]  # {id, text}


def get_evidence(query: str, book_name: str) -> dict[str, str]:
    # Stub implementation
    fake_evidences = [
        {"id": "ev_1", "text": "Jacques Paganel is a French geographer known for his absent-mindedness."},
        {"id": "ev_2", "text": "Paganel has extensive knowledge of geography and travels the world."},
        {"id": "ev_3", "text": "He often forgets things and makes mistakes due to his absent-minded nature."},
        {"id": "ev_4", "text": "Paganel is a member of the Geographical Society and writes scholarly papers."},
    ]
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
    
    # Initialize LLM with OpenRouter config
    llm = create_llm()
    
    # Format prompt
    prompt = EXTRACTION_PROMPT.format(
        book_name=state["book_name"],
        character_name=state["character_name"],
        backstory=state["backstory"],
        max_queries=MAX_QUERIES,
    )
    
    logger.debug(f"Extraction prompt: {prompt}")
    
    # Generate queries
    logger.info("Generating queries via LLM")
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    logger.debug(f"LLM response: {response_text}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response (might have markdown code blocks)
        if "```" in response_text:
            # Extract JSON from code block
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        queries = json.loads(response_text)
        if not isinstance(queries, list):
            raise ValueError("Response is not a list")
        
        # Limit to MAX_QUERIES
        queries = queries[:MAX_QUERIES]
        
        logger.info(f"Generated {len(queries)} queries")
        
    except Exception as e:
        logger.error(f"Error parsing queries: {e}")
        logger.error(f"Response text: {response_text}")
        queries = []
    
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
