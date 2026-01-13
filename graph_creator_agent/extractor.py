import logging
from langchain_core.language_models.chat_models import BaseChatModel

from graph_creator_agent.prompts import TRIPLET_EXTRACTION_PROMPT
from graph_creator_agent.types import TripletList

logger = logging.getLogger(__name__)


def generate_triplets(evidence_list: list[dict], llm: BaseChatModel, character_name: str = "", backstory: str = "") -> tuple[list[dict], str]:
    """Generate triplets from evidence using LLM with structured output.
    
    Args:
        evidence_list: List of evidence dicts with 'id' and 'text' keys.
        llm: ChatOpenAI instance configured with structured output.
        character_name: Name of the target character for focused extraction.
        backstory: The backstory text to validate (used for context).
        
    Returns:
        Tuple of (List of Triplet dicts, graph_summary string).
    """
    if not evidence_list:
        return [], ""

    all_triplets = []
    graph_summary = ""
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(TripletList)
    
    # helper to format evidence list
    formatted_evidence = []
    for ev in evidence_list:
        formatted_evidence.append(f"ID: {ev['id']}\nText: {ev['text']}\n")
    
    combined_evidence_text = "\n---\n".join(formatted_evidence)
    
    logger.info(f"Generating triplets for {len(evidence_list)} evidence items in bulk")
    
    # Format prompt
    prompt = TRIPLET_EXTRACTION_PROMPT.format(
        evidence_text=combined_evidence_text,
        character_name=character_name,
        backstory=backstory
    )
    
    try:
        # Get structured LLM response (this returns a TripletList Pydantic object)
        response = structured_llm.invoke(prompt)
        
        # Extract triplets and summary from Pydantic object
        triplets = response.triplets
        graph_summary = response.graph_summary
        
        # Convert to list of dicts for the graph store
        for triplet in triplets:
            all_triplets.append({
                "subject": triplet.subject,
                "relation": triplet.relation,
                "object": triplet.object,
                "evidence_id": triplet.evidence_id,
            })
        
        logger.info(f"Generated {len(all_triplets)} triplets")
        
    except Exception as e:
        logger.error(f"Structured output failed in graph creator: {e}")
        # Very minimal fallback
        all_triplets = []
        graph_summary = "Error during triplet generation."
    
    return all_triplets, graph_summary
