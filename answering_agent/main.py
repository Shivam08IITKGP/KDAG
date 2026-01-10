"""Answering agent main module."""
import logging

from langchain_openai import ChatOpenAI

from answering_agent.classifier import classify, ClassificationOutput
from answering_agent.evidence_generator import generate_evidence_ids, EvidenceOutput

logger = logging.getLogger(__name__)


def answer(state: dict) -> dict:
    """Answer whether backstory is consistent or contradicting.
    
    Args:
        state: PipelineState dictionary.
        
    Returns:
        Updated state with label, reasoning, and evidence_ids.
    """
    logger.info("Starting answering agent")
    logger.info(f"Book: {state['book_name']}, Character: {state['character_name']}")
    
    book_name = state["book_name"]
    character_name = state["character_name"]
    backstory = state["backstory"]
    graph_path = state.get("graph_path")
    evidences = state.get("evidences", [])
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Run classifier
    logger.info("Running classifier")
    classification: ClassificationOutput = classify(
        book_name=book_name,
        character_name=character_name,
        backstory=backstory,
        graph_path=graph_path,
        llm=llm,
    )
    
    label = classification["label"]
    reasoning = classification["reasoning"]
    
    logger.info(f"Classification result: label={label}")
    
    # Run evidence generator
    logger.info("Running evidence generator")
    evidence_output: EvidenceOutput = generate_evidence_ids(
        reasoning=reasoning,
        evidences=evidences,
        backstory=backstory,
        llm=llm,
    )
    
    evidence_ids = evidence_output["evidence_ids"]
    logger.info(f"Generated {len(evidence_ids)} evidence IDs")
    
    # Update state
    updated_state = state.copy()
    updated_state["label"] = label
    updated_state["reasoning"] = reasoning
    updated_state["evidence_ids"] = evidence_ids
    
    logger.info("Answering agent completed")
    return updated_state
