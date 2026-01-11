"""Answering agent main module."""
import logging
from answering_agent.classifier import classify, ClassificationOutput
from answering_agent.evidence_generator import retrieve_evidence_for_queries, EvidenceOutput
from extraction_agent.character_summaries import get_character_summary
from shared_config import create_llm
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def answer(state: dict) -> dict:
    """Answer whether backstory is consistent or contradicting.
    
    Args:
        state: PipelineState dictionary.
        
    Returns:
        Updated state with label, reasoning, evidence_queries, and evidence_chunks.
    """
    logger.info("Starting answering agent")
    logger.info(f"Book: {state['book_name']}, Character: {state['character_name']}")
    
    book_name = state["book_name"]
    character_name = state["character_name"]
    backstory = state["backstory"]
    graph_path = state.get("graph_path")
    
    # Get character summary
    character_summary = get_character_summary(book_name, character_name)
    if not character_summary:
        character_summary = "No canonical character information available."
    
    # Initialize LLM with OpenRouter config
    llm = create_llm()
    
    # Run classifier
    logger.info("Running classifier")
    classification: ClassificationOutput = classify(
        book_name=book_name,
        character_name=character_name,
        backstory=backstory,
        graph_path=graph_path,
        character_summary=character_summary,
        llm=llm,
    )
    
    label = classification["label"]
    reasoning = classification["reasoning"]
    evidence_queries = classification["evidence_queries"]
    
    logger.info(f"Classification result: label={label}")
    logger.info(f"Generated {len(evidence_queries)} evidence queries")
    
    # Retrieve evidence for queries
    logger.info("Retrieving evidence for queries")
    evidence_output: EvidenceOutput = retrieve_evidence_for_queries(
        evidence_queries=evidence_queries,
        book_name=book_name,
        k=3,  # Retrieve 3 chunks per query
    )
    
    evidence_chunks = evidence_output["evidence_chunks"]
    logger.info(f"Retrieved {len(evidence_chunks)} total evidence chunks")
    
    # Run NLI Checker
    logger.info("Running NLI Checker")
    from answering_agent.nli_checker import check_nli
    
    # Collect all unique evidence texts
    all_evidence_texts = set()
    
    # From extraction phase
    if state.get("evidences"):
        for ev in state["evidences"]:
            if "text" in ev:
                all_evidence_texts.add(ev["text"])
                
    # From answering phase
    if evidence_chunks:
        for ev in evidence_chunks:
            if "text" in ev:
                all_evidence_texts.add(ev["text"])
                
    nli_metrics = check_nli(backstory, list(all_evidence_texts))
    
    # Update state
    updated_state = state.copy()
    updated_state["label"] = label
    updated_state["reasoning"] = reasoning
    updated_state["evidence_queries"] = evidence_queries
    updated_state["evidence_chunks"] = evidence_chunks
    updated_state["nli_avg_entailment"] = nli_metrics["entailment_avg"]
    updated_state["nli_max_contradiction"] = nli_metrics["contradiction_max"]
    updated_state["nli_avg_contradiction"] = nli_metrics["contradiction_avg"]
    
    logger.info("Answering agent completed")
    return updated_state
