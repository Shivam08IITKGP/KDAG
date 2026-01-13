"""Answering agent main module."""
import logging
from pathlib import Path
import networkx as nx

from answering_agent.classifier import classify, ClassificationOutput, get_graph_data
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
    
    # Load and extract graph data
    narrative_summary = "No narrative summary available."
    full_graph_text = "No graph data available."
    if graph_path:
        graph_file = Path(graph_path)
        if graph_file.exists():
            try:
                graph = nx.read_graphml(graph_path)
                # Load narrative summary and full triplet text
                narrative_summary, full_graph_text = get_graph_data(graph)
                logger.info(f"Loaded and extracted graph data from {graph_path}")
            except Exception as e:
                logger.warning(f"Error loading graph: {e}")
        else:
            logger.warning(f"Graph file not found: {graph_path}")

    # Run classifier
    logger.info("Running classifier")
    classification: ClassificationOutput = classify(
        book_name=book_name,
        character_name=character_name,
        backstory=backstory,
        graph_summary=full_graph_text, # Give entire graph to LLM
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
    
    # Run NLI Checker with Graph Summary
    logger.info("Running NLI Checker with Graph Summary")
    from answering_agent.nli_checker import check_nli
    
    # We pass the narrative summary as the premise for NLI
    nli_metrics = check_nli(backstory, narrative_summary)
    
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
