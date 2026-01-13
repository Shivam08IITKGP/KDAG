import logging
from typing import TypedDict
from pydantic import BaseModel, Field

import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel

from answering_agent.prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Structured output format for backstory classification."""
    label: int = Field(description="1 if consistent, 0 if contradictory")
    reasoning: str = Field(description="Strictly 1-2 line concise analysis.")
    evidence_queries: list[str] = Field(description="Queries to fetch supporting or contradictory evidence.")


class ClassificationOutput(TypedDict):
    """Output from classification for the pipeline state."""
    label: int  # 1 or 0
    reasoning: str
    evidence_queries: list[str]


def get_graph_data(graph: nx.DiGraph) -> tuple[str, str]:
    """Extract narrative summary and full triplet text from the graph.
    
    Args:
        graph: NetworkX directed graph.
        
    Returns:
        Tuple of (narrative_summary, full_triplets_text).
    """
    if graph.number_of_nodes() == 0:
        return "No narrative summary available.", "No graph data available."
    
    # Get the narrative summary stored as a graph attribute
    narrative_summary = graph.graph.get("graph_summary", "No narrative summary available.")
    
    # Format all edges into triplets
    edges_list = []
    for u, v, data in graph.edges(data=True):
        relation = data.get("relation", "related_to")
        edges_list.append(f"{u} --[{relation}]--> {v}")
    
    full_triplets_text = "\n".join(edges_list) if edges_list else "No triplets found."
    
    return narrative_summary, full_triplets_text


def classify(
    book_name: str,
    character_name: str,
    backstory: str,
    graph_summary: str,
    character_summary: str,
    llm: BaseChatModel,
) -> ClassificationOutput:
    """Classify backstory as consistent or contradicting.
    
    Args:
        book_name: Name of the book.
        character_name: Name of the character.
        backstory: The backstory to check.
        graph_summary: Text summary of the knowledge graph.
        character_summary: Canonical summary of the character.
        llm: ChatOpenAI instance.
        
    Returns:
        ClassificationOutput TypedDict.
    """
    logger.info("Starting classification")
    
    # Format prompt
    prompt = CLASSIFICATION_PROMPT.format(
        book_name=book_name,
        character_name=character_name,
        backstory=backstory,
        full_graph=graph_summary,
        character_summary=character_summary,
    )
    
    logger.debug(f"Classification prompt: {prompt[:200]}...")
    
    # Get LLM response with structured output
    logger.info("Calling structured LLM for classification")
    try:
        structured_llm = llm.with_structured_output(ClassificationResult)
        result = structured_llm.invoke(prompt)
        
        # Strictly enforce 1-2 lines for reasoning (defense in depth)
        reasoning = result.reasoning
        reasoning_parts = reasoning.split('\n')
        if len(reasoning_parts) > 2:
            reasoning = " ".join(reasoning_parts[:2]).strip()
            
        output: ClassificationOutput = {
            "label": result.label if result.label in [0, 1] else 0,
            "reasoning": reasoning,
            "evidence_queries": result.evidence_queries,
        }
        
        logger.info(f"Classification complete: label={output['label']}")
        return output
        
    except Exception as e:
        logger.error(f"Error during structured classification: {e}")
        
        # Return default failure state
        return {
            "label": 0,
            "reasoning": f"Error during classification: {e}",
            "evidence_queries": [],
        }
