"""Classification module for answering agent."""
import json
import logging
from pathlib import Path
from typing import TypedDict

import networkx as nx
from langchain_openai import ChatOpenAI

from answering_agent.prompts import CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class ClassificationOutput(TypedDict):
    """Output from classification."""
    label: int  # 1 or 0
    reasoning: str
    evidence_queries: list[str]  # Queries to fetch supporting evidence


def summarize_graph(graph: nx.DiGraph) -> str:
    """Create a text summary of the graph.
    
    Args:
        graph: NetworkX directed graph.
        
    Returns:
        Text summary of the graph.
    """
    if graph.number_of_nodes() == 0:
        return "No graph data available."
    
    summary_parts = []
    summary_parts.append(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    # Extract some key relationships
    edges_list = []
    for u, v, data in graph.edges(data=True):
        relation = data.get("relation", "related_to")
        edges_list.append(f"{u} --[{relation}]--> {v}")
    
    if edges_list:
        summary_parts.append("\nKey relationships:")
        summary_parts.extend(edges_list[:10])  # Limit to first 10
    
    return "\n".join(summary_parts)


def classify(
    book_name: str,
    character_name: str,
    backstory: str,
    graph_summary: str,
    character_summary: str,
    llm: ChatOpenAI,
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
        graph_summary=graph_summary,
        character_summary=character_summary,
    )
    
    logger.debug(f"Classification prompt: {prompt[:200]}...")
    
    # Get LLM response
    logger.info("Calling LLM for classification")
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    logger.debug(f"LLM response: {response_text}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response
        if "```" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        
        # Validate and create TypedDict
        label = result.get("label", 0)
        if label not in [0, 1]:
            logger.warning(f"Invalid label {label}, defaulting to 0")
            label = 0
        
        reasoning = result.get("reasoning", "No reasoning provided")
        evidence_queries = result.get("evidence_queries", [])
        
        # Validate queries
        if not isinstance(evidence_queries, list):
            logger.warning(f"evidence_queries is not a list, defaulting to empty")
            evidence_queries = []
        
        output: ClassificationOutput = {
            "label": label,
            "reasoning": reasoning,
            "evidence_queries": evidence_queries,
        }
        
        logger.info(f"Classification complete: label={label}, queries={len(evidence_queries)}")
        logger.debug(f"Evidence queries: {evidence_queries}")
        return output
        
    except Exception as e:
        logger.error(f"Error parsing classification response: {e}")
        logger.error(f"Response text: {response_text}")
        
        # Return default
        return ClassificationOutput(
            label=0,
            reasoning=f"Error during classification: {e}",
            evidence_queries=[],
        )
