"""Graph creation utilities."""
import json
import logging
from pathlib import Path
from typing import TypedDict

import networkx as nx
from langchain_openai import ChatOpenAI

from graph_creator_agent.prompts import TRIPLET_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class Triplet(TypedDict):
    """Knowledge triplet structure."""
    subject: str
    relation: str
    object: str
    evidence_id: str


def load_existing_graph(book_name: str, character_name: str) -> nx.DiGraph:
    """Load existing graph or create new one.
    
    Args:
        book_name: Name of the book.
        character_name: Name of the character.
        
    Returns:
        NetworkX directed graph.
    """
    graph_dir = Path("graph_creator_agent/graph")
    graph_dir.mkdir(exist_ok=True)
    
    graph_filename = f"{book_name}_{character_name}.graphml"
    graph_path = graph_dir / graph_filename
    
    if graph_path.exists():
        logger.info(f"Loading existing graph from {graph_path}")
        try:
            graph = nx.read_graphml(graph_path)
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            logger.warning(f"Error loading graph: {e}. Creating new graph.")
    
    logger.info("Creating new graph")
    return nx.DiGraph()


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


def generate_triplets(evidence_list: list[dict], llm: ChatOpenAI) -> list[Triplet]:
    """Generate triplets from evidence using LLM.
    
    Args:
        evidence_list: List of evidence dicts with 'id' and 'text' keys.
        llm: ChatOpenAI instance.
        
    Returns:
        List of Triplet TypedDicts.
    """
    all_triplets = []
    
    for evidence in evidence_list:
        evidence_id = evidence["id"]
        evidence_text = evidence["text"]
        
        logger.info(f"Generating triplets for evidence {evidence_id}")
        
        # Format prompt
        prompt = TRIPLET_EXTRACTION_PROMPT.format(
            evidence_text=evidence_text,
            evidence_id=evidence_id,
        )
        
        # Get LLM response
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        logger.debug(f"LLM response for {evidence_id}: {response_text}")
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "```" in response_text:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            
            triplets = json.loads(response_text)
            if not isinstance(triplets, list):
                raise ValueError("Response is not a list")
            
            # Validate and convert to Triplet TypedDict
            for triplet in triplets:
                if all(key in triplet for key in ["subject", "relation", "object", "evidence_id"]):
                    all_triplets.append(Triplet(
                        subject=triplet["subject"],
                        relation=triplet["relation"],
                        object=triplet["object"],
                        evidence_id=triplet["evidence_id"],
                    ))
            
            logger.info(f"Generated {len(triplets)} triplets from evidence {evidence_id}")
            
        except Exception as e:
            logger.error(f"Error parsing triplets for evidence {evidence_id}: {e}")
            logger.error(f"Response text: {response_text}")
    
    return all_triplets


def add_triplets_to_graph(graph: nx.DiGraph, triplets: list[Triplet]) -> None:
    """Add triplets to graph as nodes and edges.
    
    Args:
        graph: NetworkX directed graph.
        triplets: List of Triplet TypedDicts.
    """
    for triplet in triplets:
        subject = triplet["subject"]
        relation = triplet["relation"]
        obj = triplet["object"]
        evidence_id = triplet["evidence_id"]
        
        # Add nodes if they don't exist
        if subject not in graph:
            graph.add_node(subject, type="entity")
        if obj not in graph:
            graph.add_node(obj, type="entity")
        
        # Add edge with relation and evidence_id
        if graph.has_edge(subject, obj):
            # Edge exists, update evidence_ids
            if "evidence_ids" not in graph[subject][obj]:
                graph[subject][obj]["evidence_ids"] = []
            if evidence_id not in graph[subject][obj]["evidence_ids"]:
                graph[subject][obj]["evidence_ids"].append(evidence_id)
        else:
            # New edge
            graph.add_edge(subject, obj, relation=relation, evidence_id=evidence_id, evidence_ids=[evidence_id])
    
    logger.info(f"Added {len(triplets)} triplets to graph. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


def save_graph(graph: nx.DiGraph, book_name: str, character_name: str) -> str:
    """Save graph to GraphML file.
    
    Args:
        graph: NetworkX directed graph.
        book_name: Name of the book.
        character_name: Name of the character.
        
    Returns:
        Path to saved graph file.
    """
    graph_dir = Path("graph_creator_agent/graph")
    graph_dir.mkdir(exist_ok=True)
    
    graph_filename = f"{book_name}_{character_name}.graphml"
    graph_path = graph_dir / graph_filename
    
    nx.write_graphml(graph, graph_path)
    logger.info(f"Saved graph to {graph_path}")
    
    return str(graph_path)
