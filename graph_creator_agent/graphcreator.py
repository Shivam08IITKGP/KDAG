"""Graph creation utilities."""
import ast
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


class TripletList(TypedDict):
    """Wrapper for list of triplets for structured output."""
    triplets: list[Triplet]


def convert_graphml_attributes(graph: nx.DiGraph) -> None:
    """Convert GraphML string attributes back to proper types.
    
    GraphML stores all attributes as strings, so we need to convert
    JSON-encoded lists back to their proper format.
    
    Args:
        graph: NetworkX graph loaded from GraphML.
    """
    # Convert edge attributes from JSON strings back to lists
    for u, v, data in graph.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, str):
                # Try to parse as JSON (for lists we saved as JSON strings)
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        data[key] = parsed
                except (json.JSONDecodeError, ValueError):
                    # Not a JSON string, keep as is
                    pass
    
    # Also check node attributes
    for node, data in graph.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        data[key] = parsed
                except (json.JSONDecodeError, ValueError):
                    pass


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
            # Convert string attributes back to proper types
            convert_graphml_attributes(graph)
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
    """Generate triplets from evidence using LLM with structured output.
    
    Args:
        evidence_list: List of evidence dicts with 'id' and 'text' keys.
        llm: ChatOpenAI instance configured with structured output.
        
    Returns:
        List of Triplet TypedDicts.
    """
    all_triplets = []
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(TripletList)
    
    for evidence in evidence_list:
        evidence_id = evidence["id"]
        evidence_text = evidence["text"]
        
        logger.info(f"Generating triplets for evidence {evidence_id}")
        
        # Format prompt
        prompt = TRIPLET_EXTRACTION_PROMPT.format(
            evidence_text=evidence_text,
            evidence_id=evidence_id,
        )
        
        try:
            # Get structured LLM response
            response: TripletList = structured_llm.invoke(prompt)
            
            # Extract triplets from response
            triplets = response.get("triplets", [])
            
            # Validate and ensure evidence_id is set correctly
            for triplet in triplets:
                # Ensure evidence_id matches
                triplet["evidence_id"] = evidence_id
                all_triplets.append(Triplet(
                    subject=triplet["subject"],
                    relation=triplet["relation"],
                    object=triplet["object"],
                    evidence_id=triplet["evidence_id"],
                ))
            
            logger.info(f"Generated {len(triplets)} triplets from evidence {evidence_id}")
            
        except Exception as e:
            logger.error(f"Error generating triplets for evidence {evidence_id}: {e}")
            # Fallback: try without structured output
            try:
                logger.warning("Falling back to JSON parsing")
                response = llm.invoke(prompt)
                response_text = response.content.strip()
                
                # Try to extract JSON from response
                if "```" in response_text:
                    start = response_text.find("[")
                    end = response_text.rfind("]") + 1
                    if start != -1 and end > start:
                        response_text = response_text[start:end]
                
                triplets = json.loads(response_text)
                if not isinstance(triplets, list):
                    raise ValueError("Response is not a list")
                
                for triplet in triplets:
                    if all(key in triplet for key in ["subject", "relation", "object"]):
                        all_triplets.append(Triplet(
                            subject=triplet["subject"],
                            relation=triplet["relation"],
                            object=triplet["object"],
                            evidence_id=evidence_id,
                        ))
                logger.info(f"Fallback: Generated {len(triplets)} triplets from evidence {evidence_id}")
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {fallback_error}")
    
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
        
        # Add edge with relation and evidence_ids (only use evidence_ids, not evidence_id)
        if graph.has_edge(subject, obj):
            # Edge exists, update evidence_ids
            edge_data = graph[subject][obj]
            
            # Ensure evidence_ids is a list
            if "evidence_ids" not in edge_data:
                edge_data["evidence_ids"] = []
            elif not isinstance(edge_data["evidence_ids"], list):
                # Convert to list if it's not already
                edge_data["evidence_ids"] = [edge_data["evidence_ids"]] if edge_data["evidence_ids"] else []
            
            # Add evidence_id if not already present
            if evidence_id not in edge_data["evidence_ids"]:
                edge_data["evidence_ids"].append(evidence_id)
            
            # Update relation if different (keep existing or update)
            # Note: We keep the existing relation, but could log if different
            if edge_data.get("relation") != relation:
                logger.debug(f"Relation mismatch for edge ({subject}, {obj}): existing='{edge_data.get('relation')}', new='{relation}'")
        else:
            # New edge - only use evidence_ids (list)
            graph.add_edge(subject, obj, relation=relation, evidence_ids=[evidence_id])
    
    logger.info(f"Added {len(triplets)} triplets to graph. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


def save_graph(graph: nx.DiGraph, book_name: str, character_name: str) -> str:
    """Save graph to GraphML file.
    
    GraphML doesn't support lists, so we convert all list attributes to JSON strings.
    
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
    
    # Convert all list attributes to JSON strings for GraphML compatibility
    # GraphML only supports scalar types (str, int, float, bool)
    for u, v, data in graph.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = json.dumps(value)
    
    # Also check node attributes (though we don't use lists there currently)
    for node, data in graph.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, list):
                data[key] = json.dumps(value)
    
    nx.write_graphml(graph, graph_path)
    logger.info(f"Saved graph to {graph_path}")
    
    return str(graph_path)
