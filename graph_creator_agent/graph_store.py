"""Module for handling graph storage and modification."""
import json
import logging
from pathlib import Path

import networkx as nx

from graph_creator_agent.types import Triplet

logger = logging.getLogger(__name__)


def _convert_graphml_attributes(graph: nx.DiGraph) -> None:
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


def sanitize_filename(name: str) -> str:
    """Sanitize filename by replacing invalid characters.
    
    Args:
        name: Original filename/component.
        
    Returns:
        Sanitized string safe for filenames.
    """
    # Replace slashes and other potentially problematic chars
    return name.replace("/", "_").replace("\\", "_").replace(":", "-")


def load_graph(book_name: str, character_name: str) -> nx.DiGraph:
    """Load existing graph or create new one.
    
    Args:
        book_name: Name of the book.
        character_name: Name of the character.
        
    Returns:
        NetworkX directed graph.
    """
    graph_dir = Path("graph_creator_agent/graph")
    graph_dir.mkdir(exist_ok=True)
    
    safe_book = sanitize_filename(book_name)
    safe_char = sanitize_filename(character_name)
    graph_filename = f"{safe_book}_{safe_char}.graphml"
    graph_path = graph_dir / graph_filename
    
    if graph_path.exists():
        logger.info(f"Loading existing graph from {graph_path}")
        try:
            graph = nx.read_graphml(graph_path)
            # Convert string attributes back to proper types
            _convert_graphml_attributes(graph)
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            logger.warning(f"Error loading graph: {e}. Creating new graph.")
    
    logger.info("Creating new graph")
    return nx.DiGraph()


def add_triplets(graph: nx.DiGraph, triplets: list[Triplet]) -> None:
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
    
    safe_book = sanitize_filename(book_name)
    safe_char = sanitize_filename(character_name)
    graph_filename = f"{safe_book}_{safe_char}.graphml"
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
    return str(graph_path)
