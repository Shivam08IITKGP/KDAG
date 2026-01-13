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


def deduplicate_triplets(triplets: list[Triplet]) -> list[Triplet]:
    """Deduplicate triplets by merging evidence IDs for identical relations.
    
    Strategy:
    - Same (subject, relation, object): merge evidence_ids, keep first
    - Same (subject, object) but different relation: keep both, log warning
    
    Args:
        triplets: List of Triplet TypedDicts.
        
    Returns:
        Deduplicated list of triplets.
    """
    # Use dict with (subject, relation, object) as key
    triplet_map = {}
    relation_conflicts = []
    
    for triplet in triplets:
        subject = triplet["subject"]
        relation = triplet["relation"]
        obj = triplet["object"]
        evidence_id = triplet["evidence_id"]
        
        key = (subject, relation, obj)
        
        if key in triplet_map:
            # Duplicate found - merge evidence_ids
            existing = triplet_map[key]
            if evidence_id not in existing.get("evidence_ids", []):
                if "evidence_ids" not in existing:
                    existing["evidence_ids"] = [existing["evidence_id"]]
                existing["evidence_ids"].append(evidence_id)
        else:
            # Check for same (subject, object) with different relation
            conflict_key = (subject, obj)
            has_conflict = False
            for existing_key in triplet_map.keys():
                if existing_key[0] == subject and existing_key[2] == obj and existing_key[1] != relation:
                    has_conflict = True
                    relation_conflicts.append({
                        "subject": subject,
                        "object": obj,
                        "relation1": existing_key[1],
                        "relation2": relation,
                    })
                    break
            
            # Add new triplet (even if conflict - we keep both)
            triplet_map[key] = triplet.copy()
    
    # Log conflicts
    if relation_conflicts:
        logger.warning(f"Found {len(relation_conflicts)} relation conflicts (same subject-object, different relations):")
        for conflict in relation_conflicts[:5]:  # Show first 5
            logger.warning(f"  ({conflict['subject']}, {conflict['object']}): '{conflict['relation1']}' vs '{conflict['relation2']}'")
    
    # Convert back to list, consolidating evidence_ids
    deduplicated = []
    for triplet in triplet_map.values():
        if "evidence_ids" in triplet:
            # Multiple evidence IDs - keep as list in evidence_id field for consistency
            # (We'll handle this properly when adding to graph)
            deduplicated.append(triplet)
        else:
            deduplicated.append(triplet)
    
    original_count = len(triplets)
    dedup_count = len(deduplicated)
    if original_count > dedup_count:
        logger.info(f"Deduplicated {original_count} triplets to {dedup_count} (removed {original_count - dedup_count} duplicates)")
    
    return deduplicated


def add_triplets(graph: nx.DiGraph, triplets: list[Triplet]) -> None:
    """Add triplets to graph as nodes and edges.
    
    Triplets are deduplicated before adding to prevent duplicate edges.
    
    Args:
        graph: NetworkX directed graph.
        triplets: List of Triplet TypedDicts.
    """
    # Deduplicate triplets first
    triplets = deduplicate_triplets(triplets)
    
    for triplet in triplets:
        subject = triplet["subject"]
        relation = triplet["relation"]
        obj = triplet["object"]
        
        # Handle both single evidence_id and list of evidence_ids from deduplication
        if "evidence_ids" in triplet:
            evidence_ids = triplet["evidence_ids"]
        else:
            evidence_ids = [triplet["evidence_id"]]
        
        # Add nodes if they don't exist
        if subject not in graph:
            graph.add_node(subject, type="entity")
        if obj not in graph:
            graph.add_node(obj, type="entity")
        
        # Add edge with relation and evidence_ids
        if graph.has_edge(subject, obj):
            # Edge exists, update evidence_ids
            edge_data = graph[subject][obj]
            
            # Ensure evidence_ids is a list
            if "evidence_ids" not in edge_data:
                edge_data["evidence_ids"] = []
            elif not isinstance(edge_data["evidence_ids"], list):
                # Convert to list if it's not already
                edge_data["evidence_ids"] = [edge_data["evidence_ids"]] if edge_data["evidence_ids"] else []
            
            # Add new evidence_ids if not already present
            for eid in evidence_ids:
                if eid not in edge_data["evidence_ids"]:
                    edge_data["evidence_ids"].append(eid)
            
            # Update relation if different (keep existing or update)
            if edge_data.get("relation") != relation:
                logger.debug(f"Relation mismatch for edge ({subject}, {obj}): existing='{edge_data.get('relation')}', new='{relation}'")
        else:
            # New edge
            graph.add_edge(subject, obj, relation=relation, evidence_ids=evidence_ids)
    
    logger.info(f"Added {len(triplets)} triplets to graph. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")


def save_graph(graph: nx.DiGraph, book_name: str, character_name: str, graph_summary: str = "") -> str:
    """Save graph to GraphML file.
    
    GraphML doesn't support lists, so we convert all list attributes to JSON strings.
    
    Args:
        graph: NetworkX directed graph.
        book_name: Name of the book.
        character_name: Name of the character.
        graph_summary: Narrative summary of the character.
        
    Returns:
        Path to saved graph file.
    """
    graph_dir = Path("graph_creator_agent/graph")
    graph_dir.mkdir(exist_ok=True)
    
    safe_book = sanitize_filename(book_name)
    safe_char = sanitize_filename(character_name)
    graph_filename = f"{safe_book}_{safe_char}.graphml"
    graph_path = graph_dir / graph_filename
    
    # Store graph summary as graph attribute
    if graph_summary:
        graph.graph["graph_summary"] = graph_summary
    
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
