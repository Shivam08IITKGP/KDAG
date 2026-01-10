"""Graph creation utilities."""
import ast
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TypedDict

import networkx as nx
from langchain_openai import ChatOpenAI

from graph_creator_agent.prompts import TRIPLET_EXTRACTION_BATCH_PROMPT

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
    """Generate triplets from all evidence at once using LLM with structured output.
    
    Processes all evidence items in a single batch call instead of one-by-one.
    
    Args:
        evidence_list: List of evidence dicts with 'id' and 'text' keys.
        llm: ChatOpenAI instance configured with structured output.
        
    Returns:
        List of Triplet TypedDicts.
    """
    if not evidence_list:
        logger.info("No evidence to process")
        return []
    
    logger.info(f"Generating triplets for all {len(evidence_list)} evidence items at once")
    
    # Format evidence items for batch processing
    evidence_items_text = ""
    for idx, evidence in enumerate(evidence_list, 1):
        evidence_items_text += f"\n--- Evidence {idx} (ID: {evidence['id']}) ---\n"
        evidence_items_text += f"{evidence['text']}\n"
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(TripletList)
    
    # Format batch prompt
    prompt = TRIPLET_EXTRACTION_BATCH_PROMPT.format(
        evidence_items=evidence_items_text,
    )
    
    try:
        logger.info("Making single LLM call to extract triplets from all evidence")
        # Get structured LLM response for all evidence at once
        response: TripletList = structured_llm.invoke(prompt)
        
        # Extract triplets from response
        triplets = response.get("triplets", [])
        
        # Validate and ensure evidence_id is set correctly
        all_triplets = []
        evidence_ids = {ev["id"] for ev in evidence_list}
        
        for triplet in triplets:
            # Validate evidence_id exists in our evidence list
            evidence_id = triplet.get("evidence_id", "")
            if evidence_id not in evidence_ids:
                logger.warning(f"Triplet has invalid evidence_id '{evidence_id}', skipping: {triplet}")
                continue
            
            all_triplets.append(Triplet(
                subject=triplet["subject"],
                relation=triplet["relation"],
                object=triplet["object"],
                evidence_id=evidence_id,
            ))
        
        logger.info(f"Generated {len(all_triplets)} triplets from {len(evidence_list)} evidence items in a single batch")
        
        # Log triplet count per evidence_id
        evidence_id_counts = Counter(t["evidence_id"] for t in all_triplets)
        logger.info(f"Triplet distribution: {dict(evidence_id_counts)}")
        
        return all_triplets
        
    except Exception as e:
        logger.error(f"Error generating triplets in batch: {e}")
        # Fallback: try without structured output
        try:
            logger.warning("Falling back to JSON parsing")
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            if "```" in response_text:
                # Find JSON block
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
            
            # Parse JSON
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "triplets" in parsed:
                triplets = parsed["triplets"]
            elif isinstance(parsed, list):
                triplets = parsed
            else:
                raise ValueError("Response is not in expected format")
            
            if not isinstance(triplets, list):
                raise ValueError("Triplets is not a list")
            
            all_triplets = []
            evidence_ids = {ev["id"] for ev in evidence_list}
            
            for triplet in triplets:
                if not all(key in triplet for key in ["subject", "relation", "object", "evidence_id"]):
                    logger.warning(f"Triplet missing required fields, skipping: {triplet}")
                    continue
                
                evidence_id = triplet["evidence_id"]
                if evidence_id not in evidence_ids:
                    logger.warning(f"Triplet has invalid evidence_id '{evidence_id}', skipping: {triplet}")
                    continue
                
                all_triplets.append(Triplet(
                    subject=triplet["subject"],
                    relation=triplet["relation"],
                    object=triplet["object"],
                    evidence_id=evidence_id,
                ))
            
            logger.info(f"Fallback: Generated {len(all_triplets)} triplets from {len(evidence_list)} evidence items")
            return all_triplets
            
        except Exception as fallback_error:
            logger.error(f"Fallback parsing also failed: {fallback_error}")
            return []


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
