import networkx as nx
import random
import string

def generate_random_backstory_graph(
    book_name="In Search of the Castaways",
    character_name="Jacques Paganel",
    num_triplets=15
):
    """
    Generate a random knowledge graph for testing the backstory consistency system.
    
    Creates a graph with:
    - Nodes representing entities (subjects and objects from triplets)
    - Edges representing relations
    - Edge attributes for evidence_id
    
    Args:
        book_name: Name of the book
        character_name: Name of the character
        num_triplets: Number of random triplets to generate
    
    Returns:
        NetworkX graph ready to save as GraphML
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Sample entities related to the character
    entities = [
        character_name,
        "Lord Glenarvan",
        "Captain Grant",
        "Mary Grant",
        "Robert Grant",
        "Lady Helena",
        "Patagonia",
        "Australia",
        "New Zealand",
        "Geographic Society",
        "Scotland",
        "France",
        "HMS Britannia",
        "Duncan yacht"
    ]
    
    # Sample relations
    relations = [
        "is_secretary_of",
        "traveled_to",
        "rescued",
        "accompanied",
        "member_of",
        "met_in",
        "sailed_on",
        "searched_for",
        "befriended",
        "wrote_about",
        "studied",
        "navigated_through"
    ]
    
    # Generate random triplets
    for i in range(num_triplets):
        # Randomly select subject, relation, object
        subject = random.choice(entities)
        relation = random.choice(relations)
        object_entity = random.choice([e for e in entities if e != subject])
        
        # Generate random evidence ID
        evidence_id = f"ev_{random.randint(1000, 9999)}"
        
        # Add edge with attributes
        G.add_edge(
            subject,
            object_entity,
            relation=relation,
            evidence_id=evidence_id,
            label=relation  # For visualization
        )
    
    # Add some node attributes
    for node in G.nodes():
        G.nodes[node]['entity_type'] = 'character' if node in [character_name, "Lord Glenarvan", "Captain Grant", "Mary Grant", "Robert Grant", "Lady Helena"] else 'location' if node in ["Patagonia", "Australia", "New Zealand", "Scotland", "France"] else 'organization'
    
    return G

# Generate and save the graph
if __name__ == "__main__":
    book_name = "In Search of the Castaways"
    character_name = "Jacques Paganel"
    
    # Generate graph
    graph = generate_random_backstory_graph(book_name, character_name, num_triplets=20)
    
    # Create output directory
    import os
    os.makedirs("graph_creator_agent/graph", exist_ok=True)
    
    # Save as GraphML
    output_path = f"graph_creator_agent/graph/{book_name}_{character_name}.graphml"
    nx.write_graphml(graph, output_path)
    
    print(f"Generated test graph: {output_path}")
    print(f"\nGraph Statistics:")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"\nSample Triplets:")
    for i, (u, v, data) in enumerate(list(graph.edges(data=True))[:5]):
        print(f"{i+1}. ({u}) --[{data['relation']}]--> ({v}) [Evidence: {data['evidence_id']}]")