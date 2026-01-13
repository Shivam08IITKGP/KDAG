import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import the main entry point
from answering_agent.main import answer

def run_test():
    # --- 1. SETUP PATHS ---
    # The graph is in a sibling folder to 'answering_agent'
    # We need to go up one level from answering_agent, then into graph_creator_agent/graph
    current_dir = os.path.dirname(os.path.abspath(__file__))  # answering_agent directory
    parent_dir = os.path.dirname(current_dir)  # project root
    graph_dir = os.path.join(parent_dir, "graph_creator_agent", "graph")
    graph_file = os.path.join(graph_dir, "In Search of the Castaways_Jacques Paganel.graphml")
    
    print(f"Looking for graph at: {graph_file}")
    if not os.path.exists(graph_file):
        print(f"⚠️  WARNING: Graph file not found at path. Please check the folder structure.")
        print(f"Current directory: {current_dir}")
        print(f"Parent directory: {parent_dir}")
        print(f"Graph directory: {graph_dir}")
    else:
        print(f"✅ Graph file found!")

    # --- 2. DEFINE INPUT STATE ---
    input_state = {
        "book_name": "In Search of the Castaways",
        "character_name": "Jacques Paganel",
        
        # A specific claim about Paganel to test consistency
        "backstory": (
            "Jacques Paganel is a geographer from the Geographic Society who "
            "accidentally boarded the Duncan yacht. He later accompanied the "
            "expedition across Australia."
        ),
        
        "graph_path": graph_file,  # Use the full path to the .graphml file
        
        # MOCK EVIDENCE LIST
        # I have mapped these IDs directly to the 'evidence_id' fields in your GraphML file.
        "evidences": [
            {
                "id": "ev_8781",  # Matches XML edge: Geographic Society -> Jacques Paganel
                "text": "Jacques Paganel is a distinguished secretary of the Geographic Society in Paris."
            },
            {
                "id": "ev_2337",  # Matches XML edge: Australia -> Jacques Paganel
                "text": "Paganel traveled with the group across the treacherous landscapes of Australia."
            },
            {
                "id": "ev_5133",  # Matches XML edge: Australia -> Duncan yacht
                "text": "The Duncan yacht waited for the travelers off the coast of Australia."
            },
            {
                "id": "ev_9999",  # Fake ID (not in graph) to test filtering
                "text": "Paganel once traveled to the moon in a cannon."
            }
        ]
    }

    print("\n--- Starting Test for Jacques Paganel ---")
    
    # --- 3. RUN THE AGENT ---
    try:
        result_state = answer(input_state)
        
        # --- 4. PRINT RESULTS ---
        print("\n" + "="*40)
        print("✅ TEST COMPLETE")
        print("="*40)
        print(f"Label:     {result_state['label']} (1=Consistent, 0=Contradicting)")
        print(f"Reasoning: {result_state['reasoning']}")
        print(f"Evidence:  {result_state['evidence_ids']}")
        print("="*40)
        
    except Exception as e:
        print(f"\n❌ Error running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()