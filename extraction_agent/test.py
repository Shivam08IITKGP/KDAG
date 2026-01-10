import sys
import os
import logging
from dotenv import load_dotenv

# 1. Setup Logging
# This allows you to see the log messages defined in your main.py (logger.info, etc.)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 2. Load Environment Variables (for OpenRouter Key)
load_dotenv()

# Check if key exists to prevent immediate crash in shared_config
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY not found in environment variables.")

# 3. Import the extract function
# We import from the module path assuming test_agent.py is in the root
try:
    from extraction_agent.main import extract
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the root directory containing 'shared_config.py' and the 'extraction_agent' folder.")
    sys.exit(1)

def run_test():
    # 4. Define the input state
    # Using Thalcave from In Search of the Castaways
    state = {
        "book_name": "In Search of the Castaways",
        "character_name": "Thalcave",
        "backstory": (
            "Thalcave is a Patagonian Indian who serves as a guide to "
            "Lord Glenarvan and his party during their trek across South America "
            "to find Captain Grant. He is known for his silence, dignity, "
            "and expert knowledge of the terrain."
        )
    }

    print(f"\n{'='*50}")
    print(f"Testing Extraction Agent for: {state['character_name']}")
    print(f"{'='*50}\n")

    # 5. Execute the agent
    # This will use the real OpenRouter LLM via shared_config.py
    try:
        result_state = extract(state)
        
        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        
        # Display Queries
        print("\n--- Generated Queries ---")
        if result_state.get("queries"):
            for idx, query in enumerate(result_state["queries"], 1):
                print(f"{idx}. {query}")
        else:
            print("No queries generated.")

        # Display Evidence (Note: Will be empty as per your stub implementation)
        print("\n--- Retrieved Evidences ---")
        if result_state.get("evidences"):
            for ev in result_state["evidences"]:
                print(f"- {ev}")
        else:
            print("No evidence found (Expected behavior: get_evidence is a stub).")

    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()