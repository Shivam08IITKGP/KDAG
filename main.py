import logging
import datetime
from pathlib import Path

# Import the core agent entry points
from extraction_agent.main import extract
from graph_creator_agent.main import create_graph
from answering_agent.main import answer
from ML_answering_final.features import extract_features
from Graphrag.pathway.build_index import build_index
import pandas as pd

def setup_logging():
    """Sets up unified logging for the HIVE system."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Suppress verbose logs from dependencies
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    return str(log_file)
    
def ensure_book_indexed(book_name: str):
    """Checks if a book needs to be indexed before retrieval."""
    # Mapping of book names to their source text files
    book_map = {
        "In Search of the Castaways": "Books/In search of the castaways.txt",
        "The Count of Monte Cristo": "Books/The Count of Monte Cristo.txt"
    }
    
    source_path = book_map.get(book_name)
    if source_path and Path(source_path).exists():
        # build_index inside build_index.py now handles the existence check
        build_index(source_path, book_name)
    else:
        logging.getLogger(__name__).warning(f"Could not find source text for book: {book_name}")

def run_pipeline_for_row(row_data: dict) -> dict:
    """
    Executes the sequential agents for a single backstory.
    
    Args:
        row_data: Dictionary containing 'book_name', 'char', and 'content'.
        
    Returns:
        The final pipeline state dictionary containing labels and evidence traces.
    """
    logger = logging.getLogger(__name__)
    
    # Step 0: Ensure the book is indexed in the vector store
    ensure_book_indexed(row_data["book_name"])
    
    # Initialize the Pipeline State
    state = {
        "book_name": row_data["book_name"],
        "character_name": row_data["char"],
        "backstory": row_data["content"],
        "queries": [],
        "evidences": [],
        "graph_path": None,
        "label": None,
        "reasoning": None
    }
    
    logger.info(f"--- HIVE Pipeline Start: {state['character_name']} ({state['book_name']}) ---")
    
    try:
        # Step 1: Query Generation & Evidence Extraction
        logger.info("Executing Step 1: Extraction Agent...")
        state = extract(state)
        
        # Step 2: Knowledge Graph Synthesis
        logger.info("Executing Step 2: Graph Creator Agent...")
        state = create_graph(state)
        
        # Step 3: Consistency Assessment & NLI Verification
        logger.info("Executing Step 3: Answering Agent...")
        state = answer(state)
        
        logger.info("--- HIVE Pipeline Completed Successfully ---")
        return state
        
    except Exception as e:
        logger.error(f"Pipeline failed during row execution: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Setup
    setup_logging()
    from utils.input import get_input_data
    import argparse
    
    parser = argparse.ArgumentParser(description="HIVE: Pipeline Controller")
    parser.add_argument("--index", type=int, help="Process a single row index from utils/train.csv")
    parser.add_argument("--train-data", action="store_true", help="Process ALL rows in utils/train.csv and save features")
    parser.add_argument("--start", type=int, help="Start index for range processing")
    parser.add_argument("--end", type=int, help="End index for range processing")
    args = parser.parse_args()
    
    # Determine the range of indices to process
    start_idx, end_idx = None, None
    
    if args.train_data:
        df = pd.read_csv("utils/train.csv")
        start_idx, end_idx = 0, len(df) - 1
    elif args.index is not None:
        start_idx, end_idx = args.index, args.index
    elif args.start is not None and args.end is not None:
        start_idx, end_idx = args.start, args.end
    elif any(v is not None for v in vars(args).values() if v is not False):
        # Some other argument was passed but not enough for a range
        pass
    else:
        # No arguments passed - Prompt the user
        try:
            print("\n--- HIVE Interactive Range Selector ---")
            start_input = input("Enter start row index (0-based): ")
            end_input = input("Enter end row index (0-based): ")
            start_idx = int(start_input)
            end_idx = int(end_input)
        except ValueError:
            print("Invalid input. Please enter numeric indices.")
            exit(1)

    if start_idx is not None and end_idx is not None:
        # Load data
        df = pd.read_csv("utils/train.csv")
        output_file = Path("output/features_output.csv")
        output_file.parent.mkdir(exist_ok=True)
        
        print(f"\nHIVE: Processing rows {start_idx} to {end_idx}...")
        
        for i in range(start_idx, end_idx + 1):
            if i >= len(df):
                print(f"Skipping index {i} (out of range)")
                break
                
            row = df.iloc[i]
            print(f"\n[{i+1}/{len(df)}] Processing: {row['char']} ({row['book_name']})")
            
            try:
                row_data = {
                    "book_name": row["book_name"],
                    "char": row["char"],
                    "content": row["content"]
                }
                state = run_pipeline_for_row(row_data)
                
                # Extract and save features (always helpful for training data)
                features = extract_features(state)
                features["row_index"] = i
                features["id"] = row.get("id", i)
                
                # Save incrementally to features_output.csv
                pd.DataFrame([features]).to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
                
                # Print result for terminal visibility
                verdict_str = "CONSISTENT" if state.get('label') == 1 else "CONTRADICTING"
                print(f"\n--- Result for {row['char']} ---")
                print(f"Verdict:         {verdict_str} ({state.get('label')})")
                print(f"NLI Entailment:  {state.get('nli_avg_entailment', 0):.4f} (Avg)")
                print(f"NLI Contradict:  {state.get('nli_max_contradiction', 0):.4f} (Max), {state.get('nli_avg_contradiction', 0):.4f} (Avg)")
                print(f"Reasoning: {state.get('reasoning')}")
                print("-" * 60)
                    
            except Exception as e:
                print(f"Error on row {i}: {e}")
                continue
                
        print(f"\nProcessing complete. Cumulative features saved to {output_file}")
    else:
        parser.print_help()
