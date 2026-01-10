"""Main entry point for the Evidence-Grounded Backstory Consistency System."""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from extraction_agent.main import extract
from graph_creator_agent.main import create_graph
from answering_agent.main import answer
from Graphrag.pathway.build_index import build_index
from input import get_input_data

# Load environment variables
load_dotenv()

class PipelineState(TypedDict):
    """Global state for the LangGraph pipeline."""
    book_name: str
    character_name: str
    backstory: str
    
    queries: list[str]
    evidences: list[dict]  # {id, text}
    
    graph_path: str | None
    
    label: int | None
    reasoning: str | None
    evidence_ids: list[str] | None


def setup_logging() -> str:
    """Set up logging with file and console handlers.
    
    Returns:
        Path to the log file created.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{timestamp}.log"
    
    # Create formatter with machine-readable format
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return str(log_file)




def print_output(state: PipelineState):
    """Print formatted output."""
    label_text = "CONSISTENT" if state["label"] == 1 else "CONTRADICTING"
    
    print(f"\nLabel: {label_text} ({state['label']})")
    print(f"\nReasoning:\n{state['reasoning']}\n")
    
    print("Supporting Evidence IDs:")
    if state["evidence_ids"]:
        # Get evidence texts from state
        evidence_dict = {ev["id"]: ev["text"] for ev in state["evidences"]}
        for ev_id in state["evidence_ids"]:
            ev_text = evidence_dict.get(ev_id, "N/A")
            print(f"- {ev_id} - {ev_text}")
    else:
        print("- No evidence IDs provided")


def run_pipeline_for_row(row_data: dict) -> PipelineState:
    """Run the pipeline for a single row of data.
    
    Args:
        row_data: Dictionary with book_name, char, and content.
        
    Returns:
        Final pipeline state.
    """
    logger = logging.getLogger(__name__)
    
    # Initialize state
    initial_state: PipelineState = {
        "book_name": row_data["book_name"],
        "character_name": row_data["char"],
        "backstory": row_data["content"],
        "queries": [],
        "evidences": [],
        "graph_path": None,
        "label": None,
        "reasoning": None,
        "evidence_ids": None,
    }
    
    logger.info(f"Processing: {row_data['char']} from {row_data['book_name']}")
    
    # Create LangGraph workflow
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("extract", extract)
    workflow.add_node("create_graph", create_graph)
    workflow.add_node("answer", answer)
    
    # Add edges
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "create_graph")
    workflow.add_edge("create_graph", "answer")
    workflow.add_edge("answer", END)
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("Running pipeline")
    
    # Run pipeline
    final_state = app.invoke(initial_state)
    logger.info("Pipeline completed successfully")
    
    return final_state


def main():
    """Main entry point."""
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Evidence-Grounded Backstory Consistency System")

    # Get number of rows to process from user input
    try:
        num_rows = int(input("Enter number of rows to process from train.csv (default: 2): ") or "2")
    except ValueError:
        num_rows = 2
        logger.info("Invalid input, using default: 2 rows")

    # Read data using input.py function
    try:
        df = get_input_data(num_rows)
        logger.info(f"Loaded {len(df)} rows from train.csv")
        logger.info(f"Processing {len(df)} backstories")
    except Exception as e:
        logger.error(f"Error reading train.csv: {e}")
        raise
    
    # Process each row
    results = []
    output_file = "output.csv"
    # Initialize output file with headers
    pd.DataFrame(columns=["book_name", "character_name", "predictions", "reasoning"]).to_csv(output_file, index=False)
    logger.info(f"Initialized {output_file}")

    flag = defaultdict(int)
    for idx, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing Row {idx + 1}/{len(df)}")
        print(f"Character: {row['char']}")
        print(f"Book: {row['book_name']}")
        print(f"{'='*80}\n")
        # topchunks = retrieve_topk("")

        book_path = ""
        # Handle book name matching (CSV has "In Search of the Castaways" with title case)
        book_name_lower = row['book_name'].lower()
        if book_name_lower == "in search of the castaways":
            book_path = "Books/In search of the castaways.txt"
        elif row['book_name'] == "The Count of Monte Cristo":
            book_path = "Books/The Count of Monte Cristo.txt"
        else:
            logger.warning(f"Book '{row['book_name']}' not found in predefined paths.")
            print(f"❌ Book '{row['book_name']}' not found. Skipping this row.")
            continue

        if flag[row['book_name']] == 0:
            # build_index will normalize book name to lowercase internally
            # Pass defaults for optional parameters (will use env vars or defaults)
            build_index(book_path, row['book_name'])
            flag[row['book_name']] = 1

        try:
            row_data = {
                "book_name": row["book_name"],
                "char": row["char"],
                "content": row["content"]
            }
            
            final_state = run_pipeline_for_row(row_data)
            results.append(final_state)
            
            # Print output for this row
            print_output(final_state)

            # SAVE INCREMENTALLY
            label_str = "CONSISTENT" if final_state.get("label") == 1 else "CONTRADICTING"
            reasoning = final_state.get("reasoning", "")
            current_row_df = pd.DataFrame([{
                "book_name": final_state.get("book_name"),
                "character_name": final_state.get("character_name"),
                "predictions": label_str,
                "reasoning": reasoning
            }])
            current_row_df.to_csv(output_file, mode='a', header=False, index=False)
            logger.info(f"Appended result for {row['char']} to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing row {idx + 1}: {e}", exc_info=True)
            print(f"❌ Error processing this row: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed processing {len(results)}/{len(df)} backstories")
    print(f"{'='*80}\n")

    # Save output to CSV
    if results:
        output_rows = []
        for state in results:
            label_str = "CONSISTENT" if state.get("label") == 1 else "CONTRADICTING"
            reasoning = state.get("reasoning", "")
            output_rows.append({
                "book_name": state.get("book_name"),
                "character_name": state.get("character_name"),
                "predictions": label_str,
                "reasoning": reasoning
            })
        
        output_df = pd.DataFrame(output_rows)
        output_file = "output.csv"
        output_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        print(f"✅ Results saved to {output_file}")
    else:
        logger.warning("No results to save.")



if __name__ == "__main__":
    main()