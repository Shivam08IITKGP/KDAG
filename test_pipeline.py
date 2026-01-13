"""End-to-End Test Pipeline with ML Decision Layer."""
import logging
import pandas as pd
import time
from pathlib import Path
from typing import TypedDict

# Reuse existing pipeline components
from main import run_pipeline_for_row, setup_logging
from Graphrag.pathway.build_index import build_index
from ML_answering_final.features import extract_features
from ML_answering_final.train import load_inference_artifacts, predict_single_sample
from answering_agent.justification import generate_justification

import argparse

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="utils/test.csv", help="Input CSV")
    parser.add_argument("--limit", type=int, help="Limit number of rows")
    args = parser.parse_args()

    # Setup
    log_file = setup_logging()
    logger.info(f"Starting Test Pipeline. Log: {log_file}")
    
    # Load Test Data
    try:
        df = pd.read_csv(args.input)
        if args.limit:
            df = df.head(args.limit)
        logger.info(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        logger.error(f"Failed to load test.csv: {e}")
        return

    # Load ML Artifacts
    pca_model, clf_model = load_inference_artifacts()
    if not clf_model:
        logger.error("Failed to load ML models. Aborting.")
        return
        
    # Output file setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "final_output.csv"
    feature_file = output_dir / "features_test.csv"
    
    results = []
    
    # Track indexed books to avoid redundant calls
    indexed_books = set()

    print(f"\nProcessing {len(df)} rows...")
    
    for i, row in df.iterrows():
        print(f"\n[{i+1}/{len(df)}] Processing: {row['char']} ({row['book_name']})")
        
        # 1. Indexing (if needed)
        book_name = row['book_name']
        if book_name not in indexed_books:
            # Map book name to path (reusing logic from main.py)
            book_path = ""
            if book_name.lower() == "in search of the castaways":
                book_path = "Books/In search of the castaways.txt"
            elif book_name == "The Count of Monte Cristo":
                book_path = "Books/The Count of Monte Cristo.txt"
            
            if book_path:
                print(f"Ensuring index for: {book_name}")
                build_index(book_path, book_name)
                indexed_books.add(book_name)
            else:
                logger.warning(f"Unknown book: {book_name}")
        
        # 2. Run Base Pipeline (LLM Classification + Graph Construction)
        try:
            row_data = {
                "book_name": row["book_name"],
                "char": row["char"],
                "content": row["content"]
            }
            state = run_pipeline_for_row(row_data)
            
            # 3. Extract Features
            features = extract_features(state)
            
            # 4. ML Inference
            ml_pred, ml_conf = predict_single_sample(features, pca_model, clf_model)
            llm_pred = features["llm_prediction"]
            
            final_label = ml_pred
            final_reasoning = state["reasoning"]
            final_evidence = state.get("evidence_chunks", [])
            
            # 5. Review & Justify (The "Defense Attorney")
            # If ML disagrees with LLM
            if ml_pred != llm_pred:
                print(f"⚠️ DISAGREEMENT! LLM: {llm_pred} | ML: {ml_pred} (Conf: {ml_conf:.2f})")
                print("Invoking Justification Agent...")
                
                # Get graph summary from state (was generated in answering_agent.main)
                # We need to rely on what run_pipeline_for_row returned. 
                # Note: run_pipeline_for_row returns the final state of the graph execution.
                # However, graph_summary is usually passed into 'classify' but not explicitly stored in state
                # unless we added it.
                # Let's check state keys. Answering Agent main adds 'nli_checker' etc.
                # It does NOT strictly persist 'graph_summary' in the TypedDict unless we add it to PipelineState.
                # FIX: We can re-summarize or fetch it.
                # Actually, in Implementaion Plan we said "Update answering_agent/main.py to generate and pass graph summary".
                # Let's assume for now we can regenerate it or we missed storing it.
                # To be safe, let's regenerate it quickly if missing.
                
                # Actually, looking at answering/main.py again (viewed earlier), 
                # variables 'graph_summary' is local. It is NOT in PipelineState.
                # We should update answering/main.py to store it, OR just re-read the graph here.
                # Re-reading is safer for now without changing `main.py` state definition.
                
                graph_path = state.get("graph_path")
                narrative_summary = "Graph unavailable."
                full_graph_text = "Graph unavailable."
                if graph_path:
                    import networkx as nx
                    from answering_agent.classifier import get_graph_data
                    if Path(graph_path).exists():
                        g = nx.read_graphml(graph_path)
                        narrative_summary, full_graph_text = get_graph_data(g)
                
                # Get character summary (also local in answer function)
                from extraction_agent.character_summaries import get_character_summary
                char_summary = get_character_summary(book_name, row['char'])
                
                justification = generate_justification(
                    book_name=book_name,
                    character_name=row['char'],
                    backstory=row['content'],
                    narrative_summary=narrative_summary,
                    full_graph_text=full_graph_text,
                    character_summary=char_summary,
                    target_label=ml_pred
                )
                
                final_reasoning = justification["reasoning"]
                final_evidence = justification["evidence_chunks"]
                print(f"New Reasoning Generated: {final_reasoning[:100]}...")
            
            else:
                print(f"✅ Agreement: {ml_pred}")

            # 6. Save Result
            encoded_label = "CONSISTENT" if final_label == 1 else "CONTRADICTING"
            
            result_row = {
                "id": row["id"],
                "book_name": row["book_name"],
                "char": row["char"],
                "final_label": encoded_label,
                "ml_confidence": ml_conf,
                "llm_agreement": (ml_pred == llm_pred),
                "final_reasoning": final_reasoning,
                "evidence_used": str([c.get('text', '') for c in final_evidence])
            }
            results.append(result_row)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            # Save features for debugging
            features["id"] = row["id"]
            features["ml_pred"] = ml_pred
            features["llm_pred"] = llm_pred
            pd.DataFrame([features]).to_csv(feature_file, mode='a', header=(i==0), index=False)
            
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}", exc_info=True)
            print(f"Error: {e}")
            
    print(f"\nPipeline Complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
