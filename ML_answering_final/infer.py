"""Inference script for single-row prediction."""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ML_answering_final.features import get_backstory_embedding, features_to_array


def load_model(model_path: str):
    """Load trained model from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def generate_features_for_row(row_idx: int, csv_path: str = "train.csv") -> dict:
    """Generate features for a single row.
    
    Note: This requires running the pipeline first to get LLM and NLI scores.
    For standalone inference, we use default values.
    
    Args:
        row_idx: Row index in CSV
        csv_path: Path to CSV file
        
    Returns:
        Feature dictionary
    """
    df = pd.read_csv(csv_path)
    row = df.iloc[row_idx]
    
    backstory = row["content"]
    embedding = get_backstory_embedding(backstory)
    
    # Build features
    # Note: For full pipeline, these come from pipeline state
    # For standalone, we set defaults (user should run full pipeline first)
    features = {
        "llm_prediction": 0,  # Placeholder - should come from pipeline
        "contradiction_max": 0.0,
        "consistency_avg": 0.0,
        "contradiction_avg": 0.0,
    }
    
    # Add embedding
    for i, val in enumerate(embedding):
        features[f"emb_{i}"] = float(val)
    
    return features, row


def predict(model, features: dict) -> tuple[int, str]:
    """Run model prediction.
    
    Args:
        model: Trained model
        features: Feature dictionary
        
    Returns:
        (binary_label, human_readable_label)
    """
    X = features_to_array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    label = "CONSISTENT" if pred == 1 else "CONTRADICTING"
    return pred, label


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single row")
    parser.add_argument("row_index", type=int, help="Row index from train.csv")
    parser.add_argument("--model", "-m", default="ML_answering_final/model.pkl", help="Model path")
    parser.add_argument("--csv", "-c", default="train.csv", help="CSV file path")
    parser.add_argument("--features-csv", "-f", default=None, help="Features CSV (if available)")
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    model = load_model(args.model)
    
    if args.features_csv:
        # Use pre-computed features
        df = pd.read_csv(args.features_csv)
        feature_cols = ["llm_prediction", "contradiction_max", "consistency_avg", "contradiction_avg"]
        feature_cols += [f"emb_{i}" for i in range(384)]
        
        row_features = df.iloc[args.row_index][feature_cols].to_dict()
        info_df = pd.read_csv(args.csv)
        row_info = info_df.iloc[args.row_index]
    else:
        # Generate embedding only (LLM/NLI scores need full pipeline)
        print("Warning: Using embedding only. Run full pipeline for complete features.")
        row_features, row_info = generate_features_for_row(args.row_index, args.csv)
    
    pred, label = predict(model, row_features)
    
    print("\n" + "="*60)
    print("INFERENCE RESULT")
    print("="*60)
    print(f"Row Index:  {args.row_index}")
    print(f"Character:  {row_info['char']}")
    print(f"Book:       {row_info['book_name']}")
    print(f"Prediction: {label} ({pred})")
    print("="*60)


if __name__ == "__main__":
    main()
