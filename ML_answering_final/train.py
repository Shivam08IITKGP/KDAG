"""Train ML models (Logistic Regression + XGBoost) on extracted features."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Only Logistic Regression will be trained.")


def load_features(features_csv: str, labels_csv: str = "train.csv") -> tuple[np.ndarray, np.ndarray]:
    """Load features and merge with ground truth labels from train.csv.
    
    Args:
        features_csv: Path to CSV with extracted features (features_output.csv)
        labels_csv: Path to CSV with ground truth labels (train.csv)
        
    Returns:
        X (features), y (labels)
    """
    # Load features
    df_features = pd.read_csv(features_csv)
    print(f"Loaded {len(df_features)} rows from {features_csv}")
    
    # Load ground truth labels from train.csv
    df_labels = pd.read_csv(labels_csv)
    print(f"Loaded {len(df_labels)} rows from {labels_csv}")
    
    # The features_output.csv has 'row_index' column that maps to train.csv index
    if "row_index" not in df_features.columns:
        raise ValueError("features_output.csv must have 'row_index' column")
    
    # Merge to get ground truth labels
    # train.csv has: id, book_name, char, caption, content, label
    # The 'label' column in train.csv is the ground truth
    df_merged = df_features.copy()
    
    # Get labels for each row_index
    labels = []
    for idx in df_features["row_index"]:
        if idx < len(df_labels):
            label_str = df_labels.iloc[idx]["label"]
            # Handle lowercase labels: "consistent" -> 1, "contradict" -> 0
            if isinstance(label_str, str):
                labels.append(1 if label_str.lower() == "consistent" else 0)
            else:
                labels.append(int(label_str))
        else:
            print(f"Warning: row_index {idx} not found in train.csv")
            labels.append(0)
    
    df_merged["ground_truth"] = labels
    
    # Feature columns
    feature_cols = ["llm_prediction", "contradiction_max", "consistency_avg", "contradiction_avg"]
    feature_cols += [f"emb_{i}" for i in range(384)]
    
    # Check which columns exist
    available_cols = [c for c in feature_cols if c in df_merged.columns]
    print(f"Using {len(available_cols)} feature columns")
    
    X = df_merged[available_cols].values
    y = np.array(labels)
    
    print(f"Label distribution: {np.sum(y == 1)} consistent, {np.sum(y == 0)} contradicting")
    
    return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray, model, model_name: str, test_size: float = 0.2):
    """Train and evaluate a model."""
    
    # Handle small datasets - if too small for stratify, use simple split
    if len(y) < 5:
        print(f"Warning: Very small dataset ({len(y)} samples). Using all data for training.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails, use simple split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["CONTRADICTING", "CONSISTENT"], zero_division=0))
    
    return model


def save_model(model, path: str, model_name: str):
    """Save trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"{model_name} saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models on extracted features")
    parser.add_argument("--input", "-i", default="features_output.csv", help="Input CSV with features")
    parser.add_argument("--labels", "-l", default="train.csv", help="CSV with ground truth labels")
    parser.add_argument("--output-dir", "-o", default="ML_answering_final", help="Output directory for models")
    parser.add_argument("--test-size", "-t", type=float, default=0.2, help="Test set fraction")
    args = parser.parse_args()
    
    print(f"Loading features from: {args.input}")
    print(f"Loading labels from: {args.labels}")
    
    X, y = load_features(args.input, args.labels)
    print(f"\nTotal samples: {len(X)} with {X.shape[1]} features")
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model = train_and_evaluate(X, y, lr_model, "LOGISTIC REGRESSION", args.test_size)
    save_model(lr_model, f"{args.output_dir}/logreg_model.pkl", "Logistic Regression")
    
    # Train XGBoost
    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model = train_and_evaluate(X, y, xgb_model, "XGBOOST", args.test_size)
        save_model(xgb_model, f"{args.output_dir}/xgb_model.pkl", "XGBoost")
    else:
        print("\nSkipping XGBoost (not installed). Run: pip install xgboost")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
