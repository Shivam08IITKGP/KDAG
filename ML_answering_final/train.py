"""Train ML models (Logistic Regression + XGBoost) on extracted features."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA

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
    
    # Separate core features and embeddings
    core_feature_cols = ["llm_prediction", "contradiction_max", "consistency_avg", "contradiction_avg"]
    emb_cols = [f"emb_{i}" for i in range(384)]
    
    # Check core features
    available_core = [c for c in core_feature_cols if c in df_merged.columns]
    print(f"Using {len(available_core)} core feature columns")
    
    # Check embeddings
    available_emb = [c for c in emb_cols if c in df_merged.columns]
    
    X_core = df_merged[available_core].values
    y = np.array(labels)
    
    if len(available_emb) > 0:
        print(f"Applying PCA to {len(available_emb)} embedding dimensions -> 7 components")
        X_emb = df_merged[available_emb].values
        
        pca = PCA(n_components=10, random_state=42)
        X_pca = pca.fit_transform(X_emb)
        
        # Concatenate core features with PCA components
        X = np.hstack([X_core, X_pca])
        print(f"Final feature vector size: {X.shape[1]} (Core: {X_core.shape[1]} + PCA: {X_pca.shape[1]})")
        
        # We need to return the PCA model to save it
        return X, y, pca
    else:
        print("Warning: No embeddings found. Using only core features.")
        return X_core, y, None


def train_and_evaluate(X: np.ndarray, y: np.ndarray, model, model_name: str, test_size: float = 0.2, train_all: bool = False):
    """Train and evaluate a model.
    
    Args:
        X: features
        y: labels
        model: sklearn/xgboost model
        model_name: string for printing
        test_size: fraction for testing
        train_all: if True, train on everything and evaluate on everything
    """
    
    if train_all:
        X_train, X_test, y_train, y_test = X, X, y, y
        split_msg = f"Training and Evaluating on ALL {len(X)} samples"
    # Handle small datasets - if too small for stratify, use simple split
    elif len(y) < 5:
        print(f"Warning: Very small dataset ({len(y)} samples). Using all data for training.")
        X_train, X_test, y_train, y_test = X, X, y, y
        split_msg = f"Training and Evaluating on ALL {len(X)} samples"
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
        split_msg = f"Training set: {len(X_train)} samples\nTest set: {len(X_test)} samples"
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(split_msg)
    
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
    
    
def load_inference_artifacts(output_dir: str = "ML_answering_final"):
    """Load trained artifacts for inference.
    
    Args:
        output_dir: Directory containing models
        
    Returns:
        tuple (pca_model, clf_model) or (None, None) if not found
    """
    try:
        pca_path = f"{output_dir}/pca_model.pkl"
        # First try XGBoost, fall back to Logistic Regression
        xgb_path = f"{output_dir}/xgb_model.pkl"
        lr_path = f"{output_dir}/logreg_model.pkl"
        
        pca_model = None
        if Path(pca_path).exists():
            with open(pca_path, "rb") as f:
                pca_model = pickle.load(f)
                
        clf_model = None
        if HAS_XGBOOST and Path(xgb_path).exists():
            print(f"Loading XGBoost model from {xgb_path}")
            with open(xgb_path, "rb") as f:
                clf_model = pickle.load(f)
        elif Path(lr_path).exists():
            print(f"Loading Logistic Regression model from {lr_path}")
            with open(lr_path, "rb") as f:
                clf_model = pickle.load(f)
        else:
            print("Error: No trained classification model found.")
            
        return pca_model, clf_model
        
    except Exception as e:
        print(f"Error loading inference artifacts: {e}")
        return None, None


def predict_single_sample(features_dict: dict, pca_model, clf_model) -> tuple[int, float]:
    """Predict label for a single sample.
    
    Args:
        features_dict: Dictionary containing features
        pca_model: Trained PCA model
        clf_model: Trained classifier
        
    Returns:
        (prediction_label, confidence_score)
        Label: 1 (CONSISTENT), 0 (CONTRADICTING)
    """
    # Define feature order - MUST match training exactly
    core_features = [
        features_dict.get("llm_prediction", 0.5), # Default to uncertain if missing
        features_dict.get("contradiction_max", 0.0),
        features_dict.get("consistency_avg", 0.0),
        features_dict.get("contradiction_avg", 0.0)
    ]
    
    # Extract embeddings
    embeddings = []
    # If passed as list
    if "embeddings" in features_dict:
        embeddings = features_dict["embeddings"]
    # If passed as dict keys
    else:
        for i in range(384):
            embeddings.append(features_dict.get(f"emb_{i}", 0.0))
            
    embeddings = np.array(embeddings).reshape(1, -1)
    
    # Apply PCA if available
    if pca_model:
        pca_featuers = pca_model.transform(embeddings)
        final_features = np.hstack([np.array(core_features).reshape(1, -1), pca_featuers])
    else:
        final_features = np.array(core_features).reshape(1, -1)
        
    # Predict
    prediction = clf_model.predict(final_features)[0]
    
    # Get probability/confidence if available
    confidence = 1.0
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(final_features)[0]
        confidence = probs[prediction]
        
    return int(prediction), float(confidence)



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
    parser.add_argument("--all", action="store_true", help="Train and evaluate on all samples (no split)")
    args = parser.parse_args()
    
    print(f"Loading features from: {args.input}")
    print(f"Loading labels from: {args.labels}")
    
    X, y, pca_model = load_features(args.input, args.labels)
    print(f"\nTotal samples: {len(X)} with {X.shape[1]} features")
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save PCA model
    if pca_model:
        save_model(pca_model, f"{args.output_dir}/pca_model.pkl", "PCA Model")
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model = train_and_evaluate(X, y, lr_model, "LOGISTIC REGRESSION", args.test_size, args.all)
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
        xgb_model = train_and_evaluate(X, y, xgb_model, "XGBOOST", args.test_size, args.all)
        save_model(xgb_model, f"{args.output_dir}/xgb_model.pkl", "XGBoost")
    else:
        print("\nSkipping XGBoost (not installed). Run: pip install xgboost")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
