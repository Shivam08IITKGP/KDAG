"""Feature extraction for ML decision layer."""
import numpy as np
from sentence_transformers import SentenceTransformer

# Load MiniLM model (384-dim, lightweight)
_model = None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def get_backstory_embedding(text: str) -> np.ndarray:
    """Generate 384-dim embedding for backstory text.
    
    Args:
        text: Backstory text
        
    Returns:
        numpy array of shape (384,)
    """
    model = get_embedding_model()
    return model.encode(text, convert_to_numpy=True)


def extract_features(state: dict) -> dict:
    """Extract all features from pipeline state.
    
    Args:
        state: PipelineState dictionary with all results
        
    Returns:
        Dictionary with feature names and values
    """
    # LLM prediction (binary)
    llm_pred = 1 if state.get("label") == 1 else 0
    
    # NLI scores
    contradiction_max = state.get("nli_max_contradiction", 0.0)
    consistency_avg = state.get("nli_avg_entailment", 0.0)
    contradiction_avg = state.get("nli_avg_contradiction", 0.0)
    
    # Backstory embedding
    backstory = state.get("backstory", "")
    embedding = get_backstory_embedding(backstory)
    
    # Build feature dict
    features = {
        "llm_prediction": llm_pred,
        "contradiction_max": contradiction_max,
        "consistency_avg": consistency_avg,
        "contradiction_avg": contradiction_avg,
    }
    
    # Add embedding columns
    for i, val in enumerate(embedding):
        features[f"emb_{i}"] = float(val)
    
    return features


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to numpy array for model input.
    
    Args:
        features: Dictionary from extract_features
        
    Returns:
        numpy array of shape (388,)
    """
    arr = [
        features["llm_prediction"],
        features["contradiction_max"],
        features["consistency_avg"],
        features["contradiction_avg"],
    ]
    # Add embedding values
    for i in range(384):
        arr.append(features[f"emb_{i}"])
    
    return np.array(arr)
