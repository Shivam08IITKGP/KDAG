"""NLI Checker module for verify consistency using Cross-Encoders."""
import logging
from typing import TypedDict, List
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cache the model to avoid reloading
_NLI_MODEL = None
MODEL_NAME = 'cross-encoder/nli-deberta-v3-base'

class NLIScores(TypedDict):
    entailment_avg: float
    contradiction_max: float
    contradiction_avg: float
    details: List[dict]

def _get_model():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        logger.info(f"Loading NLI model: {MODEL_NAME} with max_length=1024")
        # The cleanest way to set max length for CrossEncoder is via the constructor
        _NLI_MODEL = CrossEncoder(MODEL_NAME, max_length=1024)
        
        # Verification log
        if hasattr(_NLI_MODEL, 'tokenizer'):
             actual_max = getattr(_NLI_MODEL.tokenizer, 'model_max_length', 'unknown')
             logger.info(f"✅ Model loaded. Tokenizer model_max_length: {actual_max}")

    return _NLI_MODEL

def check_nli(backstory: str, graph_summary: str) -> NLIScores:
    """
    Calculate NLI metrics for backstory against the graph summary.
    
    Args:
        backstory: The backstory text to verify.
        graph_summary: Text summary of the knowledge graph (facts + timeline).
        
    Returns:
        NLIScores dictionary with entailment/contradiction probabilities.
    """
    if not graph_summary or graph_summary == "No graph data available.":
        return {
            "entailment_avg": 0.0,
            "contradiction_max": 0.0,
            "contradiction_avg": 0.0,
            "details": []
        }

    model = _get_model()
    
    # Create a single pair: (Graph Summary, Backstory)
    # The CrossEncoder typically expects (Premise, Hypothesis)
    # Premise = Graph Summary (Ground Truth)
    # Hypothesis = Backstory (Claim)
    pairs = [(graph_summary, backstory)]
    
    logger.info(f"Running NLI on single pair: Graph Summary ({len(graph_summary)} chars) vs Backstory ({len(backstory)} chars)")
    
    scores = model.predict(pairs)
    
    import numpy as np
    
    def softmax(x):
        # Handle 1D array output for single pair
        if x.ndim == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    probs = softmax(scores)
    
    # Mapping for Deberta v3 base NLI: 0: contradiction, 1: entailment, 2: neutral
    contradiction_prob = float(probs[0][0])
    entailment_prob = float(probs[0][1])
    neutral_prob = float(probs[0][2])

    logger.info(f"NLI Result: Entailment={entailment_prob:.4f}, Contradiction={contradiction_prob:.4f}")
    
    details = [{
        "evidence_prefix": "Graph Summary",
        "contradiction": contradiction_prob,
        "entailment": entailment_prob,
        "neutral": neutral_prob
    }]
    
    # Since we have only one pair, Max = Avg
    return {
        "entailment_avg": entailment_prob,
        "contradiction_max": contradiction_prob,
        "contradiction_avg": contradiction_prob,
        "details": details
    }
