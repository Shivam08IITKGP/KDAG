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
        logger.info(f"Loading NLI model: {MODEL_NAME}")
        _NLI_MODEL = CrossEncoder(MODEL_NAME)
    return _NLI_MODEL

def check_nli(backstory: str, evidences: list[str]) -> NLIScores:
    """
    Calculate NLI metrics for backstory against evidences.
    
    Args:
        backstory: The backstory text to verify.
        evidences: List of evidence text strings.
        
    Returns:
        NLIScores dictionary with max and average scores.
    """
    if not evidences:
        return {
            "entailment_avg": 0.0,
            "contradiction_max": 0.0,
            "contradiction_avg": 0.0,
            "details": []
        }

    model = _get_model()
    pairs = [(evidence, backstory) for evidence in evidences]
    
    logger.info(f"Running NLI on {len(pairs)} pairs")
    
    scores = model.predict(pairs)
    
    import numpy as np
    
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    probs = softmax(scores)
    
    # Mapping for Deberta v3 base NLI: 0: contradiction, 1: entailment, 2: neutral
    contradiction_probs = probs[:, 0]
    entailment_probs = probs[:, 1]
    neutral_probs = probs[:, 2]

    entailment_avg = np.mean(entailment_probs)
    contradiction_max = np.max(contradiction_probs)
    contradiction_avg = np.mean(contradiction_probs)
    
    details = []
    for i, evidence in enumerate(evidences):
        details.append({
            "evidence_prefix": evidence[:50] + "...",
            "contradiction": float(probs[i][0]),
            "entailment": float(probs[i][1]),
            "neutral": float(probs[i][2])
        })
        
    logger.info(f"NLI Metrics: E_Avg={entailment_avg:.4f}, C_Max={contradiction_max:.4f}, C_Avg={contradiction_avg:.4f}")
    
    return {
        "entailment_avg": float(entailment_avg),
        "contradiction_max": float(contradiction_max),
        "contradiction_avg": float(contradiction_avg),
        "details": details
    }
