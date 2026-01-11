
import logging
import numpy as np
from answering_agent.nli_checker import check_nli

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_score_aggregation():
    print("Verifying NLI Score Aggregation Logic...")
    
    # Mock data
    backstory = "The character is a brave knight."
    evidences = [
        "The character is a brave knight and fights dragons.", # High entailment
        "The character is a coward and runs away.",          # High contradiction
        "The character likes apples."                        # Neutral
    ]
    
    # Run check_nli
    scores = check_nli(backstory, evidences)
    
    print("\n--- Results ---")
    print(f"Evidences: {len(evidences)}")
    print(f"Entailment Avg (Consistency): {scores['entailment_avg']:.4f}")
    print(f"Contradiction Max: {scores['contradiction_max']:.4f}")
    print(f"Contradiction Avg: {scores['contradiction_avg']:.4f}")
    
    print("\n--- Detailed Scores per Evidence ---")
    for detail in scores['details']:
        print(f"Prefix: {detail['evidence_prefix']}")
        print(f"  Entailment: {detail['entailment']:.4f}")
        print(f"  Contradiction: {detail['contradiction']:.4f}")
        print(f"  Neutral: {detail['neutral']:.4f}")

    # Manual calculation check
    # We can't easily reproduce the exact model output without the model, 
    # but we can verify that the aggregates match the details.
    
    detail_entailments = [d['entailment'] for d in scores['details']]
    detail_contradictions = [d['contradiction'] for d in scores['details']]
    
    calc_ent_avg = np.mean(detail_entailments)
    calc_con_max = np.max(detail_contradictions)
    calc_con_avg = np.mean(detail_contradictions)
    
    print("\n--- Verification ---")
    print(f"Calculated Entailment Avg: {calc_ent_avg:.4f} == Reported: {scores['entailment_avg']:.4f}")
    print(f"Calculated Contradiction Max: {calc_con_max:.4f} == Reported: {scores['contradiction_max']:.4f}")
    print(f"Calculated Contradiction Avg: {calc_con_avg:.4f} == Reported: {scores['contradiction_avg']:.4f}")
    
    assert np.isclose(calc_ent_avg, scores['entailment_avg']), "Entailment Avg mismatch"
    assert np.isclose(calc_con_max, scores['contradiction_max']), "Contradiction Max mismatch"
    assert np.isclose(calc_con_avg, scores['contradiction_avg']), "Contradiction Avg mismatch"
    
    print("\nSUCCESS: All aggregations verified correctly against detailed scores.")

if __name__ == "__main__":
    test_score_aggregation()
