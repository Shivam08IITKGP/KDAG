"""Evidence generator module for answering agent."""
import json
import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI

from answering_agent.prompts import EVIDENCE_SELECTION_PROMPT

logger = logging.getLogger(__name__)

class EvidenceOutput(TypedDict):
    """Output from evidence generator."""
    evidence_ids: list[str]


def generate_evidence_ids(
    reasoning: str,
    evidences: list[dict],
    backstory: str,
    llm: ChatOpenAI,
) -> EvidenceOutput:
    """Generate evidence IDs that support or contradict the reasoning.
    
    Args:
        reasoning: The reasoning from classification.
        evidences: List of evidence dicts with 'id' and 'text' keys.
        backstory: The backstory being checked.
        llm: ChatOpenAI instance.
        
    Returns:
        EvidenceOutput TypedDict.
    """
    logger.info("Starting evidence ID generation")
    
    # Format evidence list
    evidence_list_parts = []
    for ev in evidences:
        ev_id = ev.get("id", "unknown")
        ev_text = ev.get("text", "")
        evidence_list_parts.append(f"ID: {ev_id}\nText: {ev_text}\n")
    
    evidence_list = "\n".join(evidence_list_parts)
    
    # Format prompt
    prompt = EVIDENCE_SELECTION_PROMPT.format(
        reasoning=reasoning,
        backstory=backstory,
        evidence_list=evidence_list,
    )
    
    logger.debug(f"Evidence selection prompt: {prompt[:200]}...")
    
    # Get LLM response
    logger.info("Calling LLM for evidence selection")
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    logger.debug(f"LLM response: {response_text}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response
        if "```" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        
        # Validate and create TypedDict
        evidence_ids = result.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            logger.warning(f"evidence_ids is not a list: {evidence_ids}")
            evidence_ids = []
        
        # Filter to only include IDs that exist in evidences
        valid_evidence_ids = {ev.get("id") for ev in evidences}
        evidence_ids = [ev_id for ev_id in evidence_ids if ev_id in valid_evidence_ids]
        
        output: EvidenceOutput = {
            "evidence_ids": evidence_ids,
        }
        
        logger.info(f"Generated {len(evidence_ids)} evidence IDs")
        return output
        
    except Exception as e:
        logger.error(f"Error parsing evidence selection response: {e}")
        logger.error(f"Response text: {response_text}")
        
        # Return default
        return EvidenceOutput(evidence_ids=[])
