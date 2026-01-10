"""Module for extracting knowledge triplets from text using LLM."""
import json
import logging
from typing import List

from langchain_openai import ChatOpenAI

from graph_creator_agent.prompts import TRIPLET_EXTRACTION_PROMPT
from graph_creator_agent.types import Triplet, TripletList

logger = logging.getLogger(__name__)


def generate_triplets(evidence_list: list[dict], llm: ChatOpenAI) -> list[Triplet]:
    """Generate triplets from evidence using LLM with structured output.
    
    Args:
        evidence_list: List of evidence dicts with 'id' and 'text' keys.
        llm: ChatOpenAI instance configured with structured output.
        
    Returns:
        List of Triplet TypedDicts.
    """
    if not evidence_list:
        return []

    all_triplets = []
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(TripletList)
    
    # helper to format evidence list
    formatted_evidence = []
    for ev in evidence_list:
        formatted_evidence.append(f"ID: {ev['id']}\nText: {ev['text']}\n")
    
    combined_evidence_text = "\n---\n".join(formatted_evidence)
    
    logger.info(f"Generating triplets for {len(evidence_list)} evidence items in bulk")
    
    # Format prompt
    prompt = TRIPLET_EXTRACTION_PROMPT.format(
        evidence_text=combined_evidence_text
    )
    
    try:
        # Get structured LLM response
        response: TripletList = structured_llm.invoke(prompt)
        
        # Extract triplets from response
        triplets = response.get("triplets", [])
        
        # Validate and ensure evidence_id is set correctly
        for triplet in triplets:
            if not triplet.get("evidence_id"):
                 logger.warning(f"Triplet missing evidence_id: {triplet}")
                 continue

            all_triplets.append(Triplet(
                subject=triplet["subject"],
                relation=triplet["relation"],
                object=triplet["object"],
                evidence_id=triplet["evidence_id"],
            ))
        
        logger.info(f"Generated {len(all_triplets)} triplets in total")
        
    except Exception as e:
        logger.error(f"Error generating triplets in bulk: {e}")
        # Fallback: try without structured output
        try:
            logger.warning("Falling back to JSON parsing")
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Try to extract JSON from response
            if "```" in response_text:
                start = response_text.find("[")
                end = response_text.rfind("]") + 1
                if start != -1 and end > start:
                    response_text = response_text[start:end]
                # Check for wrapped json block specifically 
                elif "```json" in response_text:
                     start = response_text.find("{")
                     end = response_text.rfind("}") + 1
                     response_text = response_text[start:end]

            # Try parsing as object with 'triplets' key first
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and "triplets" in data:
                    triplets = data["triplets"]
                elif isinstance(data, list):
                    triplets = data
                else:
                    raise ValueError("Response is not a list or dict with 'triplets' key")
            except json.JSONDecodeError:
                # If simple parse fails, try to find the list or dict again
                 start_list = response_text.find("[")
                 start_dict = response_text.find("{")
                 
                 if start_list != -1 and (start_dict == -1 or start_list < start_dict):
                      end = response_text.rfind("]") + 1
                      triplets = json.loads(response_text[start_list:end])
                 elif start_dict != -1:
                      end = response_text.rfind("}") + 1
                      data = json.loads(response_text[start_dict:end])
                      triplets = data.get("triplets", [])
                 else:
                      raise

            
            for triplet in triplets:
                if all(key in triplet for key in ["subject", "relation", "object", "evidence_id"]):
                    all_triplets.append(Triplet(
                        subject=triplet["subject"],
                        relation=triplet["relation"],
                        object=triplet["object"],
                        evidence_id=triplet["evidence_id"],
                    ))
            logger.info(f"Fallback: Generated {len(all_triplets)} triplets")
        except Exception as fallback_error:
            logger.error(f"Fallback parsing also failed: {fallback_error}")
    
    return all_triplets
