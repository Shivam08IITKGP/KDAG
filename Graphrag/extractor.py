"""
Entity Extraction and Graph Generation Module
Handles LLM-based extraction of entities, relationships, and character states
"""

import json
import logging
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime

import requests
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    SYSTEM_PROMPT_ENTITY_EXTRACTION,
    SYSTEM_PROMPT_CONTRADICTION_DETECTION,
    SYSTEM_PROMPT_CLAIM_EXTRACTION,
    VERBOSE,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

class GraphRAGExtractor:
    """
    Handles extraction of entities, relationships, and states from narrative text
    using OpenRouter API with free-tier LLM models.
    """

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.llm_model = LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
        }

    def call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """
        Call OpenRouter API to extract structured data from text.
        
        Args:
            system_prompt: System role definition
            user_prompt: The text to analyze
            temperature: Lower = more deterministic, higher = more creative
        
        Returns:
            JSON response from LLM
        """
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": 4096,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected API response: {result}")
                return "{}"

        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return "{}"

    def extract_entities_and_relationships(self, text_chunk: str, chunk_id: int) -> Dict[str, Any]:
        """
        Extract entities, relationships, and states from a text chunk.
        
        Args:
            text_chunk: The narrative text to analyze
            chunk_id: Unique identifier for this chunk
        
        Returns:
            Dictionary with extracted data
        """
        if VERBOSE:
            logger.info(f"Extracting entities from chunk {chunk_id}...")

        prompt = f"""Analyze this narrative text chunk and extract all entities, relationships, and character states:

---TEXT CHUNK {chunk_id}---
{text_chunk}
---END CHUNK---

Return ONLY valid JSON.
Do NOT include explanations, markdown, or text outside JSON.
If unsure, return empty lists but still valid JSON.
"""

        response = self.call_llm(SYSTEM_PROMPT_ENTITY_EXTRACTION, prompt, temperature=0.2)

        try:
            # Try to parse JSON
            data = json.loads(response)
            data["chunk_id"] = chunk_id
            data["extraction_timestamp"] = datetime.now().isoformat()
            return data
        except json.JSONDecodeError:
            logger.debug(f"Non-JSON response for chunk {chunk_id}, skipping")

            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    data["chunk_id"] = chunk_id
                    return data
                except json.JSONDecodeError:
                    pass
            return {
                "characters": [],
                "locations": [],
                "events": [],
                "relationships": [],
                "chunk_id": chunk_id,
                "error": "Failed to parse extraction",
            }

    def extract_backstory_claims(self, backstory_text: str, character_name: str) -> Dict[str, Any]:
        """
        Break down a character backstory into atomic, verifiable claims.
        
        Args:
            backstory_text: The backstory to deconstruct
            character_name: Character whose backstory we're analyzing
        
        Returns:
            Dictionary with extracted claims
        """
        if VERBOSE:
            logger.info(f"Extracting claims from {character_name}'s backstory...")

        prompt = f"""Character: {character_name}

Backstory:
{backstory_text}

Extract EVERY atomic claim that can be verified against the novel text."""

        response = self.call_llm(SYSTEM_PROMPT_CLAIM_EXTRACTION, prompt, temperature=0.1)

        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data
                except json.JSONDecodeError:
                    pass
            return {
                "character": character_name,
                "claims": [],
                "error": "Failed to parse claims",
            }

    def detect_contradictions(
        self,
        backstory_claim: str,
        evidence_pack: List[Dict[str, str]],
        character_name: str,
    ) -> Dict[str, Any]:
        """
        Check if a backstory claim contradicts evidence from the novel.
        
        Args:
            backstory_claim: The claim to verify
            evidence_pack: List of evidence snippets from novel
            character_name: Character being analyzed
        
        Returns:
            Dictionary with contradiction analysis
        """
        evidence_text = "\n".join([f"- {e.get('text', '')} (Ch. {e.get('chapter', 'unknown')})" for e in evidence_pack])

        prompt = f"""Character: {character_name}

Backstory Claim:
{backstory_claim}

Evidence from Novel:
{evidence_text if evidence_text else "No specific evidence found"}

Analyze for contradictions."""

        response = self.call_llm(SYSTEM_PROMPT_CONTRADICTION_DETECTION, prompt, temperature=0.1)

        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data
                except json.JSONDecodeError:
                    pass
            return {
                "claim": backstory_claim,
                "is_contradicted": False,
                "contradiction_type": "unknown",
                "confidence": 0.0,
                "reasoning": "Could not analyze",
            }

    def build_relationship_graph(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a graph of relationships from all extractions.
        
        Args:
            extractions: List of extraction results
        
        Returns:
            List of relationship edges
        """
        relationships = {}

        for extraction in extractions:
            for rel in extraction.get("relationships", []):
                source = rel.get("source_character", "").strip()
                target = rel.get("target_character", "").strip()

                if not source or not target:
                    continue

                # Create unique key for relationship (order-agnostic for undirected, order-aware for directed)
                key = f"{source}|{target}"

                if key not in relationships:
                    relationships[key] = {
                        "source": source,
                        "target": target,
                        "relationship_type": rel.get("relationship_type", "Unknown"),
                        "description": rel.get("description", ""),
                        "sentiment": rel.get("sentiment", "Neutral"),
                        "occurrences": 1,
                        "chunks": [extraction.get("chunk_id", 0)],
                    }
                else:
                    # Merge occurrences
                    relationships[key]["occurrences"] += 1
                    relationships[key]["chunks"].append(extraction.get("chunk_id", 0))

        return list(relationships.values())

    def extract_timeline(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a chronological timeline of events from extractions.
        
        Args:
            extractions: List of extraction results
        
        Returns:
            Sorted list of timeline events
        """
        events = []

        for extraction in extractions:
            for event in extraction.get("events", []):
                events.append({
                    "name": event.get("name", "Event"),
                    "description": event.get("description", ""),
                    "participants": event.get("participants", []),
                    "location": event.get("location", "Unknown"),
                    "timestamp_hint": event.get("timestamp_hint", "Unspecified"),
                    "event_type": event.get("event_type", "Other"),
                    "chunk_id": extraction.get("chunk_id", 0),
                })

        return events
