"""Prompts for graph creator agent."""
TRIPLET_EXTRACTION_PROMPT = """Extract knowledge triplets from the following evidence about the character.

Evidence:
{evidence_text}

Evidence ID: {evidence_id}

Extract triplets in the format: (subject, relation, object)
Each triplet should represent a fact about the character.

Return a JSON list of objects, each with:
- subject: string
- relation: string
- object: string
- evidence_id: string (use the provided evidence_id)

Example:
[
  {{"subject": "Character Name", "relation": "is", "object": "geographer", "evidence_id": "ev_1"}},
  {{"subject": "Character Name", "relation": "has_trait", "object": "absent-minded", "evidence_id": "ev_1"}}
]
"""
