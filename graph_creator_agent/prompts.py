"""Prompts for graph creator agent."""
TRIPLET_EXTRACTION_PROMPT = """Extract knowledge triplets from the following evidence about the character.

Evidence:
{evidence_text}

Evidence ID: {evidence_id}

Extract triplets in the format: (subject, relation, object)
Each triplet should represent a fact about the character.

Return a JSON object with a "triplets" key containing a list of objects, each with:
- subject: string
- relation: string
- object: string
- evidence_id: string (must be exactly: {evidence_id})

Example format (replace "ev_1" with the actual evidence_id):
{{
  "triplets": [
    {{"subject": "Character Name", "relation": "is", "object": "geographer", "evidence_id": "{evidence_id}"}},
    {{"subject": "Character Name", "relation": "has_trait", "object": "absent-minded", "evidence_id": "{evidence_id}"}}
  ]
}}
"""
