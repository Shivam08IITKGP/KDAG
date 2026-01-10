"""Prompts for answering agent."""
CLASSIFICATION_PROMPT = """You are a consistency checker for character backstories.

Compare the given backstory against the character's knowledge graph and determine if it is consistent or contradicting.

Book: {book_name}
Character: {character_name}
Backstory: {backstory}

Knowledge Graph Summary:
{graph_summary}

Analyze the backstory for:
1. Character graph consistency (does it match known facts about the character?)
2. Narrative constraints (does it fit the story context?)
3. Causal consistency (are the events logically consistent?)

Return a JSON object with:
- label: 1 if CONSISTENT, 0 if CONTRADICTING
- reasoning: A detailed explanation of your analysis

Example:
{{
  "label": 1,
  "reasoning": "The backstory is consistent because..."
}}
"""

EVIDENCE_SELECTION_PROMPT = """Select evidence IDs that best support or contradict the given reasoning.

Reasoning: {reasoning}

Backstory: {backstory}

Available Evidence:
{evidence_list}

Return a JSON object with:
- evidence_ids: A list of evidence IDs (strings) that are most relevant to the reasoning

Example:
{{
  "evidence_ids": ["ev_1", "ev_2", "ev_3"]
}}
"""
