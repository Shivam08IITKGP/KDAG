"""Prompts for answering agent."""
CLASSIFICATION_PROMPT = """
You are a verification agent that checks whether a character backstory is CONSISTENT with a given knowledge graph extracted from a novel.

IMPORTANT RULES:
- Use ONLY the provided knowledge graph summary.
- Do NOT use outside knowledge or assumptions.
- The knowledge graph represents ground-truth facts from the book.
- If the backstory directly contradicts any graph fact, label = 0.
- If the backstory is fully supported or not contradicted, label = 1.
- Missing information is NOT a contradiction.

INPUTS:
Book: {book_name}
Character: {character_name}

Backstory to Verify:
{backstory}

Knowledge Graph Summary (facts and relations):
{graph_summary}

DECISION CRITERIA:
1. Factual consistency: Are all claims compatible with graph facts?
2. Temporal/causal consistency: Do events align logically with known causes and outcomes?
3. Narrative consistency: Does the backstory violate established story constraints?

OUTPUT FORMAT (strict JSON):
{{
  "label": 1 or 0,
  "reasoning": "Step-by-step justification explicitly referencing graph facts or stating absence of contradiction."
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
