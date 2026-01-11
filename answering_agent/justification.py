"""Justification Agent: Defense Attorney logic."""
import logging
from typing import TypedDict, List
from shared_config import create_llm
from langchain_core.prompts import ChatPromptTemplate
from answering_agent.evidence_generator import retrieve_evidence_for_queries

logger = logging.getLogger(__name__)

class JustificationOutput(TypedDict):
    reasoning: str
    evidence_chunks: List[str]

DEFENSE_ATTORNEY_SYSTEM_PROMPT = """
You are an expert narrative consistency verifier acting in a DEFENSIVE JUSTIFICATION role.

Your task is NOT to decide the label.
The final verdict has already been determined as: {target_label}
(1 = CONSISTENT, 0 = CONTRADICTORY)

Your responsibility is to produce a rigorous, canon-grounded reasoning that JUSTIFIES this verdict.

You must follow the same analytical rigor as a normal classification, but your reasoning must be strictly aligned with the given label.
You are not allowed to overturn or hedge against the verdict.

---CANONICAL CONTEXT---
Book: {book_name}
Character: {character_name}

Canonical Character Summary:
{character_summary}

Knowledge Graph (canonical facts extracted from the novel):
{graph_summary}

---BACKSTORY TO ANALYZE---
{backstory}

---CRITICAL ANALYSIS FRAMEWORK---

You must analyze the backstory through the SAME THREE LENSES, in the SAME ORDER, as a standard verification.
However, your conclusions in each lens must SUPPORT the final verdict: {target_label}.

---

**LENS 1: FACTUAL CONTRADICTION (Hardest Check)**

- If {target_label} = 0 (CONTRADICTORY):
  Identify the exact factual, historical, or timeline-based contradiction.
  Cite the specific canonical fact or graph sequence that makes the backstory impossible.

- If {target_label} = 1 (CONSISTENT):
  Explicitly argue that no canonical fact or graph sequence directly negates the backstory.
  If something appears suspicious, explain why it does NOT rise to the level of a factual contradiction.

Use the Knowledge Graph’s chronological relations (FOLLOWED_BY, NEXT_ACTION_IS, NEXT_LOCATED_AT) to support your argument.

---

**LENS 2: NARRATIVE PLAUSIBILITY (Medium Check)**

- If {target_label} = 0 (CONTRADICTORY):
  Explain how the backstory violates narrative constraints such as:
  - Character agency
  - Established roles
  - Causal chains
  - Completeness of canon-described events

- If {target_label} = 1 (CONSISTENT):
  Argue that the backstory respects:
  - The character’s canonical role
  - The novel’s internal logic
  - Known plot dependencies

If ambiguity exists, resolve it in favor of the given verdict.

---

**LENS 3: ADDITIONAL INFORMATION (Easiest Check)**

- If {target_label} = 1 (CONSISTENT):
  Frame the backstory as a natural, non-conflicting expansion that:
  - Fills gaps
  - Adds motivation
  - Elaborates on vague canon without altering facts

- If {target_label} = 0 (CONTRADICTORY):
  You may either:
  - Briefly state that Lens 3 is not applicable due to earlier contradictions
  - Or explain why the added detail exceeds permissible expansion

---

**ADVERSARIAL STABILITY CHECK**

Before finalizing, ensure:
- You are not contradicting any explicit canon
- You are not inventing new facts
- You are not weakening the given verdict with speculative language

You must sound confident, precise, and evidence-driven.

---

**OUTPUT FORMAT (STRICT JSON)**

{
  "label": {target_label},
  "reasoning": "A strictly 1 to 2 line concise justification of why the given verdict is correct based on the canon/graph.",
  "evidence_queries": [
    "Query 1: [Specific canonical fact that supports the verdict]",
    "Query 2: [Timeline or causal verification query]",
    "Query 3: [Narrative or role-based verification query]"
  ]
}

**IMPORTANT CONSTRAINTS**
- The verdict ({target_label}) is FINAL and authoritative.
- The 'reasoning' MUST be strictly 1 to 2 lines only.
- Do NOT say 'could be' or 'might be'.
- Do NOT hedge.
- Do NOT introduce counter-arguments.
- Your role is to JUSTIFY, not to decide.
"""

def generate_justification(
    book_name: str,
    character_name: str,
    backstory: str,
    graph_summary: str,
    character_summary: str,
    target_label: int,  # 1 for Consistent, 0 for Contradictory
) -> JustificationOutput:
    """Generate reasoning and retrieve evidence to justify a specific label."""
    
    label_str = "CONSISTENT" if target_label == 1 else "CONTRADICTING"
    logger.info(f"Generating justification for FORCED verdict: {label_str}")
    
    llm = create_llm()
    prompt = ChatPromptTemplate.from_template(DEFENSE_ATTORNEY_SYSTEM_PROMPT)
    chain = prompt | llm
    
    response = chain.invoke({
        "book": book_name,
        "character": character_name,
        "backstory": backstory,
        "graph_summary": graph_summary,
        "character_summary": character_summary,
        "target_label": label_str
    })
    
    content = response.content
    logger.debug(f"Defense Attorney Raw Output:\n{content}")
    
    # Parse JSON output
    try:
        # Try to extract JSON from response
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]
        
        result = json.loads(content)
        reasoning = result.get("reasoning", "")
        queries = result.get("evidence_queries", [])
        
    except Exception as e:
        logger.error(f"Error parsing Defense Attorney response: {e}")
        # Fallback to manual parsing if JSON fails
        reasoning_lines = []
        queries = []
        capture_reasoning = False
        
        for line in content.split('\n'):
            line_strip = line.strip()
            if "reasoning\": \"" in line_strip.lower() or "reasoning\":" in line_strip.lower():
                capture_reasoning = True
                parts = line_strip.split(":", 1)
                if len(parts) > 1:
                    val = parts[1].strip().strip('",')
                    if val: reasoning_lines.append(val)
            elif "query" in line_strip.lower() and ":" in line_strip:
                capture_reasoning = False
                parts = line_strip.split(":", 1)
                if len(parts) > 1:
                    queries.append(parts[1].strip().strip('",[]'))
            elif capture_reasoning and line_strip and not line_strip.startswith("}"):
                reasoning_lines.append(line_strip.strip('",'))
        
        reasoning = " ".join(reasoning_lines).strip()
            
    if not reasoning:
        reasoning = f"The ML model has determined this claim is {label_str} based on aggregated feature analysis."
    
    # Strictly enforce 1-2 lines by taking the first two sentences or lines if somehow it's still long
    reasoning_parts = reasoning.split('\n')
    if len(reasoning_parts) > 2:
        reasoning = " ".join(reasoning_parts[:2]).strip()
        
    logger.info(f"Generated Reasoning: {reasoning}")
    logger.info(f"Generated Queries: {queries}")
    
    # Retrieve evidence
    evidence_output = retrieve_evidence_for_queries(
        evidence_queries=queries,
        book_name=book_name,
        k=2 # Get top 2 chunks per query
    )
    
    return {
        "reasoning": reasoning,
        "evidence_chunks": evidence_output["evidence_chunks"]
    }
