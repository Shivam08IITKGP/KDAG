"""Justification Agent: Defense Attorney logic."""
import logging
import json
from typing import TypedDict, List
from shared_config import create_llm
from langchain_core.prompts import ChatPromptTemplate
from answering_agent.evidence_generator import retrieve_evidence_for_queries

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

class JustificationLLMOutput(BaseModel):
    """Output structure for the Justification LLM."""
    label: str = Field(description="The determined label: CONSISTENT or CONTRADICTING.")
    reasoning: str = Field(description="A strictly 1 to 2 line concise justification.")
    evidence_queries: List[str] = Field(description="Specific queries to fetch supporting evidence.")

class JustificationOutput(TypedDict):
    """Final output from the Justification Agent."""
    reasoning: str
    evidence_chunks: List[dict]

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

Narrative Summary (overview of character's life):
{narrative_summary}

Knowledge Graph (all canonical facts extracted from the novel):
{full_graph_text}

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

**OUTPUT INSTRUCTIONS**
- The reasoning MUST be strictly 1 to 2 lines only.
- Do NOT say 'could be' or 'might be'.
- Do NOT hedge.
- Do NOT introduce counter-arguments.
- Your role is to JUSTIFY, not to decide.
"""

def generate_justification(
    book_name: str,
    character_name: str,
    backstory: str,
    narrative_summary: str,
    full_graph_text: str,
    character_summary: str,
    target_label: int,  # 1 for Consistent, 0 for Contradictory
) -> JustificationOutput:
    """Generate reasoning and retrieve evidence to justify a specific label."""
    
    label_str = "CONSISTENT" if target_label == 1 else "CONTRADICTING"
    logger.info(f"Generating justification for FORCED verdict: {label_str}")
    
    llm = create_llm()
    structured_llm = llm.with_structured_output(JustificationLLMOutput)
    
    prompt = ChatPromptTemplate.from_template(DEFENSE_ATTORNEY_SYSTEM_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result: JustificationLLMOutput = chain.invoke({
            "book_name": book_name,
            "character_name": character_name,
            "backstory": backstory,
            "narrative_summary": narrative_summary,
            "full_graph_text": full_graph_text,
            "character_summary": character_summary,
            "target_label": label_str
        })
        
        reasoning = result.reasoning
        queries = result.evidence_queries
        
    except Exception as e:
        logger.error(f"Error calling structured LLM for Justification: {e}")
        reasoning = f"The model has determined this claim is {label_str} based on aggregated character analysis."
        queries = [f"Find evidence related to {character_name} in {book_name}"]

    # Strictly enforce 1-2 lines by taking the first two sentences or lines
    reasoning_parts = reasoning.split('\n')
    if len(reasoning_parts) > 2:
        reasoning = " ".join(reasoning_parts[:2]).strip()
    
    # If it's still very long, just keep the first 200 characters or so to be safe
    # But usually the split by newline is enough for 'lines'
        
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

