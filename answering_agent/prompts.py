CLASSIFICATION_PROMPT = """
You are an expert narrative consistency verifier. Your task is to determine whether a proposed character backstory is CONSISTENT (1) or CONTRADICTORY (0) with the canonical novel.

---CANONICAL CONTEXT---
Book: {book_name}
Character: {character_name}

Canonical Character Summary:
{character_summary}

Knowledge Graph (canonical facts extracted from the novel):
{graph_summary}

---BACKSTORY TO VERIFY---
{backstory}

---CRITICAL ANALYSIS FRAMEWORK---

You must evaluate the backstory through THREE lenses, in this exact order:

**LENS 1: FACTUAL CONTRADICTION (Hardest Check)**
Does the backstory contain claims that directly oppose established canon?

⚠️ COMMON CONTRADICTION PATTERNS TO DETECT:
- **Historical Falsification**: "Napoleon's triumph at Waterloo" (Napoleon LOST at Waterloo → CONTRADICTION)
- **Impossibility**: Character meets someone who doesn't exist yet, or is in two places simultaneously
- **Canonical Event Insertion**: Adding new arrests, trials, meetings with major characters not mentioned in canon
- **Timeline Breaks**: Events dated before/after they canonically occurred
- **Relationship Fabrication**: Creating meetings between characters who never met (e.g., "Noirtier met Count of Monte Cristo" when Noirtier is paralyzed before Monte Cristo's revenge begins)

🔍 **NOTE ON KNOWLEDGE GRAPH FLOW**:
The Knowledge Graph contains sequential relations like `FOLLOWED_BY`, `NEXT_LOCATED_AT`, and `NEXT_ACTION_IS`. These represent the canonical chronological flow of the novel. Use these to verify if the backstory honors the order of events (e.g., if backstory says Character went A -> C -> B, but graph shows A -> B -> C, it's a CONTRADICTION).

🔍 ACTION: Search the graph for ANY fact or sequential flow that makes the backstory claim impossible.
- If found → STOP. Label = 0 (CONTRADICTORY)
- If not found → Continue to Lens 2

---

**LENS 2: NARRATIVE PLAUSIBILITY (Medium Check)**
Does the backstory respect the novel's established narrative constraints?

⚠️ CONTRADICTION PATTERNS:
- **Character Agency Violation**: Backstory gives a minor character major plot influence (e.g., "Paganel declined to chart Britannia's maiden voyage" implies he had that authority)
- **Tone Mismatch**: Adding supernatural/mystical elements to realistic novels (e.g., "breathed in mother's spirit through bones")
- **Causal Chain Break**: If backstory event X happened, does later canonical event Y still make logical sense?
  - Example: "Ayrton locked Captain Grant in a keel-less lifeboat" → But canon shows Captain Grant was shipwrecked differently
- **Completeness Violation**: Canon describes ALL of something (e.g., all known imprisonments, all children), and backstory adds a new instance

🔍 ACTION: Ask yourself:
1. Does the novel's plot REQUIRE that this backstory detail did NOT happen?
2. Would accepting this backstory create a paradox with later events?
3. Is the backstory too specific about events the novel treats as unknown/mysterious?

- If YES to any → Label = 0 (CONTRADICTORY)
- If NO to all → Continue to Lens 3

---

**LENS 3: ADDITIONAL INFORMATION (Easiest Check)**
Is the backstory simply adding new, non-conflicting details?

✅ CONSISTENT PATTERNS:
- Expands on vague canon (e.g., "he had a difficult childhood" → "his mother died when he was young")
- Adds internal motivations (e.g., "political arguments with father" when canon confirms political opposition)
- Fills narrative gaps without breaking established facts
- Provides plausible explanations for canonical traits

🔍 ACTION: If the backstory:
- Does NOT contradict Lens 1 (facts)
- Does NOT break Lens 2 (narrative logic)
- Simply enriches the character's history

→ Label = 1 (CONSISTENT)

---

**ADVERSARIAL CHECK (Apply to ALL backstories)**

Before finalizing your decision, play devil's advocate:

**If you're leaning CONSISTENT:**
- Could this backstory break any timeline we haven't checked?
- Does it create impossible coincidences (e.g., two unrelated characters meeting before the main plot)?
- Does it give the character knowledge/skills they shouldn't have?

**If you're leaning CONTRADICTORY:**
- Are we 100% sure the canon explicitly contradicts this, or are we just guessing?
- Could the novel's vagueness actually accommodate this backstory?
- Is there any reading of the canon that makes this plausible?

---

**WORKED EXAMPLES (Study These Patterns)**

**Example 1: CONTRADICTION via Historical Falsification**
Backstory: "Napoleon's triumph at Waterloo ended his hopes; in 1815 he withdrew from active plotting."
Analysis:
- Lens 1: Napoleon LOST at Waterloo (1815) → Historical fact contradicted
- Decision: Label = 0
Reasoning: "The backstory contains a factual error (Napoleon's 'triumph' at Waterloo), which he lost. This makes the entire causal claim false."

**Example 2: CONTRADICTION via Impossible Meeting**
Backstory: "Through underground circles he met the Count of Monte Cristo and fed the avenger vital information."
Canon: Noirtier is paralyzed from ~1829; Monte Cristo's revenge occurs 1838-1844
Analysis:
- Lens 1: Timeline impossible—Noirtier cannot speak/move during Monte Cristo's active period
- Lens 2: Narrative constraint—novel never mentions this connection
Decision: Label = 0
Reasoning: "Noirtier's paralysis (canonical) prevents him from meeting Monte Cristo in 'underground circles.' The novel establishes no such relationship."

**Example 3: CONTRADICTION via Canonical Event Addition**
Backstory: "He declined to take part in charting the maiden voyage of the Britannia; when news of her wreck arrived he filled three diary pages..."
Canon: Paganel is a geographer, not a ship's navigator; novel never mentions he was offered this role
Analysis:
- Lens 2: Adds a major decision (declining an expedition) not hinted at in canon
- Backstory implies causal guilt that contradicts Paganel's canonical character arc
Decision: Label = 0
Reasoning: "The novel never suggests Paganel had authority over the Britannia expedition. This backstory fabricates a decision point and guilt that contradict his established role as a bumbling, well-meaning scholar."

**Example 4: CONSISTENT via Plausible Expansion**
Backstory: "Villefort's drift toward the royalists disappointed him; father and son argued politics at every family gathering."
Canon: Noirtier = Bonapartist, Villefort = Royalist (explicitly stated); relationship is strained
Analysis:
- Lens 1: No factual contradiction
- Lens 2: Canon confirms political opposition and strained relationship
- Lens 3: "Argued at every gathering" is a plausible elaboration
Decision: Label = 1
Reasoning: "The novel establishes Noirtier and Villefort's opposing politics and tense relationship. Frequent political arguments are a natural, non-contradictory elaboration of this dynamic."

**Example 5: CONSISTENT via Gap-Filling**
Backstory: "His parents were targeted in a reprisal for supporting the Revolution; his mother was killed, deepening his distrust of authority."
Canon: Noirtier was a revolutionary; novel never mentions his parents
Analysis:
- Lens 1: No contradiction (parents never discussed in canon)
- Lens 2: Adds motivation that aligns with his revolutionary character
- Lens 3: Fills a gap without breaking narrative logic
Decision: Label = 1
Reasoning: "Canon never describes Noirtier's parents. This backstory provides a plausible origin for his revolutionary fervor without contradicting any established facts."

---

**OUTPUT FORMAT (STRICT JSON)**

{{
  "label": 1 or 0,
  "reasoning": "Step-by-step analysis through Lens 1 → Lens 2 → Lens 3, citing specific canonical facts or narrative constraints violated/respected. Reference the worked examples' reasoning style.",
  "evidence_queries": [
    "Query 1: [Specific canonical fact to verify, e.g., 'Verify Napoleon's fate at Waterloo 1815']",
    "Query 2: [Timeline check, e.g., 'When was Noirtier paralyzed vs. when did Monte Cristo appear']",
    "Query 3: [Causal check, e.g., 'Does novel mention Paganel's role in Britannia expedition planning']"
  ]
}}

**REMEMBER**: 
- Default to CONTRADICTORY (0) when in doubt about major claims
- Default to CONSISTENT (1) only for minor elaborations on confirmed canon
- ALWAYS cite specific canonical facts in your reasoning
"""