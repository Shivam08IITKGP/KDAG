"""Enhanced prompts for graph creator agent with improved extraction logic."""

TRIPLET_EXTRACTION_PROMPT = """You are extracting structured knowledge triplets to validate a character's backstory against canonical novel text.

**TARGET CHARACTER: {character_name}**
**BACKSTORY CLAIM TO VALIDATE:**
{backstory}

**Evidence Chunks from Novel:**
{evidence_text}

---MISSION---
1. NARRATIVE SUMMARY: Write a 5-7 sentence narrative summary of {character_name} BASED ONLY on the evidence provided. This summary should capture their origins, roles, and key life events in a chronological flow.
2. KNOWLEDGE GRAPH: Build a structured knowledge graph (triplets) that captures ALL facts from the evidence to confirm or contradict the backstory.
Focus on: origins, family, formative experiences, relationships, skills, beliefs, key events, timeline markers.

---CRITICAL EXTRACTION RULES---

1. **SUBJECT NORMALIZATION (ABSOLUTE REQUIREMENT):**
   - Subject MUST ALWAYS be the full canonical name: "{character_name}"
   - NEVER use: "he", "she", "they", "the character", "him", "her", partial names
   - When evidence uses pronouns/partial names, replace with "{character_name}"
   - Example: Evidence says "Paganel was absent-minded" → Subject: "Jacques Paganel"

2. **RELATION TYPES - USE THESE CATEGORIES:**
   
   **BIOGRAPHICAL:**
   - BORN_IN, BORN_ON, DIED_IN, AGED, NATIONALITY_IS
   - CHILDHOOD_IN, RAISED_BY, ORPHANED_AT, GREW_UP_IN
   
   **FAMILY & RELATIONSHIPS:**
   - FATHER_IS, MOTHER_IS, SIBLING_OF, MARRIED_TO, CHILD_OF
   - FRIEND_OF, ENEMY_OF, ALLY_OF, MENTOR_OF, STUDENT_OF
   - TRAVELED_WITH, WORKED_WITH, BETRAYED_BY, RESCUED_BY
   
   **TRAITS & CHARACTERISTICS:**
   - HAS_TRAIT (for personality: "absent-minded", "brave", "cunning")
   - PHYSICAL_TRAIT (for appearance: "scarred left arm", "tall build")
   - KNOWN_FOR (reputation/defining feature)
   - SPEAKS_LANGUAGE, FLUENT_IN
   
   **ACTIONS & EVENTS:**
   - PARTICIPATED_IN, WITNESSED, SURVIVED, ESCAPED_FROM
   - DISCOVERED, CREATED, DESTROYED, STOLE, RESCUED
   - KILLED, MURDERED_BY, FOUGHT_AT, DEFEATED
   - MISTAKENLY_DID (for errors/accidents)
   
   **AFFILIATIONS & ROLES:**
   - MEMBER_OF, SECRETARY_OF, CAPTAIN_OF, LEADER_OF
   - EMPLOYED_BY, SERVED_UNDER, WORKS_FOR
   - EXILED_FROM, EXPELLED_FROM, IMPRISONED_IN
   
   **KNOWLEDGE & SKILLS:**
   - EXPERT_IN, STUDIED, KNOWS_ABOUT, TRAINED_IN
   - AUTHORED, PUBLISHED, RESEARCHED
   
   **POSSESSIONS & LOCATIONS:**
   - OWNS, CARRIES, WEARS, POSSESSES
   - LIVES_IN, RESIDES_AT, DEPARTED_FROM, ARRIVED_AT
   - STATIONED_AT, BASED_IN, TRAVELED_TO
   
   **TEMPORAL & CAUSAL:**
   - OCCURRED_DURING, HAPPENED_BEFORE, CAUSED, RESULTED_IN
   - MOTIVATED_BY, DRIVEN_BY, FEARED, HOPED_FOR
   - FOLLOWED_BY, NEXT_LOCATED_AT, NEXT_ACTION_IS (for chronological flow)

3. **NARRATIVE ADJACENCY & FLOW (CRITICAL):**
   - The `evidence_id` contains a numeric chunk index (e.g., `...__chunk__42`). These indices represent the sequential flow of the novel.
   - You MUST identify chronological progressions (journeys, aging, series of tasks) across chunks.
   - If Chunk N mentions "Location A" and Chunk N+k mentions "Location B" as part of the same sequence, create triplets showing this flow.
   - Example: "{character_name}" --[FOLLOWED_BY]--> "Next Event" or "{character_name}" --[NEXT_LOCATED_AT]--> "New City".
   - Use these links to ensure the graph captures the *direction* and *flow* of the character's life, not just isolated facts.

4. **OBJECT SPECIFICITY RULES:**
   
   **DO extract:**
   - Specific names: "Lord Glenarvan", "Geographical Society of Paris"
   - Concrete dates/ages: "1815", "age 42", "during the expedition"
   - Precise locations: "Château d'If", "Paris", "New Zealand's North Island"
   - Specific traits: "absent-minded", "expert navigator", "feared authority"
   - Concrete events: "trial of Louis XVI", "mutiny on Britannia"
   
   **DON'T extract:**
   - Vague references: "his companions", "the crew", "those events"
   - Generic concepts: "challenges", "difficulties", "adventures"
   - Pronouns as objects: "them", "it", "that"
   - Uncertain info: "possibly", "might have", "seems to"

4. **BACKSTORY VALIDATION FOCUS:**
   
   Extract triplets that address these validation dimensions:
   
   a) **Timeline Consistency:**
      - Ages, dates, birth years, death dates
      - Sequence of events (what happened before/after)
      - Duration of experiences
   
   b) **Causal Coherence:**
      - Motivations → Actions
      - Events → Consequences
      - Experiences → Personality traits
   
   c) **Relationship Network:**
      - Family members (confirm/deny backstory claims)
      - Mentors, allies, enemies
      - Professional affiliations
   
   d) **Character Development:**
      - Formative experiences
      - Skills acquired (and when/how/where)
      - Belief changes over time
   
   e) **Narrative Constraints:**
      - Character's physical location at key times
      - Knowledge the character should/shouldn't have
      - Abilities demonstrated in the novel

5. **EXTRACTION STRATEGY:**
   
   For EACH evidence chunk:
   - Read for facts about {character_name}'s: past, family, skills, beliefs, key moments
   - Extract 3-7 triplets (favor completeness for backstory validation)
   - Prioritize facts that could contradict common backstory assumptions
   - Include both positive facts (X did Y) and negative facts (X never did Z, X wasn't at Y)
   
   **Special attention to:**
   - First mentions of character
   - Backstory exposition passages
   - Dialogue revealing past events
   - Descriptions of skills/knowledge
   - References to family, childhood, training

6. **EVIDENCE CITATION:**
   - Each triplet MUST cite the exact evidence_id where the fact appears
   - If one fact spans multiple chunks, create separate triplets with different evidence_ids
   - Never cite evidence_ids not provided in the input

7. **HANDLING IMPLICIT INFORMATION:**
   
   **DO infer when safe:**
   - Evidence: "Paganel, secretary of the Geographical Society" 
     → ("Jacques Paganel", "IS_SECRETARY_OF", "Geographical Society of Paris")
   - Evidence: "His father's death shaped his distrust of nobility"
     → ("Jacques Paganel", "FATHER_IS", "deceased")
     → ("Jacques Paganel", "MOTIVATED_BY", "father's death")
     → ("Jacques Paganel", "HAS_TRAIT", "distrusts nobility")
   
   **DON'T infer when uncertain:**
   - Evidence: "He met someone important in Paris"
     → Skip (who? when? too vague)

8. **QUALITY CHECKLIST - Before outputting each triplet, verify:**
   - [ ] Subject is EXACTLY "{character_name}" (full canonical name)
   - [ ] Relation is UPPER_SNAKE_CASE and from categories above
   - [ ] Object is concrete, specific, and adds validation value
   - [ ] Evidence_id is valid and the fact is actually in that chunk
   - [ ] Triplet helps confirm/deny backstory claims about origins, family, skills, or key events
   - [ ] No redundancy with other triplets from same evidence

---OUTPUT INSTRUCTIONS---
Generate a `graph_summary` (1-3 sentences) and a list of `triplets`.
Each triplet must include `subject`, `relation`, `object`, and `evidence_id`.
"""