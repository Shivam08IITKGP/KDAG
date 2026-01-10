"""Prompts for graph creator agent."""
TRIPLET_EXTRACTION_PROMPT = """Extract structured knowledge triplets from the following list of evidence texts to build a character knowledge graph.

Evidence List:
{evidence_text}

---GOAL---
Extract meaningful facts as (subject, relation, object) triplets from ALL the provided evidence items.
Each evidence item starts with "ID: <id>" followed by the text.
You MUST cite the correct Evidence ID for each extracted triplet.

---STRICT RULES---

1. **ENTITY NORMALIZATION (CRITICAL):**
   - **NO PRONOUNS:** Never use "He", "She", "They", "It", "The Character", or "This person"
   - **FULL NAMES ONLY:** Always use the complete proper name (e.g., "Jacques Paganel" not "Paganel" or "he")
   - **CONSISTENCY:** If evidence mentions "he" or a partial name, infer the full name from context
   - **NO GENERIC REFERENCES:** Avoid "the protagonist", "the geographer" as subjects

2. **RELATION QUALITY:**
   - Use **UPPER_SNAKE_CASE** format (e.g., "IS_MEMBER_OF", "TRAVELED_TO")
   - Be **specific and meaningful**: "MISTAKENLY_BOARDED" > "BOARDED", "SPEAKS_FLUENTLY" > "SPEAKS"
   - Common relation types for novels:
     * Character traits: HAS_TRAIT, KNOWN_FOR, PERSONALITY_IS
     * Relationships: FRIEND_OF, ENEMY_OF, FAMILY_OF, WORKS_WITH, TRAVELS_WITH
     * Actions: TRAVELED_TO, MISTAKENLY_DID, DISCOVERED, CREATED, DESTROYED
     * Affiliations: MEMBER_OF, WORKS_FOR, BELONGS_TO, REPRESENTS
     * Possessions: OWNS, CARRIES, WEARS
     * Locations: LIVES_IN, BORN_IN, LOCATED_AT, DEPARTED_FROM, ARRIVED_AT
     * Knowledge/Skills: SPEAKS, STUDIES, KNOWS_ABOUT, EXPERT_IN

3. **OBJECT SPECIFICITY:**
   - **CONCRETE ENTITIES:** Use specific names, places, or concepts
   - **NO VAGUE PHRASES:** 
     * BAD: "comical mistakes" -> GOOD: "frequent absent-mindedness"
     * BAD: "his friends" -> GOOD: "Lord Glenarvan" (or skip if names unknown)
     * BAD: "confuses languages" -> GOOD: Use relation "CONFUSED" object "Portuguese for Spanish"
   - **TRAITS AS ADJECTIVES:** For character traits, use clear adjective phrases: "absent-minded", "highly intelligent", "extremely brave"

4. **EXTRACTION GUIDELINES:**
   - Extract **2-5 triplets** per evidence chunk (quality over quantity)
   - Only extract facts **explicitly stated or strongly implied**
   - **Skip uncertain information** - if unsure about full name or specifics, omit that triplet
   - Each triplet should add **unique information** to the knowledge graph
   - Avoid redundant triplets (e.g., don't extract both "LOCATED_AT: Paris" and "LIVES_IN: Paris" from same evidence)

5. **NOVEL-SPECIFIC FOCUS:**
   - Prioritize: character relationships, personality traits, key actions, locations visited
   - Capture: motivations, conflicts, turning points, affiliations
   - Include: temporal context when available (e.g., "during the expedition")

---OUTPUT FORMAT---
Return **ONLY** valid JSON with a "triplets" key containing an array of objects.

Each triplet object must have exactly these fields:
- "subject": string (Full proper name, no pronouns)
- "relation": string (UPPER_SNAKE_CASE)
- "object": string (Specific entity/concept/trait)
- "evidence_id": string (The exact ID of the evidence where this fact was found)

Example:
{{
  "triplets": [
    {{"subject": "Jacques Paganel", "relation": "IS_SECRETARY_OF", "object": "Geographical Society of Paris", "evidence_id": "chunk_123"}},
    {{"subject": "Jacques Paganel", "relation": "HAS_TRAIT", "object": "absent-minded", "evidence_id": "chunk_124"}},
    {{"subject": "Jacques Paganel", "relation": "MISTAKENLY_BOARDED", "object": "Duncan", "evidence_id": "chunk_125"}}
  ]
}}

DO NOT include any markdown formatting, code blocks, or explanatory text. Return ONLY the JSON object.
"""
