EXTRACTION_PROMPT = """You are an expert evidence retrieval specialist. Your task is to generate targeted search queries that will uncover ALL canonical information needed to verify or refute a character backstory.

---CANONICAL CONTEXT---
Book: {book_name}
Character: {character_name}

Character Summary (What we know from the novel):
{character_summary}

Backstory to Verify:
{backstory}

---QUERY GENERATION STRATEGY---

You must generate {max_queries} queries following this prioritized framework:

**TIER 1: CLAIM-SPECIFIC VERIFICATION (Highest Priority)**
For EACH specific claim in the backstory, generate a query that tests it.

Examples:
- Backstory says: "Napoleon's triumph at Waterloo" 
  → Query: "Napoleon Waterloo 1815 battle outcome victory defeat"
  
- Backstory says: "met the Count of Monte Cristo in underground circles"
  → Query: "{character_name} met Count Monte Cristo when where"
  
- Backstory says: "declined to chart Britannia maiden voyage"
  → Query: "{character_name} Britannia expedition role offered declined"

**TIER 2: TEMPORAL VERIFICATION**
Generate queries that establish the character's timeline to check for impossibilities.

Query Templates:
- "{character_name} timeline birth death major life events"
- "{character_name} whereabouts location [YEAR from backstory]"
- "{character_name} capabilities abilities [YEAR from backstory]"
- "when did {character_name} become paralyzed imprisoned exiled" (if relevant)

**TIER 3: RELATIONSHIP VERIFICATION**
For any person/organization mentioned in the backstory, verify the relationship.

Query Templates:
- "{character_name} relationship with [PERSON mentioned in backstory]"
- "{character_name} met [PERSON] when where circumstances"
- "{character_name} connection to [ORGANIZATION/EVENT mentioned]"

**TIER 4: CONTRADICTORY EVIDENCE SEARCH**
Generate queries that would DISPROVE the backstory if found.

Examples:
- If backstory says "helped Napoleon escape"
  → Query: "Napoleon escape attempts who helped canonical account"
  
- If backstory says "never met his father"
  → Query: "{character_name} father relationship meetings interactions"

**TIER 5: COMPLETENESS CHECK**
Search for canonical descriptions that might list ALL instances of something.

Query Templates:
- "{character_name} all arrests imprisonments trials" (to check if backstory adds unlisted ones)
- "{character_name} all children family members" (to check if backstory invents relatives)
- "{character_name} complete political history affiliations" (to check for unlisted events)

---CRITICAL RULES---

1. **BE SPECIFIC**: Extract exact names, dates, locations, events from the backstory
   - BAD: "{character_name} political activities"
   - GOOD: "{character_name} political activities 1815 after Waterloo Napoleon"

2. **NO PRESUPPOSITION**: Never assume backstory claims are true
   - BAD: "Where did {character_name} burn the betrothal contract" (assumes it happened)
   - GOOD: "{character_name} betrothal contract Saint-Méran burned destroyed"

3. **SEARCH FOR NEGATION**: Look for evidence of what DIDN'T happen
   - Include keywords: "never", "not", "unable", "impossible", "prevented"
   - Example: "{character_name} never met unable to meet [PERSON]"

4. **TEMPORAL PRECISION**: Include years, periods, sequences
   - Extract dates from backstory: "in 1804", "during Revolution", "after Waterloo"
   - Add temporal keywords: "before", "after", "during", "when", "timeline"

5. **CAUSAL CHAIN SEARCH**: For backstory events, search for their canonical causes/effects
   - Backstory: "fled to Australia after mutiny"
   - Query: "{character_name} arrived Australia when how mutiny Britannia"

6. **FACTUAL CROSS-CHECK**: For historical events mentioned, verify basic facts
   - Backstory mentions "Waterloo" → Query: "Napoleon Waterloo 1815 outcome"
   - Backstory mentions "French Revolution trials" → Query: "Louis XVI trial 1793 participants"

---OUTPUT INSTRUCTIONS---
Generate exactly {max_queries} queries. Each query should be 5-15 words, optimized for semantic search.
Prioritize Tier 1 (claim-specific) queries first.
Make each query unique and non-redundant.
Focus on claims that could make the backstory CONTRADICTORY if proven false.
Include the character name in 60% of queries, omit in 40% (to catch general event descriptions).
"""