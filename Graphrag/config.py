"""
Configuration file for the Narrative Auditor (GraphRAG-based contradiction detector)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ========== API CONFIGURATION ==========
# OpenRouter API (Free tier with community models)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model selection (switch between these as needed)
# Free models: mistral-7b, neural-chat-7b, zephyr-7b
LLM_MODEL = "mistralai/mistral-7b-instruct"  # Free model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")  # Falls back to local if needed

# ========== NEO4J CONFIGURATION ==========
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ========== TEXT PROCESSING CONFIGURATION ==========
CHUNK_SIZE = 1500  # tokens per chunk
CHUNK_OVERLAP = 200  # token overlap for context
MIN_SCENE_LENGTH = 500  # minimum tokens in a scene

# ========== ENTITY TYPES FOR EXTRACTION ==========
ENTITY_TYPES = [
    "Character",      # Main/secondary characters
    "Location",       # Places where events occur
    "Event",          # Key narrative events
    "Object",         # Important items/artifacts
    "Organization",   # Groups, institutions
    "Relationship",   # Character relationships
]

# ========== CHARACTER STATE TYPES ==========
CHARACTER_STATE_TYPES = [
    "Location",           # Where they are
    "Physical_State",     # Injured, healthy, appearance changes
    "Relationship",       # Bonds with other characters
    "Social_Status",      # Rich, poor, reputation
    "Goal",               # What they're trying to achieve
    "Skill",              # Abilities they possess
    "Possession",         # What they own/carry
    "Emotional_State",    # Happy, angry, confused, etc.
    "Occupation",         # Job/role in society
    "Age_Group",          # Approximate age
]

# ========== CONTRADICTION DETECTION TYPES ==========
CONTRADICTION_TYPES = [
    "temporal",           # Event sequence violations
    "spatial",            # Location impossibilities
    "causal",             # Cause-effect violations
    "physical",           # Physical law violations
    "logical",            # Logical inconsistencies
    "character_state",    # State changes that contradict earlier states
]

# ========== GRAPH CONFIGURATION ==========
BATCH_SIZE = 10  # Nodes to process before committing to Neo4j
ENABLE_AUTO_DELETE_EXISTING = False  # Set True to clear graph on startup
MAX_RELATIONSHIPS_PER_ENTITY = 15  # Limit relationship extraction

# ========== FILE PATHS ==========
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
NOVEL_FILE = os.path.join(INPUT_DIR, "novel.txt")

# ========== LOGGING ==========
LOG_LEVEL = "INFO"
VERBOSE = True

# ========== PROMPTS ==========

SYSTEM_PROMPT_ENTITY_EXTRACTION = """You are an expert literary analyst specializing in narrative structure and character analysis.

Extract structured information from the given narrative text chunk. Return ONLY valid JSON.

Focus on:
1. Characters (with their current state/actions in this chunk)
2. Locations where scenes occur
3. Key events that happen
4. Character relationships and interactions
5. Character state changes (emotional, physical, social)

Return JSON with this exact structure:
{
  "characters": [
    {
      "name": "Character Name",
      "character_type": "Protagonist|Antagonist|Supporting|Minor",
      "description": "Brief description",
      "states": [
        {
          "type": "Physical_State|Location|Emotional_State|Social_Status|Goal|Skill|Possession|Relationship|Occupation|Age_Group",
          "value": "description of state",
          "confidence": 0.85
        }
      ],
      "interactions": ["Character A", "Character B"],
      "actions": "What the character did in this chunk"
    }
  ],
  "locations": [
    {
      "name": "Location Name",
      "location_type": "City|Building|Region|Natural|Other",
      "description": "Brief description and significance"
    }
  ],
  "events": [
    {
      "name": "Event Name",
      "description": "What happened",
      "participants": ["Character A", "Character B"],
      "location": "Where it happened",
      "timestamp_hint": "Approximate when",
      "event_type": "Action|Dialogue|Discovery|Conflict|Resolution|Other"
    }
  ],
  "relationships": [
    {
      "source_character": "Character A",
      "target_character": "Character B",
      "relationship_type": "Family|Romance|Friendship|Enmity|Professional|Mentor|Other",
      "description": "Nature of relationship",
      "sentiment": "Positive|Negative|Neutral|Ambiguous"
    }
  ]
}

Extract ONLY information explicitly stated or clearly implied in the text. Be accurate."""

SYSTEM_PROMPT_CONTRADICTION_DETECTION = """You are a narrative consistency expert. Your job is to detect contradictions in backstory claims against novel evidence.

For each BACKSTORY CLAIM provided, analyze the EVIDENCE from the novel text.

Determine if there is ANY contradiction:
1. TEMPORAL: Did claim say event X happened at time T, but evidence shows character was elsewhere?
2. SPATIAL: Did claim say character was in location A, but evidence shows they were in location B?
3. PHYSICAL: Does the claim contradict physical facts (e.g., broken arm but later fighting)?
4. CAUSAL: Does the claim contradict cause-effect shown in evidence?
5. CHARACTER_STATE: Does the claim contradict established traits/states?
6. LOGICAL: Is there a logical impossibility?

Return ONLY valid JSON:
{
  "claim": "Original backstory claim text",
  "is_contradicted": true|false,
  "contradiction_type": "temporal|spatial|physical|causal|character_state|logical|none",
  "confidence": 0.95,
  "reasoning": "Detailed explanation of contradiction or consistency",
  "evidence_snippets": ["Direct quote or paraphrase from novel"],
  "severity": "critical|moderate|minor|none",
  "chapter_references": "Ch. X, Scene Y",
  "recommendation": "Action to investigate or text to review"
}

Only mark as contradicted if you find CLEAR EVIDENCE of impossibility.
Be conservative: if something could be plausible, mark as consistent."""

SYSTEM_PROMPT_CLAIM_EXTRACTION = """You are an expert at breaking down complex backstories into atomic, verifiable claims.

Given a character backstory, extract EVERY claim that can be verified against narrative evidence.

Break down compound claims into simple ones. For example:
- "John broke his arm in 2020 and couldn't play cricket" becomes:
  - "John had a broken arm"
  - "The broken arm occurred in 2020"
  - "John couldn't play cricket due to the injury"

Return ONLY valid JSON:
{
  "character": "Character Name",
  "claims": [
    {
      "claim_id": "CLAIM_001",
      "claim_text": "Simple, atomic claim",
      "claim_type": "Physical_State|Temporal|Relationship|Event|Location|Skill|Possession|Emotional|Other",
      "entities_involved": ["Character A", "Character B"],
      "related_events": ["Event Name"],
      "timeframe": "Specific year/period if mentioned, or 'unspecified'",
      "importance": "Critical|Important|Minor",
      "source_text": "Direct quote from backstory"
    }
  ],
  "total_claims": 5,
  "character_summary": "Brief overview of backstory"
}

Extract ALL verifiable claims, even seemingly minor ones."""

# ========== EMBEDDING CONFIGURATION ==========
EMBEDDING_DIM = 1536  # OpenAI embedding dimension
LOCAL_EMBEDDING_DIM = 384  # For local embeddings if needed

# ========== COMMUNITY DETECTION ==========
LEIDEN_RESOLUTION = 1.0  # Community detection granularity
MIN_COMMUNITY_SIZE = 3  # Minimum nodes to form a community
