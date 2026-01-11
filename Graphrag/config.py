"""Centralized configuration for Graph RAG system."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================================
# Retrieval Parameters
# ============================================================================
RETRIEVAL_TOP_K = 8  # Final number of chunks to return
RETRIEVAL_INITIAL_K = 50  # Initial candidates for reranking

# ============================================================================
# Chunking Parameters
# ============================================================================
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# ============================================================================
# Embedding Models
# ============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================================
# Qdrant Connection
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# ============================================================================
# Batch Processing
# ============================================================================
DEFAULT_BATCH_SIZE = 64  # For embedding generation
UPSERT_BATCH_SIZE = 500  # For Qdrant upsert operations
