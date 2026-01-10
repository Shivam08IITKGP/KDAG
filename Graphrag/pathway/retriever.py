# retriever.py
# Usage:
#   from retriever import retrieve_topk, list_indices
#   hits = retrieve_topk("my_novel", "Where did the protagonist go?", k=5)
#
# The retriever uses the same SentenceTransformer model as the index. It queries Qdrant and
# returns a list of dictionaries with {id, text, score, chunk_index, metadata}.

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder

INDEX_ROOT = Path("indexes")  # not used for qdrant, kept for compatibility
DEFAULT_EMBEDDING_MODEL = "avsolatorio/GIST-small-Embedding-v0"
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# in-memory cache for clients / embedders
_cache = {}


def _get_qdrant_client(url: str = DEFAULT_QDRANT_URL, api_key: str | None = DEFAULT_QDRANT_API_KEY) -> QdrantClient:
    key = f"qdrant::{url}::{api_key}"
    if key in _cache:
        return _cache[key]
    client = QdrantClient(url=url, api_key=api_key)
    _cache[key] = client
    return client


def _get_embedder(model: str = DEFAULT_EMBEDDING_MODEL):
    if model in _cache:
        return _cache[model]
    embedder = SentenceTransformerEmbedder(model=model, call_kwargs={"show_progress_bar": False})
    _cache[model] = embedder
    return embedder


def list_collections(qdrant_url: str = DEFAULT_QDRANT_URL, qdrant_api_key: str | None = DEFAULT_QDRANT_API_KEY) -> List[str]:
    client = _get_qdrant_client(qdrant_url, qdrant_api_key)
    cols = client.get_collections().collections
    return [c.name for c in cols]


# ...existing code...

def retrieve_topk(name: str, query: str, k: int = 5, model: str = DEFAULT_EMBEDDING_MODEL, qdrant_url: str = DEFAULT_QDRANT_URL, qdrant_api_key: str | None = DEFAULT_QDRANT_API_KEY) -> List[Dict[str, Any]]:
    """
    Query the Qdrant collection for top-k chunks.
    name: the novel name used at index time (collection is '{name}_collection')
    Returns a list of dicts: {id, text, score, chunk_index, metadata}
    """
    collection = f"{name}_collection"
    client = _get_qdrant_client(qdrant_url, qdrant_api_key)

    # embed query
    embedder = _get_embedder(model)
    q_emb = embedder.__wrapped__([query])[0].astype(float).tolist()

    # perform search - UPDATED METHOD CALL
    search_result = client.query_points(
        collection_name=collection, 
        query=q_emb, 
        limit=k, 
        with_payload=True
    ).points

    hits = []
    for hit in search_result:
        payload = hit.payload or {}
        hits.append(
            {
                "id": payload.get("id", str(hit.id)),
                "text": payload.get("text", ""),
                "score": float(hit.score) if hit.score is not None else 0.0,
                "chunk_index": payload.get("chunk_index", None),
                "metadata": payload.get("metadata", None),
            }
        )
    return hits