# retriever.py
# Usage:
#   from retriever import retrieve_topk, list_indices
#   hits = retrieve_topk("my_novel", "Where did the protagonist go?", k=5)
#
# The retriever uses the same SentenceTransformer model as the index. It queries Qdrant and
# returns a list of dictionaries with {id, text, score, chunk_index, metadata}.

from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import configuration
from Graphrag.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    QDRANT_URL,
    QDRANT_API_KEY,
    RETRIEVAL_TOP_K,
    RETRIEVAL_INITIAL_K,
)

# in-memory cache for clients / embedders
_cache = {}


def _get_qdrant_client(url: str = QDRANT_URL, api_key: str | None = QDRANT_API_KEY) -> QdrantClient:
    key = f"qdrant::{url}::{api_key}"
    if key in _cache:
        return _cache[key]
    client = QdrantClient(url=url, api_key=api_key)
    _cache[key] = client
    return client


def _get_embedder(model: str = EMBEDDING_MODEL):
    if model in _cache:
        return _cache[model]
    embedder = SentenceTransformer(model)
    _cache[model] = embedder
    return embedder


def _get_reranker(model: str = RERANKER_MODEL):
    if model in _cache:
        return _cache[model]
    reranker = CrossEncoder(model)
    _cache[model] = reranker
    return reranker


def list_collections(qdrant_url: str = QDRANT_URL, qdrant_api_key: str | None = QDRANT_API_KEY) -> List[str]:
    client = _get_qdrant_client(qdrant_url, qdrant_api_key)
    cols = client.get_collections().collections
    return [c.name for c in cols]


def retrieve_topk(name: str, query: str, k: int = RETRIEVAL_TOP_K, 
                  model: str = EMBEDDING_MODEL, 
                  reranker_model: str = RERANKER_MODEL,
                  qdrant_url: str = QDRANT_URL, 
                  qdrant_api_key: str | None = QDRANT_API_KEY) -> List[Dict[str, Any]]:
    """
    Query the Qdrant collection for top-k chunks.
    1. Retrieve initial candidates using vector search.
    2. Rerank using CrossEncoder.
    3. Return top-k.
    """
    # Normalize book name to lowercase for consistent collection naming
    name = name.lower()
    collection = f"{name}_collection"
    client = _get_qdrant_client(qdrant_url, qdrant_api_key)

    # embed query
    embedder = _get_embedder(model)
    q_emb = embedder.encode(query, convert_to_numpy=True).astype(float).tolist()

    # 1. Retrieve initial candidates
    initial_k = RETRIEVAL_INITIAL_K
    search_result = client.query_points(
        collection_name=collection, 
        query=q_emb, 
        limit=initial_k, 
        with_payload=True
    ).points

    if not search_result:
        return []

    # Prepare for reranking
    # CrossEncoder input: list of pairs (query, doc_text)
    hits = []
    
    # Store temporary mapping to preserve metadata after reranking
    # We'll use index in search_result to map back
    candidates = []
    
    for hit in search_result:
        payload = hit.payload or {}
        text = payload.get("text", "")
        candidates.append([query, text])
    
    # 2. Rerank
    reranker = _get_reranker(reranker_model)
    scores = reranker.predict(candidates)
    
    # Combine results
    reranked_results = []
    for i, score in enumerate(scores):
        hit = search_result[i]
        payload = hit.payload or {}
        reranked_results.append({
            "id": payload.get("id", str(hit.id)),
            "text": payload.get("text", ""),
            "score": float(score), # New reranking score
            "vector_score": float(hit.score) if hit.score is not None else 0.0, # Original vector score
            "chunk_index": payload.get("chunk_index", None),
            "metadata": payload.get("metadata", None),
        })
    
    # Sort by new score descending
    reranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 3. Return top k
    return reranked_results[:k]