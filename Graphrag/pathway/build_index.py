# build_index.py
# Usage:
#  - Ensure Qdrant is running (default http://localhost:6333). Optionally set QDRANT_URL and QDRANT_API_KEY env vars.
#  - python build_index.py --path ./data/my_novel.txt --name my_novel
#
# This script:
#  - reads the file using Pathway FS connector (fallback to plain open)
#  - splits text with Pathway TokenCountSplitter
#  - computes embeddings with Pathway SentenceTransformerEmbedder (HuggingFace sentence-transformers)
#  - creates / upserts points in Qdrant collection named "{name}_collection"

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter

# Configuration defaults
DEFAULT_EMBEDDING_MODEL = "avsolatorio/GIST-small-Embedding-v0"  # compact model; change if desired
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
DEFAULT_BATCH = 128


def read_text_via_pathway(path: str) -> str:
    """
    Try to read file using Pathway fs connector (format plaintext_by_file).
    Fall back to plain Python file read if necessary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    try:
        # Use plaintext format instead of plaintext_by_file to get actual content
        table = pw.io.fs.read(str(path), format="plaintext", mode="static")
        df = pw.debug.table_to_pandas(table)
        
        # Look for the actual text content column
        if "data" in df.columns:
            content = "\n".join(df["data"].astype(str).tolist())
            if len(content) > 500:  # Sanity check - should be much larger for a novel
                return content
        
        # If that doesn't work, try other column names
        for col in df.columns:
            if col not in ["path", "modified_at", "created_at", "owner", "size", "seen_at"]:
                content = "\n".join(df[col].astype(str).tolist())
                if len(content) > 500:
                    return content
    except Exception as e:
        print(f"Pathway reading failed: {e}, falling back to plain Python")

    # Fallback plain python read (this should work)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    print(f"Fallback read successful: {len(content)} characters")
    return content


def split_text_tokencount(text: str, min_tokens: int = 100, max_tokens: int = 400, encoding_name: str = "cl100k_base") -> List[Tuple[str, dict]]:
    """
    Use Pathway TokenCountSplitter to split text into chunks.
    Returns a list of tuples (chunk_text, metadata_dict).
    We call the underlying __wrapped__ method directly.
    """
    splitter = TokenCountSplitter(min_tokens=min_tokens, max_tokens=max_tokens, encoding_name=encoding_name)
    # TokenCountSplitter.__wrapped__ expects a single string and returns a list[tuple[str, dict]]
    chunks = splitter.__wrapped__(text)
    # Ensure we return plain python types
    return [(str(t[0]), dict(t[1] if len(t) > 1 and t[1] is not None else {})) for t in chunks]


def embed_texts(texts: List[str], model_name: str, batch_size: int = DEFAULT_BATCH) -> np.ndarray:
    """
    Compute embeddings using Pathway SentenceTransformerEmbedder.__wrapped__()
    Returns array shape (n, dim) dtype=float32
    """
    embedder = SentenceTransformerEmbedder(model=model_name, call_kwargs={"show_progress_bar": False})
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = embedder.__wrapped__(batch)
        for e in batch_embs:
            embs.append(np.asarray(e, dtype=np.float32))
    if len(embs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(embs)


def create_or_update_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create collection if not exists. Use cosine distance.
    """
    existing = client.get_collections().collections
    if any(c.name == collection_name for c in existing):
        # optional: verify vector_size compatibility omitted
        return
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
    )


def upsert_to_qdrant(client: QdrantClient, collection_name: str, embeddings: np.ndarray, chunks: List[Tuple[str, dict]], name: str, batch_size: int = 500):
    """
    Upsert vectors into Qdrant. Payload includes id, text, metadata, chunk_index.
    Point ids are numeric (we use sequential ints).
    """
    n = embeddings.shape[0]
    assert n == len(chunks), "embeddings / chunks length mismatch"

    for i in range(0, n, batch_size):
        end = min(n, i + batch_size)
        batch_emb = embeddings[i:end].astype(float).tolist()
        points = []
        for j, emb in enumerate(batch_emb, start=i):
            chunk_text, metadata = chunks[j]
            payload = {"id": f"{name}__chunk__{j}", "text": chunk_text, "metadata": metadata, "chunk_index": j}
            points.append(qdrant_models.PointStruct(id=j, vector=emb, payload=payload))
        client.upsert(collection_name=collection_name, points=points)


def build_index(novel_path: str, name: str, embedding_model: str, qdrant_url: str, qdrant_api_key: str | None):
    print(f"Reading novel: {novel_path}")
    text = read_text_via_pathway(novel_path)
    print(f"Read {len(text)} characters.")

    # split into chunks
    print("Splitting text with TokenCountSplitter (token-aware)...")
    chunks = split_text_tokencount(text, min_tokens=100, max_tokens=400, encoding_name="cl100k_base")
    print(f"Got {len(chunks)} chunks.")

    # get chunk texts for embedding
    texts = [c[0] for c in chunks]

    # embed
    print(f"Computing embeddings with model '{embedding_model}' (this may download the model)...")
    embeddings = embed_texts(texts, model_name=embedding_model)
    print("Embeddings shape:", embeddings.shape)

    # connect to qdrant
    host = qdrant_url
    api_key = qdrant_api_key if qdrant_api_key else None
    client = QdrantClient(url=host, api_key=api_key)

    collection_name = f"{name}_collection"
    create_or_update_qdrant_collection(client, collection_name, vector_size=embeddings.shape[1])

    # upsert into qdrant
    print(f"Upserting {embeddings.shape[0]} vectors into Qdrant collection '{collection_name}'...")
    upsert_to_qdrant(client, collection_name, embeddings, chunks, name)
    print("Indexing done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to novel file (txt)")
    parser.add_argument("--name", required=True, help="Name for the index / novel (used in IDs & collection)")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--qdrant_url", default=DEFAULT_QDRANT_URL, help="Qdrant URL (e.g. http://localhost:6333)")
    parser.add_argument("--qdrant_api_key", default=DEFAULT_QDRANT_API_KEY, help="Qdrant API key (optional)")
    args = parser.parse_args()

    build_index(args.path, args.name, args.model, args.qdrant_url, args.qdrant_api_key)