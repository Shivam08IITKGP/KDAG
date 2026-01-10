# build_index.py
# Usage:
#  - Ensure Qdrant is running (default http://localhost:6333). Optionally set QDRANT_URL and QDRANT_API_KEY env vars.
#  - python build_index.py --path ./data/my_novel.txt --name my_novel
#
# This script:
#  - reads the file
#  - splits text with RecursiveCharacterTextSplitter
#  - computes embeddings with SentenceTransformer
#  - creates / upserts points in Qdrant collection named "{name}_collection"

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Import configuration
from Graphrag.config import (
    EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_BATCH_SIZE,
    UPSERT_BATCH_SIZE,
)


def read_text(path: str) -> str:
    """
    Read file content.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        print(f"Read successful: {len(content)} characters")
        return content
    except Exception as e:
        print(f"Reading failed: {e}")
        return ""


def split_text_recursive(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, dict]]:
    """
    Use RecursiveCharacterTextSplitter to split text into chunks.
    Returns a list of tuples (chunk_text, metadata_dict).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = splitter.split_text(text)
    # We don't have per-chunk metadata from the splitter itself, so we return empty dicts
    return [(t, {}) for t in texts]


def embed_texts(texts: List[str], model_name: str, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Compute embeddings using SentenceTransformer.
    Returns array shape (n, dim) dtype=float32
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def create_or_update_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """
    Create collection if not exists. Use cosine distance.
    """
    existing = client.get_collections().collections
    if any(c.name == collection_name for c in existing):
        return
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
    )


def upsert_to_qdrant(client: QdrantClient, collection_name: str, embeddings: np.ndarray, chunks: List[Tuple[str, dict]], name: str, batch_size: int = 500):
    """
    Upsert vectors into Qdrant. Payload includes id, text, metadata, chunk_index.
    Point ids are numeric (sequential ints starting from 1).
    """
    n = embeddings.shape[0]
    assert n == len(chunks), "embeddings / chunks length mismatch"

    for i in range(0, n, batch_size):
        end = min(n, i + batch_size)
        batch_emb = embeddings[i:end].astype(float).tolist()
        points = []
        for j, emb in enumerate(batch_emb, start=i):
            chunk_text, metadata = chunks[j]
            # ID is 1-based index: j + 1
            point_id = j + 1
            payload = {"id": f"{name}__chunk__{point_id}", "text": chunk_text, "metadata": metadata, "chunk_index": point_id}
            points.append(qdrant_models.PointStruct(id=point_id, vector=emb, payload=payload))
        client.upsert(collection_name=collection_name, points=points)


def build_index(novel_path: str, name: str, embedding_model: str | None = None, qdrant_url: str | None = None, qdrant_api_key: str | None = None):
    # Normalize book name to lowercase for consistent collection naming
    name = name.lower()
    
    # Use config defaults if not provided
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL
    if qdrant_url is None:
        qdrant_url = QDRANT_URL
    if qdrant_api_key is None:
        qdrant_api_key = QDRANT_API_KEY

    # Check if index already exists to avoid redundant chunking/embedding
    try:
        host = qdrant_url
        api_key = qdrant_api_key if qdrant_api_key else None
        client = QdrantClient(url=host, api_key=api_key)
        collection_name = f"{name}_collection"
        
        # We always rebuild for now to ensure new chunking/embedding strategy is applied
        # existing_collections = client.get_collections().collections
        # if any(c.name == collection_name for c in existing_collections):
        #      info = client.get_collection(collection_name)
        #      if info.points_count and info.points_count > 0:
        #          print(f"✅ Index '{collection_name}' already exists. Skipping rebuild.")
        #          return
        print(f"Target Collection: {collection_name}")
    except Exception as e:
        print(f"Warning: Could not check existing collections: {e}")

    print(f"Reading novel: {novel_path}")
    text = read_text(novel_path)
    print(f"Read {len(text)} characters.")

    # split into chunks
    print(f"Splitting text with RecursiveCharacterTextSplitter (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = split_text_recursive(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Got {len(chunks)} chunks.")

    # get chunk texts for embedding
    texts = [c[0] for c in chunks]

    # embed
    print(f"Computing embeddings with model '{embedding_model}'...")
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
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="SentenceTransformer model name")
    parser.add_argument("--qdrant_url", default=QDRANT_URL, help="Qdrant URL (e.g. http://localhost:6333)")
    parser.add_argument("--qdrant_api_key", default=QDRANT_API_KEY, help="Qdrant API key (optional)")
    args = parser.parse_args()

    build_index(args.path, args.name, args.model, args.qdrant_url, args.qdrant_api_key)