# test_pipeline.py
# Simple end-to-end test for the RAG pipeline using:
#  - build_index.py (Pathway TokenCountSplitter + SentenceTransformerEmbedder -> Qdrant)
#  - retriever.py (query embedding + Qdrant search)
#
# Usage:
#   python test_pipeline.py --novel ./data/my_novel.txt --name my_novel
#
# Requirements:
#   - Qdrant running and accessible (default http://localhost:6333)
#   - Packages: pathway[xpack-llm], sentence-transformers, qdrant-client
#     pip install "pathway[xpack-llm]" sentence-transformers qdrant-client
#
# The script will:
#   1) ensure Qdrant is reachable
#   2) build the index (upsert vectors into Qdrant) for the given novel name
#   3) run a few sample queries and print results
#   4) basic checks: each query returns >=1 hit and scores are numeric

import argparse
import sys
import time
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# import the functions from your scripts (assumes these files are in same folder)
from build_index import build_index
from retriever import retrieve_topk, list_collections

DEFAULT_QDRANT_URL = "http://localhost:6333"


def check_qdrant(url: str, api_key: str | None) -> QdrantClient:
    print(f"Checking Qdrant at {url} ...")
    try:
        client = QdrantClient(url=url, api_key=api_key)
        # simple call
        cols = client.get_collections()
        print("Qdrant reachable. Collections:", [c.name for c in cols.collections])
        return client
    except Exception as e:
        print("Failed to reach Qdrant at", url, "\nError:", e)
        raise


def run_test(novel_path: str, name: str, model: str, qdrant_url: str, qdrant_api_key: str | None, rebuild: bool, queries: List[str], topk: int):
    # 1) ensure Qdrant reachable
    client = check_qdrant(qdrant_url, qdrant_api_key)

    # 2) build index (optionally rebuild)
    if rebuild:
        print(f"\nBuilding index for '{name}' from file: {novel_path}")
        build_index(novel_path, name, embedding_model=model, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
        print("Index built. Wait briefly for Qdrant to commit...")
        time.sleep(1.0)  # small delay to ensure Qdrant processed upserts

    # verify collection exists
    collection = f"{name}_collection"
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        print(f"ERROR: expected Qdrant collection '{collection}' not found. Collections: {collections}")
        sys.exit(2)
    print(f"Collection '{collection}' present. Running retrieval tests...")

    # 3) run sample queries
    for q in queries:
        print("\n=== Query:", q)
        hits = retrieve_topk(name, q, k=topk, model=model, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
        print(f"Returned {len(hits)} hits (top {topk} requested).")
        if len(hits) == 0:
            print("WARNING: no hits returned for query. This may indicate indexing or model mismatch.")
        for i, h in enumerate(hits):
            print(f"#{i+1} id={h.get('id')} score={h.get('score'):.4f} chunk_index={h.get('chunk_index')}")
            text_preview = h.get("text", "")[:300].replace("\n", " ")
            print(f"    text (preview): {text_preview!s}")
        # Basic sanity checks
        assert isinstance(hits, list)
        if len(hits) > 0:
            assert all("score" in h for h in hits), "Missing score in hit"
            assert all(isinstance(h["score"], float) for h in hits), "Score not float"

    # 4) success
    print("\nAll retrieval tests completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG pipeline (build index + query)")
    parser.add_argument("--novel", required=True, help="Path to the novel text file (txt)")
    parser.add_argument("--name", required=True, help="Name for the index / novel (used as collection prefix)")
    parser.add_argument("--model", default="avsolatorio/GIST-small-Embedding-v0", help="SentenceTransformer model name")
    parser.add_argument("--qdrant_url", default=DEFAULT_QDRANT_URL, help="Qdrant URL (default http://localhost:6333)")
    parser.add_argument("--qdrant_api_key", default=None, help="Qdrant API key (if any)")
    parser.add_argument("--no_rebuild", action="store_true", help="If set, skip rebuilding the index and only query existing collection")
    parser.add_argument("--topk", type=int, default=5, help="Top-k to retrieve")
    args = parser.parse_args()

    # example queries to test — you can override by editing the list below
    sample_queries = [
        "Who is the protagonist?",
        "Describe the setting of the novel.",
        "What major event happens near the end?"
    ]

    try:
        run_test(
            novel_path=args.novel,
            name=args.name,
            model=args.model,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            rebuild=not args.no_rebuild,
            queries=sample_queries,
            topk=args.topk,
        )
    except AssertionError as ae:
        print("Assertion failed during tests:", ae)
        sys.exit(3)
    except Exception as e:
        print("Test failed with error:", e)
        sys.exit(4)
    sys.exit(0)