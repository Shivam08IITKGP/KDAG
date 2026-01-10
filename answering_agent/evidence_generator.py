"""Evidence retrieval module for answering agent."""
import logging
from typing import TypedDict

from Graphrag.pathway.retriever import retrieve_topk

logger = logging.getLogger(__name__)


class EvidenceOutput(TypedDict):
    """Output from evidence retrieval."""
    evidence_chunks: list[dict]  # List of {id, text, score, query}


def retrieve_evidence_for_queries(
    evidence_queries: list[str],
    book_name: str,
    k: int = 3,
) -> EvidenceOutput:
    """Retrieve evidence chunks for each query.
    
    Args:
        evidence_queries: List of queries from classifier.
        book_name: Name of the book to search.
        k: Number of chunks to retrieve per query.
        
    Returns:
        EvidenceOutput TypedDict with all retrieved chunks.
    """
    logger.info(f"Retrieving evidence for {len(evidence_queries)} queries")
    
    all_evidence_chunks = []
    seen_chunk_ids = set()
    
    for idx, query in enumerate(evidence_queries, 1):
        logger.info(f"Query {idx}/{len(evidence_queries)}: {query}")
        
        try:
            # Retrieve top-k chunks for this query
            chunks = retrieve_topk(book_name, query, k=k)
            
            logger.info(f"Retrieved {len(chunks)} chunks for query {idx}")
            
            # Add query context and deduplicate
            for chunk in chunks:
                chunk_id = chunk.get("id")
                if chunk_id not in seen_chunk_ids:
                    chunk["query"] = query  # Track which query retrieved this
                    all_evidence_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
                    logger.debug(f"  - {chunk_id}: score={chunk.get('score', 0):.3f}")
        
        except Exception as e:
            logger.error(f"Error retrieving evidence for query '{query}': {e}")
            continue
    
    logger.info(f"Total unique evidence chunks retrieved: {len(all_evidence_chunks)}")
    
    return EvidenceOutput(evidence_chunks=all_evidence_chunks)
