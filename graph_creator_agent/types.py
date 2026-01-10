"""Type definitions for graph creator agent."""
from typing import TypedDict

class Triplet(TypedDict):
    """Knowledge triplet structure."""
    subject: str
    relation: str
    object: str
    evidence_id: str


class TripletList(TypedDict):
    """Wrapper for list of triplets for structured output."""
    triplets: list[Triplet]
