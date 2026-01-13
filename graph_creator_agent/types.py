"""Type definitions for graph creator agent."""
from pydantic import BaseModel, Field


class Triplet(BaseModel):
    """Knowledge triplet structure."""
    subject: str = Field(description="The subject entity of the triplet.")
    relation: str = Field(description="The relationship between subject and object.")
    object: str = Field(description="The object entity of the triplet.")
    evidence_id: str = Field(description="The ID of the evidence chunk where this was found.")


class TripletList(BaseModel):
    """Wrapper for list of triplets and narrative summary for structured output."""
    triplets: list[Triplet] = Field(description="List of extracted knowledge triplets.")
    graph_summary: str = Field(description="A concise narrative summary of the character's journey based on the evidence.")
