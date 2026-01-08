"""
Neo4j Graph Database Manager (Option B Compatible)
Uses namespaced entity IDs for collision-free graph construction
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from neo4j import GraphDatabase
from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Neo4jGraphBuilder:
    """
    Manages Neo4j graph database operations for the Narrative Auditor.
    Uses ID-based uniqueness for all nodes and relationships.
    """

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                encrypted=False,
            )
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully!")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared successfully!")

    # ------------------------------------------------------------------
    # CONSTRAINTS
    # ------------------------------------------------------------------

    def create_constraints(self):
        """Create ID-based uniqueness constraints."""
        with self.driver.session() as session:
            session.run("""
            CREATE CONSTRAINT character_id IF NOT EXISTS
            FOR (c:Character) REQUIRE c.id IS UNIQUE
            """)

            session.run("""
            CREATE CONSTRAINT location_id IF NOT EXISTS
            FOR (l:Location) REQUIRE l.id IS UNIQUE
            """)

            session.run("""
            CREATE CONSTRAINT event_id IF NOT EXISTS
            FOR (e:Event) REQUIRE e.id IS UNIQUE
            """)

            logger.info("ID-based constraints created successfully")

    # ------------------------------------------------------------------
    # NODE CREATION
    # ------------------------------------------------------------------

    def create_character_nodes(self, entities: Dict[str, Dict[str, Any]]) -> int:
        count = 0
        query = """
        MERGE (c:Character {id: $id})
        SET c += $props
        RETURN c
        """

        with self.driver.session() as session:
            for entity in entities.values():
                if entity.get("entity_type") != "Character":
                    continue

                props = {
                    "name": entity.get("name"),
                    "character_type": entity.get("character_type", "Unknown"),
                    "description": entity.get("description", ""),
                    "first_appearance": entity.get("first_appearance", 0),
                    "last_appearance": entity.get("last_appearance", 0),
                    "appearances_count": len(entity.get("chunks_appeared", [])),
                    "created_at": datetime.now().isoformat(),
                }

                session.run(query, id=entity["id"], props=props)
                count += 1

        logger.info(f"Created {count} Character nodes")
        return count

    def create_location_nodes(self, entities: Dict[str, Dict[str, Any]]) -> int:
        count = 0
        query = """
        MERGE (l:Location {id: $id})
        SET l += $props
        RETURN l
        """

        with self.driver.session() as session:
            for entity in entities.values():
                if entity.get("entity_type") != "Location":
                    continue

                props = {
                    "name": entity.get("name"),
                    "location_type": entity.get("location_type", "Unknown"),
                    "description": entity.get("description", ""),
                    "appearances_count": len(entity.get("chunks_appeared", [])),
                    "created_at": datetime.now().isoformat(),
                }

                session.run(query, id=entity["id"], props=props)
                count += 1

        logger.info(f"Created {count} Location nodes")
        return count

    def create_event_nodes(self, events: List[Dict[str, Any]]) -> int:
        count = 0
        query = """
        MERGE (e:Event {id: $id})
        SET e += $props
        RETURN e
        """

        with self.driver.session() as session:
            for event in events:
                event_id = f"Event::{event.get('name')}"

                props = {
                    "name": event.get("name"),
                    "description": event.get("description", ""),
                    "event_type": event.get("event_type", "Other"),
                    "timestamp_hint": event.get("timestamp_hint", "Unspecified"),
                    "chunk_id": event.get("chunk_id", 0),
                    "created_at": datetime.now().isoformat(),
                }

                session.run(query, id=event_id, props=props)
                count += 1

        logger.info(f"Created {count} Event nodes")
        return count

    # ------------------------------------------------------------------
    # RELATIONSHIPS
    # ------------------------------------------------------------------

    def create_character_state_relationships(self, entities: Dict[str, Dict[str, Any]]) -> int:
        count = 0
        query = """
        MATCH (c:Character {id: $char_id})
        MERGE (s:CharacterState {
            character_id: $char_id,
            type: $state_type,
            value: $state_value
        })
        MERGE (c)-[r:HAS_STATE]->(s)
        SET r.confidence = $confidence
        RETURN r
        """

        with self.driver.session() as session:
            for entity in entities.values():
                if entity.get("entity_type") != "Character":
                    continue

                for state in entity.get("states", []):
                    session.run(
                        query,
                        char_id=entity["id"],
                        state_type=state.get("type", "Unknown"),
                        state_value=state.get("value", ""),
                        confidence=state.get("confidence", 0.5),
                    )
                    count += 1

        logger.info(f"Created {count} CharacterState relationships")
        return count

    def create_relationship_edges(self, relationships: List[Dict[str, Any]]) -> int:
        count = 0
        query = """
        MATCH (s:Character {id: $source})
        MATCH (t:Character {id: $target})
        MERGE (s)-[r:RELATES_TO]->(t)
        SET r += $props
        RETURN r
        """

        with self.driver.session() as session:
            for rel in relationships:
                props = {
                    "relationship_type": rel.get("relationship_type", "Unknown"),
                    "description": rel.get("description", ""),
                    "sentiment": rel.get("sentiment", "Neutral"),
                    "occurrences": rel.get("occurrences", 1),
                }

                session.run(
                    query,
                    source=rel["source"],
                    target=rel["target"],
                    props=props,
                )
                count += 1

        logger.info(f"Created {count} Character relationships")
        return count

    def create_event_participation_relationships(self, events: List[Dict[str, Any]]) -> int:
        count = 0
        query = """
        MATCH (c:Character {id: $char_id})
        MATCH (e:Event {id: $event_id})
        MERGE (c)-[r:PARTICIPATES_IN]->(e)
        SET r.created_at = $ts
        RETURN r
        """

        with self.driver.session() as session:
            for event in events:
                event_id = f"Event::{event.get('name')}"
                for participant in event.get("participants", []):
                    char_id = f"Character::{participant.strip()}"
                    session.run(
                        query,
                        char_id=char_id,
                        event_id=event_id,
                        ts=datetime.now().isoformat(),
                    )
                    count += 1

        logger.info(f"Created {count} Event participation relationships")
        return count

    def create_location_relationships(self, events: List[Dict[str, Any]]) -> int:
        count = 0
        query = """
        MATCH (e:Event {id: $event_id})
        MATCH (l:Location {id: $location_id})
        MERGE (e)-[r:OCCURS_AT]->(l)
        SET r.created_at = $ts
        RETURN r
        """

        with self.driver.session() as session:
            for event in events:
                location = event.get("location", "").strip()
                if not location:
                    continue

                session.run(
                    query,
                    event_id=f"Event::{event.get('name')}",
                    location_id=f"Location::{location}",
                    ts=datetime.now().isoformat(),
                )
                count += 1

        logger.info(f"Created {count} Event–Location relationships")
        return count

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_graph_statistics(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            return {
                "characters": session.run("MATCH (c:Character) RETURN count(c)").single()[0],
                "locations": session.run("MATCH (l:Location) RETURN count(l)").single()[0],
                "events": session.run("MATCH (e:Event) RETURN count(e)").single()[0],
                "states": session.run("MATCH (s:CharacterState) RETURN count(s)").single()[0],
                "relationships": session.run("MATCH ()-[r]->() RETURN count(r)").single()[0],
                "total_nodes": session.run("MATCH (n) RETURN count(n)").single()[0],
            }
