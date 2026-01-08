"""
Optimized Pipeline with Memory Management
For processing large novels (100k+ words) without OOM
"""

import os
import json
import logging
from typing import List, Dict, Any, Generator
from pathlib import Path
import gc
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    NOVEL_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ENABLE_AUTO_DELETE_EXISTING,
    LOG_LEVEL,
    VERBOSE,
)
from extractor import GraphRAGExtractor
from neo4j_manager import Neo4jGraphBuilder

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OptimizedNarrativeAuditorPipeline:
    """
    Memory-optimized pipeline for processing large novels.
    Streams chunks instead of loading everything into memory.
    """

    def __init__(self):
        """Initialize pipeline components."""
        logger.info("Initializing Optimized Narrative Auditor Pipeline...")
        self.extractor = GraphRAGExtractor()
        self.graph_builder = Neo4jGraphBuilder()
        self.entities = {}
        self.relationships = []
        self.events = []
        self.checkpoint_interval = 10  # Save every N chunks

    def load_novel_streaming(self, file_path: str) -> Generator[str, None, None]:
        """
        Load novel in chunks to avoid loading entire file into memory.
        
        Args:
            file_path: Path to novel file
        
        Yields:
            Chunks of text (word-based)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Novel file not found: {file_path}")

        logger.info(f"Loading novel from {file_path} (streaming mode)...")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            buffer = ""
            word_count = 0
            target_words = int(CHUNK_SIZE / 1.3)  # 1 token ≈ 1.3 words
            overlap_words = int(CHUNK_OVERLAP / 1.3)

            for line in f:
                buffer += " " + line.strip()
                words = buffer.split()
                word_count = len(words)

                # Yield when buffer reaches target size
                if word_count >= target_words:
                    # Keep overlap for context
                    yield " ".join(words[:target_words])
                    buffer = " ".join(words[target_words - overlap_words:])
                    word_count = len(buffer.split())

            # Yield remaining text
            if buffer.strip():
                yield buffer.strip()

    def process_novel_streaming(self, file_path: str = NOVEL_FILE) -> None:
        """
        Process novel in streaming fashion with memory cleanup.
        
        Args:
            file_path: Path to novel file
        """

        logger.info("Starting streaming pipeline...")

        # MAX_CHUNKS = 4   # ~5000 words
        chunk_id = 0
        dedup_buffer = {}  # Local buffer for deduplication
        rel_buffer = []
        event_buffer = []

        try:
            for chunk_text in self.load_novel_streaming(file_path):
                
                if VERBOSE:
                    logger.info(f"Processing chunk {chunk_id}...")

                # Extract from chunk
                extraction = self.extractor.extract_entities_and_relationships(
                    chunk_text, chunk_id
                )

                # Update local buffers
                self._update_dedup_buffer(extraction, dedup_buffer)
                rel_buffer.extend(extraction.get("relationships", []))
                event_buffer.extend(extraction.get("events", []))

                # Save checkpoint periodically
                if (chunk_id + 1) % self.checkpoint_interval == 0:
                    logger.info(f"Saving checkpoint at chunk {chunk_id}...")
                    self._save_checkpoint(chunk_id, dedup_buffer, rel_buffer, event_buffer)
                    
                    # Memory cleanup
                    gc.collect()

                chunk_id += 1

            # Final deduplication and merging
            logger.info("Finalizing deduplication and merging...")
            self.entities = dedup_buffer
            self.relationships = self._deduplicate_relationships(rel_buffer)
            self.events = event_buffer

            # Build graph
            logger.info("Building Neo4j graph...")
            graph_stats = self.graph_builder.get_graph_statistics()

            # Save final results
            self.save_extraction_results()

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total chunks processed: {chunk_id}")
            logger.info(f"Entities extracted: {len(self.entities)}")
            logger.info(f"Relationships extracted: {len(self.relationships)}")
            logger.info(f"Events extracted: {len(self.events)}")
            logger.info(f"Graph nodes: {graph_stats.get('total_nodes', 0)}")
            logger.info(f"Graph edges: {graph_stats.get('total_relationships', 0)}")
            logger.info("=" * 60)

        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _update_dedup_buffer(
        self,
        extraction: Dict[str, Any],
        buffer: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Update deduplication buffer using namespaced entity IDs:
        - Character::<name>
        - Location::<name>
        """

        chunk_id = extraction.get("chunk_id", 0)

        # ---------- Characters ----------
        for character in extraction.get("characters", []):
            name = character.get("name", "").strip()
            if not name:
                continue

            entity_id = f"Character::{name}"

            if entity_id not in buffer:
                buffer[entity_id] = {
                    "id": entity_id,
                    "name": name,
                    "entity_type": "Character",
                    "character_type": character.get("character_type", "Unknown"),
                    "description": character.get("description", ""),
                    "states": [],
                    "interactions": set(),
                    "chunks_appeared": set(),
                    "first_appearance": chunk_id,
                    "last_appearance": chunk_id,
                }
            else:
                buffer[entity_id]["last_appearance"] = chunk_id

            buffer[entity_id]["states"].extend(character.get("states", []))
            buffer[entity_id]["interactions"].update(character.get("interactions", []))
            buffer[entity_id]["chunks_appeared"].add(chunk_id)

        # ---------- Locations ----------
        for location in extraction.get("locations", []):
            name = location.get("name", "").strip()
            if not name:
                continue

            entity_id = f"Location::{name}"

            if entity_id not in buffer:
                buffer[entity_id] = {
                    "id": entity_id,
                    "name": name,
                    "entity_type": "Location",
                    "location_type": location.get("location_type", "Unknown"),
                    "description": location.get("description", ""),
                    "chunks_appeared": set(),
                    "first_appearance": chunk_id,
                    "last_appearance": chunk_id,
                }
            else:
                buffer[entity_id]["last_appearance"] = chunk_id

            buffer[entity_id]["chunks_appeared"].add(chunk_id)

    def _deduplicate_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate relationships from all chunks.
        
        Args:
            relationships: List of extracted relationships
        
        Returns:
            Deduplicated relationship list
        """
        unique_rels = {}

        for rel in relationships:
            source = f"Character::{rel.get('source_character', '').strip()}"
            target = f"Character::{rel.get('target_character', '').strip()}"


            if not source or not target:
                continue

            key = f"{source}|{target}"

            if key not in unique_rels:
                unique_rels[key] = {
                    "source": source,
                    "target": target,
                    "relationship_type": rel.get("relationship_type", "Unknown"),
                    "description": rel.get("description", ""),
                    "sentiment": rel.get("sentiment", "Neutral"),
                    "occurrences": 1,
                    "chunks": [rel.get("chunk_id", 0)],
                }
            else:
                unique_rels[key]["occurrences"] += 1
                if rel.get("chunk_id", 0) not in unique_rels[key]["chunks"]:
                    unique_rels[key]["chunks"].append(rel.get("chunk_id", 0))

        return list(unique_rels.values())

    def _save_checkpoint(
        self,
        chunk_id: int,
        entities: Dict,
        relationships: List,
        events: List,
    ) -> None:
        """Save intermediate checkpoint."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        checkpoint = {
            "chunk_id": chunk_id,
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "events_count": len(events),
        }

        checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_{chunk_id}.json")
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def save_extraction_results(self) -> None:
        """Save all extraction results to JSON files."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Convert sets to lists for JSON
        entities_serializable = {}
        for k, v in self.entities.items():
            v_copy = v.copy()
            if isinstance(v_copy.get("interactions"), set):
                v_copy["interactions"] = list(v_copy["interactions"])
            if isinstance(v_copy.get("chunks_appeared"), set):
                v_copy["chunks_appeared"] = list(v_copy["chunks_appeared"])
            entities_serializable[k] = v_copy

        # Save entities
        with open(os.path.join(OUTPUT_DIR, "entities.json"), "w") as f:
            json.dump(entities_serializable, f, indent=2)
        logger.info(f"Saved {len(self.entities)} entities")

        # Save relationships
        with open(os.path.join(OUTPUT_DIR, "relationships.json"), "w") as f:
            json.dump(self.relationships, f, indent=2)
        logger.info(f"Saved {len(self.relationships)} relationships")

        # Save events
        with open(os.path.join(OUTPUT_DIR, "events.json"), "w") as f:
            json.dump(self.events, f, indent=2)
        logger.info(f"Saved {len(self.events)} events")

    def cleanup(self):
        """Clean up resources."""
        self.graph_builder.close()


def main():
    """Main entry point with memory-optimized processing."""
    pipeline = OptimizedNarrativeAuditorPipeline()

    try:
        # Configure Neo4j if needed
        if ENABLE_AUTO_DELETE_EXISTING:
            logger.warning("Clearing existing graph...")
            pipeline.graph_builder.clear_graph()

        pipeline.graph_builder.create_constraints()

        # Run streaming pipeline
        pipeline.process_novel_streaming()

        # Build graph from results
        logger.info("Building Neo4j graph from results...")
        pipeline.graph_builder.create_character_nodes(pipeline.entities)
        pipeline.graph_builder.create_location_nodes(pipeline.entities)
        pipeline.graph_builder.create_event_nodes(pipeline.events)
        pipeline.graph_builder.create_character_state_relationships(pipeline.entities)
        pipeline.graph_builder.create_relationship_edges(pipeline.relationships)
        pipeline.graph_builder.create_event_participation_relationships(pipeline.events)
        pipeline.graph_builder.create_location_relationships(pipeline.events)

        # Get and display final statistics
        stats = pipeline.graph_builder.get_graph_statistics()
        logger.info("=" * 60)
        logger.info("FINAL GRAPH STATISTICS")
        logger.info("=" * 60)
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)
        logger.info("Graph is ready for visualization in Neo4j!")
        logger.info(f"Access Neo4j Browser at: http://localhost:7474")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()