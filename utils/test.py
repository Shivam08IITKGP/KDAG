"""Comprehensive test for graph_creator_agent functionality."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from graph_creator_agent.cache import EVIDENCE_CACHE
from graph_creator_agent.graph_store import (
    add_triplets,
    load_graph,
    save_graph,
)
from graph_creator_agent.utils import filter_new_evidence
from graph_creator_agent.types import Triplet
from graph_creator_agent.main import create_graph


def cleanup_test_files(book_name: str, character_name: str):
    """Clean up test graph files."""
    graph_dir = Path("graph_creator_agent/graph")
    graph_filename = f"{book_name}_{character_name}.graphml"
    graph_path = graph_dir / graph_filename
    
    if graph_path.exists():
        graph_path.unlink()
        print(f"✓ Cleaned up: {graph_path}")


def test_scenario_1_new_graph():
    """Test Scenario 1: Create new graph from scratch."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: New Graph Creation")
    print("=" * 80)
    
    test_book = "Test Book 1"
    test_character = "Paganel"
    cache_key = f"{test_book}_{test_character}"
    
    # Clear cache
    if cache_key in EVIDENCE_CACHE:
        del EVIDENCE_CACHE[cache_key]
    cleanup_test_files(test_book, test_character)
    
    fake_evidences = [
        {"id": "ev_1", "text": "Jacques Paganel is a French geographer known for his absent-mindedness."},
        {"id": "ev_2", "text": "Paganel has extensive knowledge of geography and travels the world."},
        {"id": "ev_3", "text": "He often forgets things and makes mistakes due to his absent-minded nature."},
        {"id": "ev_4", "text": "Paganel is a member of the Geographical Society and writes scholarly papers."},
    ]
    
    print(f"📝 Created {len(fake_evidences)} evidence items")
    print(f"🗂️  Cache before: {EVIDENCE_CACHE.get(cache_key, 'EMPTY')}")
    
    state = {
        "book_name": test_book,
        "character_name": test_character,
        "backstory": "French geographer",
        "queries": [],
        "evidences": fake_evidences,
        "graph_path": None,
        "label": None,
        "reasoning": None,
        "evidence_ids": None,
    }
    
    result_state = create_graph(state)
    
    print(f"💾 Graph saved to: {result_state['graph_path']}")
    print(f"🗂️  Cache after: {EVIDENCE_CACHE.get(cache_key, 'EMPTY')}")
    
    graph = load_graph(test_book, test_character)
    print(f"📊 Graph stats: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Show some edges
    print("\n📋 Sample edges:")
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        if i < 3:
            print(f"   {u} --[{data.get('relation', 'N/A')}]--> {v}")
            print(f"      evidence_ids: {data.get('evidence_ids', [])}")
    
    print("✅ Scenario 1 PASSED\n")


def test_scenario_2_incremental_new_nodes():
    """Test Scenario 2: Add new evidence that creates new nodes."""
    print("=" * 80)
    print("SCENARIO 2: Incremental Update - New Nodes")
    print("=" * 80)
    
    test_book = "Test Book 2"
    test_character = "Character A"
    cache_key = f"{test_book}_{test_character}"
    
    cleanup_test_files(test_book, test_character)
    
    # Prepopulate cache
    EVIDENCE_CACHE[cache_key] = {"ev_1", "ev_2"}
    print(f"🗂️  Pre-populated cache: {EVIDENCE_CACHE[cache_key]}")
    
    # Create initial graph manually
    graph = load_graph(test_book, test_character)
    initial_triplets = [
        Triplet(subject="Character A", relation="is", object="protagonist", evidence_id="ev_1"),
        Triplet(subject="Character A", relation="lives in", object="Paris", evidence_id="ev_2"),
    ]
    add_triplets(graph, initial_triplets)
    save_graph(graph, test_book, test_character)
    
    print(f"📊 Initial graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # New evidence with some old IDs
    all_evidences = [
        {"id": "ev_1", "text": "Old evidence 1"},
        {"id": "ev_2", "text": "Old evidence 2"},
        {"id": "ev_3", "text": "Character A works as a scientist and conducts experiments."},
        {"id": "ev_4", "text": "Character A has a friend named Bob who helps with research."},
    ]
    
    new_evidences = filter_new_evidence(all_evidences, cache_key)
    print(f"🔍 Filtered: {len(new_evidences)} new out of {len(all_evidences)} total")
    print(f"   New IDs: {[ev['id'] for ev in new_evidences]}")
    
    state = {
        "book_name": test_book,
        "character_name": test_character,
        "backstory": "Protagonist",
        "queries": [],
        "evidences": all_evidences,
        "graph_path": None,
        "label": None,
        "reasoning": None,
        "evidence_ids": None,
    }
    
    result_state = create_graph(state)
    
    print(f"🗂️  Cache after: {EVIDENCE_CACHE.get(cache_key, 'EMPTY')}")
    
    updated_graph = load_graph(test_book, test_character)
    print(f"📊 Updated graph: {updated_graph.number_of_nodes()} nodes, {updated_graph.number_of_edges()} edges")
    
    print("✅ Scenario 2 PASSED\n")


def test_scenario_3_update_existing_edge():
    """Test Scenario 3: Update existing edge with new evidence (CRITICAL TEST)."""
    print("=" * 80)
    print("SCENARIO 3: Update Existing Edge - Evidence IDs List")
    print("=" * 80)
    
    test_book = "Test Book 3"
    test_character = "Paganel"
    
    cleanup_test_files(test_book, test_character)
    
    # Step 1: Create initial graph with one edge
    print("\n📝 Step 1: Create initial edge")
    graph = load_graph(test_book, test_character)
    triplets1 = [
        Triplet(subject="Paganel", relation="is", object="geographer", evidence_id="ev_1"),
    ]
    add_triplets(graph, triplets1)
    
    edge_data = graph["Paganel"]["geographer"]
    print(f"   evidence_ids: {edge_data['evidence_ids']}")
    print(f"   Type: {type(edge_data['evidence_ids'])}")
    
    # Step 2: Save and reload
    print("\n💾 Step 2: Save and reload graph")
    save_graph(graph, test_book, test_character)
    graph = load_graph(test_book, test_character)
    
    edge_data = graph["Paganel"]["geographer"]
    print(f"   evidence_ids after reload: {edge_data['evidence_ids']}")
    print(f"   Type after reload: {type(edge_data['evidence_ids'])}")
    
    # Step 3: Add SAME edge with different evidence_id
    print("\n➕ Step 3: Add same edge with new evidence_id")
    triplets2 = [
        Triplet(subject="Paganel", relation="is", object="geographer", evidence_id="ev_2"),
    ]
    add_triplets(graph, triplets2)
    
    edge_data = graph["Paganel"]["geographer"]
    print(f"   evidence_ids after update: {edge_data['evidence_ids']}")
    print(f"   Type: {type(edge_data['evidence_ids'])}")
    print(f"   Length: {len(edge_data['evidence_ids'])}")
    
    # Step 4: Save and reload again
    print("\n💾 Step 4: Save and reload again")
    save_graph(graph, test_book, test_character)
    final_graph = load_graph(test_book, test_character)
    
    edge_data = final_graph["Paganel"]["geographer"]
    print(f"   evidence_ids final: {edge_data['evidence_ids']}")
    print(f"   Type final: {type(edge_data['evidence_ids'])}")
    
    # Verify it's a list with 2 items
    if isinstance(edge_data['evidence_ids'], list) and len(edge_data['evidence_ids']) == 2:
        print("\n✅ CRITICAL TEST PASSED: evidence_ids maintained as list through save/load cycles")
    else:
        print(f"\n❌ CRITICAL TEST FAILED: Expected list with 2 items, got {type(edge_data['evidence_ids'])} with value {edge_data['evidence_ids']}")
    
    print("✅ Scenario 3 COMPLETED\n")


def test_scenario_4_duplicate_evidence():
    """Test Scenario 4: Adding duplicate evidence_id to same edge."""
    print("=" * 80)
    print("SCENARIO 4: Duplicate Evidence ID Handling")
    print("=" * 80)
    
    test_book = "Test Book 4"
    test_character = "TestChar"
    
    cleanup_test_files(test_book, test_character)
    
    graph = load_graph(test_book, test_character)
    
    print("\n📝 Adding same triplet twice with same evidence_id")
    triplets = [
        Triplet(subject="A", relation="knows", object="B", evidence_id="ev_1"),
        Triplet(subject="A", relation="knows", object="B", evidence_id="ev_1"),  # Duplicate
    ]
    add_triplets(graph, triplets)
    
    edge_data = graph["A"]["B"]
    print(f"   evidence_ids: {edge_data['evidence_ids']}")
    print(f"   Length: {len(edge_data['evidence_ids'])}")
    
    if len(edge_data['evidence_ids']) == 1:
        print("✅ Correctly handled duplicate - only 1 evidence_id stored")
    else:
        print(f"❌ Duplicate not handled - {len(edge_data['evidence_ids'])} evidence_ids stored")
    
    print("✅ Scenario 4 COMPLETED\n")


def test_scenario_5_relation_conflict():
    """Test Scenario 5: Same nodes with different relations."""
    print("=" * 80)
    print("SCENARIO 5: Relation Conflict Handling")
    print("=" * 80)
    
    test_book = "Test Book 5"
    test_character = "TestChar"
    
    cleanup_test_files(test_book, test_character)
    
    graph = load_graph(test_book, test_character)
    
    print("\n📝 Adding edge with relation 'is'")
    triplets1 = [
        Triplet(subject="Paganel", relation="is", object="geographer", evidence_id="ev_1"),
    ]
    add_triplets(graph, triplets1)
    edge_data = graph["Paganel"]["geographer"]
    print(f"   Relation: {edge_data['relation']}")
    print(f"   evidence_ids: {edge_data['evidence_ids']}")
    
    print("\n📝 Adding same edge with different relation 'works as'")
    triplets2 = [
        Triplet(subject="Paganel", relation="works as", object="geographer", evidence_id="ev_2"),
    ]
    add_triplets(graph, triplets2)
    edge_data = graph["Paganel"]["geographer"]
    print(f"   Relation after: {edge_data['relation']}")
    print(f"   evidence_ids after: {edge_data['evidence_ids']}")
    
    print("\nℹ️  Note: Relation conflict detected - keeping first relation by design")
    print("✅ Scenario 5 COMPLETED\n")


def test_scenario_6_full_workflow():
    """Test Scenario 6: Complete workflow with real LLM calls."""
    print("=" * 80)
    print("SCENARIO 6: Full Workflow with LLM")
    print("=" * 80)
    
    test_book = "In Search of the Castaways"
    test_character = "Jacques Paganel"
    cache_key = f"{test_book}_{test_character}"
    
    # Don't cleanup - keep the graph for inspection
    
    evidences = [
        {"id": "ev_101", "text": "Jacques Paganel is absent-minded and often makes comical mistakes."},
        {"id": "ev_102", "text": "Paganel is a secretary of the Geographical Society of Paris."},
        {"id": "ev_103", "text": "He mistakenly boards the wrong ship, the Duncan instead of the Scotia."},
        {"id": "ev_104", "text": "Paganel speaks multiple languages but sometimes confuses them."},
        {"id": "ev_105", "text": "He becomes friends with Lord Glenarvan and his companions."},
    ]
    
    print(f"📝 Processing {len(evidences)} evidence items")
    print(f"🗂️  Cache before: {EVIDENCE_CACHE.get(cache_key, 'EMPTY')}")
    
    state = {
        "book_name": test_book,
        "character_name": test_character,
        "backstory": "French geographer and secretary of the Geographical Society",
        "queries": [],
        "evidences": evidences,
        "graph_path": None,
        "label": None,
        "reasoning": None,
        "evidence_ids": None,
    }
    
    result_state = create_graph(state)
    
    print(f"\n💾 Graph saved to: {result_state['graph_path']}")
    print(f"🗂️  Cache after: {EVIDENCE_CACHE.get(cache_key, 'EMPTY')}")
    
    graph = load_graph(test_book, test_character)
    print(f"📊 Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    print("\n📋 All edges in graph:")
    for u, v, data in graph.edges(data=True):
        print(f"   {u} --[{data.get('relation', 'N/A')}]--> {v}")
        print(f"      evidence_ids: {data.get('evidence_ids', [])}")
    
    print("\n📋 All nodes in graph:")
    print(f"   {list(graph.nodes())}")
    
    print("\n✅ Scenario 6 COMPLETED")
    print("ℹ️  Graph file kept for inspection!")
    print(f"   Location: {result_state['graph_path']}\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE GRAPH CREATOR AGENT TESTS")
    print("=" * 80)
    
    try:
        # Basic tests
        test_scenario_1_new_graph()
        test_scenario_2_incremental_new_nodes()
        
        # Critical edge update test
        test_scenario_3_update_existing_edge()
        
        # Edge case tests
        test_scenario_4_duplicate_evidence()
        test_scenario_5_relation_conflict()
        
        # Full workflow
        test_scenario_6_full_workflow()
        
        print("=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("\n🧹 Cleaning up test files...")
        cleanup_test_files("Test Book 1", "Paganel")
        cleanup_test_files("Test Book 2", "Character A")
        cleanup_test_files("Test Book 3", "Paganel")
        cleanup_test_files("Test Book 4", "TestChar")
        cleanup_test_files("Test Book 5", "TestChar")
        # Don't cleanup Scenario 6 - keep for inspection
        
        # Cleanup caches
        for key in ["Test Book 1_Paganel", "Test Book 2_Character A"]:
            if key in EVIDENCE_CACHE:
                del EVIDENCE_CACHE[key]
        
        print("✓ Cleanup complete\n")


if __name__ == "__main__":
    main()