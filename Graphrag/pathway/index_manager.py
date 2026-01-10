"""Index management utilities for Qdrant collections.

Usage:
    # List all collections
    python -m Graphrag.pathway.index_manager --list
    
    # Clear specific book
    python -m Graphrag.pathway.index_manager --clear "the count of monte cristo"
    
    # Clear all collections
    python -m Graphrag.pathway.index_manager --clear-all
    
    # Rebuild specific book
    python -m Graphrag.pathway.index_manager --rebuild --path Books/book.txt --name "book name"
"""
import argparse
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from Graphrag.config import QDRANT_URL, QDRANT_API_KEY
from Graphrag.pathway.build_index import build_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def list_collections() -> list[str]:
    """List all Qdrant collections.
    
    Returns:
        List of collection names.
    """
    client = get_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if not collection_names:
        logger.info("No collections found in Qdrant")
        return []
    
    logger.info(f"Found {len(collection_names)} collection(s):")
    for name in collection_names:
        info = client.get_collection(name)
        points_count = info.points_count if info.points_count else 0
        logger.info(f"  - {name}: {points_count} points")
    
    return collection_names


def clear_collection(book_name: str) -> bool:
    """Clear a specific book's collection.
    
    Args:
        book_name: Name of the book (will be normalized to lowercase).
        
    Returns:
        True if collection was deleted, False if not found.
    """
    # Normalize book name to match collection naming convention
    book_name = book_name.lower()
    collection_name = f"{book_name}_collection"
    
    client = get_client()
    existing = client.get_collections().collections
    
    if not any(c.name == collection_name for c in existing):
        logger.warning(f"Collection '{collection_name}' not found")
        return False
    
    client.delete_collection(collection_name)
    logger.info(f"✅ Deleted collection: {collection_name}")
    return True


def clear_all_collections() -> int:
    """Clear all collections in Qdrant.
    
    Returns:
        Number of collections deleted.
    """
    client = get_client()
    collections = client.get_collections().collections
    
    if not collections:
        logger.info("No collections to delete")
        return 0
    
    count = 0
    for collection in collections:
        client.delete_collection(collection.name)
        logger.info(f"✅ Deleted collection: {collection.name}")
        count += 1
    
    logger.info(f"Deleted {count} collection(s)")
    return count


def rebuild_index(book_path: str, book_name: str) -> None:
    """Rebuild index for a specific book.
    
    Args:
        book_path: Path to the book text file.
        book_name: Name for the book index.
    """
    # Clear existing collection
    logger.info(f"Rebuilding index for: {book_name}")
    clear_collection(book_name)
    
    # Rebuild with current config
    logger.info(f"Building new index from: {book_path}")
    build_index(book_path, book_name)
    logger.info("✅ Index rebuild complete")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Manage Qdrant collections for KDAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all collections"
    )
    
    parser.add_argument(
        "--clear",
        type=str,
        metavar="BOOK_NAME",
        help="Clear collection for specific book (case-insensitive)"
    )
    
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all collections (use with caution!)"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild index (requires --path and --name)"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        help="Path to book file (for --rebuild)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Book name (for --rebuild)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.list, args.clear, args.clear_all, args.rebuild]):
        parser.print_help()
        return
    
    if args.rebuild and (not args.path or not args.name):
        parser.error("--rebuild requires both --path and --name")
    
    # Execute commands
    try:
        if args.list:
            list_collections()
        
        if args.clear:
            clear_collection(args.clear)
        
        if args.clear_all:
            response = input("⚠️  This will delete ALL collections. Continue? (yes/no): ")
            if response.lower() == "yes":
                clear_all_collections()
            else:
                logger.info("Cancelled")
        
        if args.rebuild:
            rebuild_index(args.path, args.name)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
