#!/usr/bin/env python3
"""
Search utility for the government data pipeline.
Provides both vector search and full-text search capabilities.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import core components and Supabase manager
from core import Processor
from supabase_manager import SupabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def semantic_search(query: str, processor: Processor, top_k: int = 5, source: str = None) -> List[Dict[str, Any]]:
    """
    Perform semantic search using vector embeddings.
    
    Args:
        query: Search query
        processor: Processor instance with vector search capabilities
        top_k: Maximum number of results
        source: Optional source filter
        
    Returns:
        List of search results
    """
    try:
        logger.info(f"Performing semantic search for: '{query}'")
        results = processor.search_similar_documents(query, k=top_k)
        
        if source:
            # Filter results by source
            results = [r for r in results if source.lower() in r.get('source', '').lower()]
        
        return results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def full_text_search(query: str, db_manager: SupabaseManager, limit: int = 10, source: str = None) -> List[Dict[str, Any]]:
    """
    Perform full-text search using Supabase.
    
    Args:
        query: Search query
        db_manager: Supabase manager instance
        limit: Maximum number of results
        source: Optional source filter
        
    Returns:
        List of search results
    """
    try:
        logger.info(f"Performing full-text search for: '{query}'")
        results = db_manager.search_full_text(query, limit)
        
        if source and results:
            # Filter results by source
            results = [r for r in results if source.lower() in r.get('source_name', '').lower()]
        
        return results
    
    except Exception as e:
        logger.error(f"Error in full-text search: {e}")
        return []

def format_results(results: List[Dict[str, Any]], search_type: str) -> None:
    """
    Format and print search results.
    
    Args:
        results: Search results
        search_type: Type of search performed
    """
    if not results:
        print(f"No {search_type} search results found.")
        return
    
    print(f"\n===== {search_type.title()} Search Results ({len(results)}) =====\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result.get('title', 'Untitled')}")
        print(f"    Source: {result.get('source', result.get('source_name', 'Unknown'))}")
        print(f"    URL: {result.get('url', 'No URL')}")
        
        if search_type == "semantic":
            print(f"    Similarity: {result.get('similarity_score', 0):.2f}")
            print(f"    Summary: {result.get('summary', 'No summary')[:200]}...")
        else:  # full-text
            print(f"    Relevance: {result.get('rank', 0):.2f}")
            print(f"    Highlight: {result.get('highlight', 'No highlight')}")
        
        print()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Government data search utility')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--supabase-url', default=os.getenv('SUPABASE_URL'), help='Supabase project URL')
    parser.add_argument('--supabase-key', default=os.getenv('SUPABASE_KEY'), help='Supabase API key')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--source', help='Filter results by source')
    parser.add_argument('--semantic-only', action='store_true', help='Only perform semantic search')
    parser.add_argument('--full-text-only', action='store_true', help='Only perform full-text search')
    
    args = parser.parse_args()
    
    # Check Supabase credentials
    if not args.supabase_url or not args.supabase_key:
        logger.error("Supabase URL and key not provided. Set SUPABASE_URL and SUPABASE_KEY environment variables or use --supabase-url and --supabase-key")
        sys.exit(1)
    
    try:
        # Initialize Supabase manager
        db_manager = SupabaseManager(args.supabase_url, args.supabase_key)
        
        # Determine search modes
        do_semantic = not args.full_text_only
        do_full_text = not args.semantic_only
        
        if do_semantic:
            # Initialize processor for semantic search
            processor = Processor()
            
            # Perform semantic search
            semantic_results = semantic_search(args.query, processor, args.top_k, args.source)
            format_results(semantic_results, "semantic")
        
        if do_full_text:
            # Perform full-text search
            fulltext_results = full_text_search(args.query, db_manager, args.top_k, args.source)
            format_results(fulltext_results, "full-text")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()