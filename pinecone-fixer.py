#!/usr/bin/env python3
"""
Revised Pinecone Metadata Fixer Script for Government Data Pipeline.
Updates vector metadata in Pinecone to add missing doc_id fields.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def fix_pinecone_metadata():
    """Add doc_id field to Pinecone vector metadata using URL matching."""
    # First, check if required packages are installed
    try:
        import pinecone
        from supabase import create_client
    except ImportError:
        logger.error("Required packages not installed. Run: pip install pinecone-client supabase tqdm")
        return False
    
    # Get environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME", "govt-scrape-index")
    pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "govt-content")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([pinecone_api_key, pinecone_index, supabase_url, supabase_key]):
        logger.error("Missing required environment variables")
        return False
    
    try:
        # Initialize Pinecone
        logger.info("Connecting to Pinecone...")
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        
        # Get index
        if pinecone_index not in pc.list_indexes().names():
            logger.error(f"Index '{pinecone_index}' not found")
            return False
        
        index = pc.Index(pinecone_index)
        
        # Check if index has vectors
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors == 0:
            logger.error("Index is empty - no vectors to update")
            return False
        
        # Check namespace stats if specified
        if pinecone_namespace:
            namespaces = stats.namespaces
            if pinecone_namespace not in namespaces:
                logger.warning(f"Namespace '{pinecone_namespace}' not found in index")
                return False
            
            namespace_count = namespaces.get(pinecone_namespace, {}).get("vector_count", 0)
            logger.info(f"Found {namespace_count} vectors in namespace '{pinecone_namespace}'")
            
            if namespace_count == 0:
                logger.warning(f"Namespace '{pinecone_namespace}' is empty")
                return False
        
        # Initialize Supabase
        logger.info("Connecting to Supabase...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Fetch all documents from Supabase
        logger.info("Fetching documents from Supabase...")
        result = supabase.table("govt_documents").select("id", "url", "title").execute()
        
        if not result.data:
            logger.error("No documents found in Supabase")
            return False
        
        # Create a lookup dictionary by URL
        url_to_id = {doc["url"]: doc["id"] for doc in result.data if "url" in doc and "id" in doc}
        
        logger.info(f"Loaded {len(url_to_id)} document URL-to-ID mappings from Supabase")
        
        # Create a dummy 1536-dimension vector (OpenAI embedding size)
        dummy_vector = np.zeros(1536).tolist()
        
        # Create method to fetch actual vectors
        def fetch_vectors(limit=100):
            # Fetch existing vectors using list_vectors if available
            # Otherwise use a query
            query_response = index.query(
                vector=dummy_vector,
                top_k=limit,
                include_values=True,  # Get the actual vector values
                include_metadata=True,
                namespace=pinecone_namespace
            )
            
            return query_response.matches
        
        logger.info("Fetching vectors from Pinecone...")
        vectors = fetch_vectors(limit=100)  # Get first batch of vectors
        
        if not vectors:
            logger.error("No vectors could be retrieved from Pinecone")
            return False
        
        # Check first few vectors to see if they need updating
        sample_vectors = vectors[:10]
        need_update = any("doc_id" not in vec.metadata and "url" in vec.metadata for vec in sample_vectors)
        
        if not need_update:
            logger.info("Vectors already have doc_id field - no update needed")
            return True
        
        # Process vectors to update
        logger.info("Starting vector metadata update process...")
        
        updates = []
        success_count = 0
        error_count = 0
        
        for vec in vectors:
            # Check if vector needs updating
            if "doc_id" in vec.metadata or "url" not in vec.metadata:
                continue
                
            url = vec.metadata.get("url")
            if url in url_to_id:
                doc_id = url_to_id[url]
                
                # Create updated metadata
                new_metadata = {**vec.metadata, "doc_id": doc_id}
                
                try:
                    # Get the vector's ID and values
                    vector_id = vec.id
                    vector_values = vec.values if hasattr(vec, 'values') else dummy_vector
                    
                    # Use upsert to update the vector with the new metadata
                    # This format should work: {'id': id, 'metadata': metadata, 'values': values}
                    index.upsert(
                        vectors=[{
                            'id': vector_id,
                            'metadata': new_metadata,
                            'values': vector_values
                        }],
                        namespace=pinecone_namespace
                    )
                    
                    logger.info(f"Updated vector ID {vector_id} with doc_id {doc_id}")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error updating vector {vec.id}: {e}")
                    error_count += 1
        
        logger.info(f"Completed updates: {success_count} successful, {error_count} errors")
        
        # Verify updates
        logger.info("Verifying updates...")
        updated_vectors = fetch_vectors(limit=10)  # Get a fresh sample
        
        if updated_vectors:
            with_doc_id = sum(1 for vec in updated_vectors if "doc_id" in vec.metadata)
            logger.info(f"Verification: {with_doc_id}/{len(updated_vectors)} sampled vectors now have doc_id field")
        
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error updating Pinecone metadata: {e}", exc_info=True)
        return False

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description='Pinecone Metadata Fixer Script')
    parser.add_argument('--force', action='store_true', help='Force update even if vectors appear to have doc_id')
    parser.add_argument('--verbose', action='store_true', help='Show verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=== Pinecone Metadata Fixer (Revised) ===")
    print("This script will update your Pinecone vectors to add missing doc_id fields.\n")
    
    # Confirm with user
    if not args.force:
        confirm = input("Are you sure you want to proceed? This will modify your Pinecone data. (y/n): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    success = fix_pinecone_metadata()
    
    if success:
        print("\n=== SUCCESS ===")
        print("Your Pinecone vectors have been updated with doc_id fields.")
        print("Now try running hybrid-search-optimized.py again, and you should see vector results.")
    else:
        print("\n=== FAILED ===")
        print("There was an error updating your Pinecone vectors.")
        print("Check the logs above for more information.")
        print("\nALTERNATIVE SOLUTION:")
        print("Since updating Pinecone vectors is problematic, you can use the URL fallback approach:")
        print("1. Add the URL fallback methods to hybrid-search-optimized.py")
        print("2. This will fetch summaries by URL instead of doc_id")

if __name__ == "__main__":
    main()