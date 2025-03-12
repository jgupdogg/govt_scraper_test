#!/usr/bin/env python3
"""
Pinecone Diagnostic Script for Government Data Pipeline.
Checks the content of your Pinecone vector database.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
import pinecone
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_pinecone_index():
    """Check if Pinecone index exists and contains vectors."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "govt-scrape-index")
    namespace = os.getenv("PINECONE_NAMESPACE", "govt-content")
    
    if not api_key:
        logger.error("Pinecone API key not found in environment variables")
        return False
    
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=api_key)
        logger.info(f"Connected to Pinecone successfully")
        
        # List all indexes
        indexes = pc.list_indexes().names()
        logger.info(f"Available indexes: {indexes}")
        
        if index_name not in indexes:
            logger.error(f"Index '{index_name}' not found")
            return False
        
        # Connect to index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        total_vectors = stats.total_vector_count
        logger.info(f"Total vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            logger.warning("Index is empty - no vectors found")
            return False
        
        # Check namespace if specified
        if namespace:
            namespaces = stats.namespaces
            if namespace not in namespaces:
                logger.warning(f"Namespace '{namespace}' not found in index")
                return False
            
            namespace_count = namespaces.get(namespace, {}).get("vector_count", 0)
            logger.info(f"Vectors in namespace '{namespace}': {namespace_count}")
            
            if namespace_count == 0:
                logger.warning(f"Namespace '{namespace}' is empty")
                return False
        
        # Try to fetch a few vectors to examine
        try:
            fetch_response = index.fetch(ids=["1", "2", "3", "gov-1", "gov-2", "gov-3"], namespace=namespace)
            vectors = fetch_response.vectors
            
            if not vectors:
                logger.warning("No vectors fetched with sample IDs")
                # Try another approach - query with a generic embedding
                import numpy as np
                dummy_vector = np.random.rand(1536).tolist()  # OpenAI embedding dimension
                
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=5,
                    include_metadata=True,
                    namespace=namespace
                )
                
                matches = query_response.matches
                logger.info(f"Query returned {len(matches)} matches")
                
                if matches:
                    # Print the first match's metadata
                    logger.info(f"Sample vector ID: {matches[0].id}")
                    logger.info(f"Sample metadata: {matches[0].metadata}")
                    
                    # Check if doc_id exists in metadata
                    if "doc_id" not in matches[0].metadata:
                        logger.warning("'doc_id' field not found in vector metadata")
                        print("\nPROBLEM: Vectors don't contain 'doc_id' field in metadata")
                        print("This is likely why your hybrid search isn't returning vector results")
                        
                    return True
            else:
                logger.info(f"Retrieved {len(vectors)} vectors")
                
                # Print sample vector info
                for vec_id, vector in vectors.items():
                    logger.info(f"Vector ID: {vec_id}")
                    logger.info(f"Metadata: {vector.metadata}")
                    
                    # Check if doc_id exists in metadata
                    if vector.metadata and "doc_id" not in vector.metadata:
                        logger.warning("'doc_id' field not found in vector metadata")
                        print("\nPROBLEM: Vectors don't contain 'doc_id' field in metadata")
                        print("This is likely why your hybrid search isn't returning vector results")
                    
                    break  # Just show the first one
                
                return True
                
        except Exception as e:
            logger.error(f"Error fetching vectors: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {e}")
        return False

def fix_suggestions():
    """Print suggestions to fix Pinecone issues."""
    print("\n=== RECOMMENDATIONS ===")
    print("If your Pinecone index is empty or doesn't contain correct metadata, you need to:")
    
    print("\n1. Modify your embedding storage code in the processor.py to include doc_id:")
    print("""
    # In the store_embedding method of your Processor class
    def store_embedding(self, document) -> str:
        # Create a unique ID
        embedding_id = f"gov-{hashlib.md5(document.url.encode()).hexdigest()[:12]}"
        
        # Create LangChain Document
        lc_doc = LCDocument(
            page_content=document.summary,
            metadata={
                "url": document.url,
                "title": document.title,
                "source": document.source_name, 
                "subsource": document.subsource_name,
                "content": document.summary,
                "doc_id": document.doc_id,  # THIS IS THE CRITICAL FIELD
                "processed_at": datetime.now().isoformat()
            }
        )
        
        # Store in vector store
        ids = self._vector_store.add_documents([lc_doc], ids=[embedding_id])
    """)
    
    print("\n2. Re-process your documents to update the embeddings:")
    print("   python runner.py --process-only")
    
    print("\n3. For a quick fix without reprocessing, you can try using URL matching instead:")
    print("""
    # In the hybrid_search method
    # Replace:
    doc_id = result.get("doc_id")
    if doc_id and str(doc_id) in doc_lookup:
    
    # With:
    doc_id = result.get("doc_id")
    url = result.get("url")
    if doc_id and str(doc_id) in doc_lookup:
        # Use doc_id lookup
    elif url:
        # Find by URL
        matching_doc = next((doc for doc in documents if doc.get("url") == url), None)
        if matching_doc:
            result["summary"] = matching_doc.get("summary")
    """)

def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description='Pinecone Diagnostic Script')
    parser.add_argument('--verbose', action='store_true', help='Show verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=== Pinecone Diagnostic Tool ===")
    print("Checking your Pinecone configuration and content...\n")
    
    check_pinecone_index()
    fix_suggestions()

if __name__ == "__main__":
    main()