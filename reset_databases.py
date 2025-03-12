#!/usr/bin/env python3
"""
Reset script for government data pipeline databases.
Resets Supabase, Pinecone, and Neo4j (optional) databases.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
import time
import pinecone
from supabase import create_client

# Try to import Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j Python driver not installed. To enable Neo4j reset, run: pip install neo4j")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def reset_supabase(supabase_url, supabase_key):
    """
    Reset Supabase tables by clearing data and displaying SQL for manual recreation.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Initializing Supabase client")
        supabase = create_client(supabase_url, supabase_key)
        
        # We can't execute arbitrary SQL via RPC - that's a security feature of Supabase
        # Instead, let's clear the existing tables and print SQL for manual execution
        logger.info("Attempting to clear data from existing tables")
        
        # Delete all data from tables (if they exist)
        try:
            # Delete in reverse order of dependencies
            logger.info("Clearing govt_documents table")
            supabase.table("govt_documents").delete().execute()
            logger.info("Clearing govt_subsources table")
            supabase.table("govt_subsources").delete().execute()
            logger.info("Clearing govt_sources table")
            supabase.table("govt_sources").delete().execute()
            logger.info("Successfully cleared all tables")
        except Exception as e:
            logger.warning(f"Error clearing tables: {e}")
            logger.warning("Tables may not exist yet or another issue occurred")
        
        # Print the SQL for manual execution in Supabase SQL Editor
        logger.info("To fully reset and recreate tables, run this SQL in the Supabase SQL Editor:")
        sql_for_manual_execution = """
        -- Drop existing tables (in reverse order of dependencies)
        DROP TABLE IF EXISTS govt_documents;
        DROP TABLE IF EXISTS govt_subsources;
        DROP TABLE IF EXISTS govt_sources;
        
        -- Recreate tables
        CREATE TABLE govt_sources (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            base_url TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE govt_subsources (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES govt_sources(id),
            name VARCHAR(255) NOT NULL,
            url_pattern TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE TABLE govt_documents (
            id SERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            source_name VARCHAR(255) NOT NULL,
            subsource_name VARCHAR(255) NOT NULL,
            content TEXT,
            content_hash VARCHAR(32),
            summary TEXT,
            embedding_id VARCHAR(100),
            status VARCHAR(50) DEFAULT 'new',
            scrape_time TIMESTAMPTZ,
            process_time TIMESTAMPTZ,
            last_checked TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create full text search index
        CREATE INDEX idx_govt_documents_content_tsvector
        ON govt_documents USING gin(to_tsvector('english', content));
        
        -- Set up RLS
        ALTER TABLE govt_sources ENABLE ROW LEVEL SECURITY;
        ALTER TABLE govt_subsources ENABLE ROW LEVEL SECURITY;
        ALTER TABLE govt_documents ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Allow service role access to govt_sources"
            ON govt_sources FOR ALL
            USING (auth.role() = 'service_role');
            
        CREATE POLICY "Allow service role access to govt_subsources"
            ON govt_subsources FOR ALL
            USING (auth.role() = 'service_role');
            
        CREATE POLICY "Allow service role access to govt_documents"
            ON govt_documents FOR ALL
            USING (auth.role() = 'service_role');
        
        -- Create stored procedure for search
        CREATE OR REPLACE FUNCTION search_documents(search_query TEXT, max_results INT)
        RETURNS TABLE (
          id INT,
          url TEXT,
          title TEXT,
          source_name TEXT,
          subsource_name TEXT,
          highlight TEXT,
          rank FLOAT
        ) AS $
        BEGIN
          RETURN QUERY
          SELECT 
              d.id, 
              d.url, 
              d.title, 
              d.source_name, 
              d.subsource_name,
              ts_headline('english', d.content, plainto_tsquery('english', search_query), 
                        'MaxFragments=3, MaxWords=30, MinWords=5') as highlight,
              ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', search_query)) as rank
          FROM govt_documents d
          WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', search_query)
          ORDER BY rank DESC
          LIMIT max_results;
        END;
        $ LANGUAGE plpgsql;
        """
        print("\n" + "-"*80)
        print("SQL TO RUN IN SUPABASE SQL EDITOR:")
        print("-"*80)
        print(sql_for_manual_execution)
        print("-"*80 + "\n")
        
        # Note: In reality, executing arbitrary SQL via RPC might not be allowed
        # due to security restrictions in Supabase. If this approach doesn't work,
        # we'll need to handle this differently, such as:
        # 1. Create a PostgreSQL function that does the reset
        # 2. Call that function via RPC
        # 3. Or provide SQL to run manually in the SQL Editor
        
        # Empty tables as a fallback if the SQL execution fails
        try:
            # Delete all data from tables
            supabase.table("govt_documents").delete().neq("id", 0).execute()
            supabase.table("govt_subsources").delete().neq("id", 0).execute()
            supabase.table("govt_sources").delete().neq("id", 0).execute()
            logger.info("All data deleted from tables")
        except Exception as e:
            logger.warning(f"Could not delete data via API, may need SQL: {e}")
            
        # Check storage buckets
        try:
            storage_buckets = supabase.storage.list_buckets()
            doc_bucket_exists = any(b["name"] == "documents" for b in storage_buckets)
            
            if doc_bucket_exists:
                # Empty the bucket
                files = supabase.storage.from_("documents").list()
                for file in files:
                    supabase.storage.from_("documents").remove([file["name"]])
                logger.info("Cleared documents storage bucket")
            else:
                # Create the bucket
                supabase.storage.create_bucket("documents")
                logger.info("Created documents storage bucket")
        except Exception as e:
            logger.warning(f"Error managing storage buckets: {e}")
        
        logger.info("Supabase reset completed")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting Supabase: {e}")
        return False

def reset_pinecone(api_key, index_name):
    """
    Reset Pinecone index by deleting and recreating it.
    
    Args:
        api_key: Pinecone API key
        index_name: Name of the Pinecone index
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Initializing Pinecone client with new API")
        from pinecone import Pinecone, ServerlessSpec
        
        # Use new Pinecone API
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            logger.info(f"Deleting existing Pinecone index: {index_name}")
            pc.delete_index(index_name)
            
            # Wait for deletion to complete
            logger.info("Waiting for index deletion to complete...")
            time.sleep(20)  # Pinecone operations can take time to propagate
            
        # Create new index
        logger.info(f"Creating new Pinecone index: {index_name}")
        
        # Create index with serverless spec
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # You may need to change this based on your requirements
            )
        )
        
        # Wait for creation to complete
        logger.info("Waiting for index creation to complete...")
        time.sleep(20)
        
        # Verify index was created
        if index_name in pc.list_indexes().names():
            logger.info("Pinecone index reset completed successfully")
            return True
        else:
            logger.error("Failed to create Pinecone index")
            return False
        
    except Exception as e:
        logger.error(f"Error resetting Pinecone: {e}")
        return False

def reset_neo4j(uri, username, password):
    """
    Reset Neo4j database by deleting all nodes and relationships.
    
    Args:
        uri: Neo4j URI
        username: Neo4j username
        password: Neo4j password
        
    Returns:
        bool: True if successful
    """
    if not NEO4J_AVAILABLE:
        logger.error("Neo4j Python driver not installed. Run 'pip install neo4j' to enable Neo4j reset.")
        return False
        
    try:
        logger.info("Connecting to Neo4j database")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connection
        driver.verify_connectivity()
        
        with driver.session() as session:
            # Delete all nodes and relationships
            logger.info("Deleting all nodes and relationships")
            session.run("MATCH (n) DETACH DELETE n")
            
            # Verify database is empty
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            
            if count == 0:
                logger.info("Neo4j database reset completed successfully")
                return True
            else:
                logger.error(f"Neo4j database still contains {count} nodes after reset")
                return False
        
    except Exception as e:
        logger.error(f"Error resetting Neo4j database: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Reset databases for government data pipeline')
    parser.add_argument('--supabase-url', default=os.getenv('SUPABASE_URL'), help='Supabase project URL')
    parser.add_argument('--supabase-key', default=os.getenv('SUPABASE_KEY'), help='Supabase API key')
    parser.add_argument('--pinecone-key', default=os.getenv('PINECONE_API_KEY'), help='Pinecone API key')
    parser.add_argument('--pinecone-index', default=os.getenv('PINECONE_INDEX_NAME', 'govt-scrape-index'), help='Pinecone index name')
    parser.add_argument('--neo4j-uri', default=os.getenv('NEO4J_URI'), help='Neo4j URI')
    parser.add_argument('--neo4j-user', default=os.getenv('NEO4J_USERNAME'), help='Neo4j username')
    parser.add_argument('--neo4j-pass', default=os.getenv('NEO4J_PASSWORD'), help='Neo4j password')
    parser.add_argument('--skip-supabase', action='store_true', help='Skip Supabase reset')
    parser.add_argument('--skip-pinecone', action='store_true', help='Skip Pinecone reset')
    parser.add_argument('--skip-neo4j', action='store_true', help='Skip Neo4j reset')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without making changes')
    
    args = parser.parse_args()
    
    # Check if dry run
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        
        if not args.skip_supabase:
            if args.supabase_url and args.supabase_key:
                logger.info(f"Would reset Supabase at {args.supabase_url}")
            else:
                logger.warning("Supabase credentials missing")
        
        if not args.skip_pinecone:
            if args.pinecone_key and args.pinecone_index:
                logger.info(f"Would reset Pinecone index {args.pinecone_index}")
            else:
                logger.warning("Pinecone credentials or index name missing")
        
        if not args.skip_neo4j:
            if args.neo4j_uri and args.neo4j_user and args.neo4j_pass:
                logger.info(f"Would reset Neo4j database at {args.neo4j_uri}")
            else:
                logger.warning("Neo4j credentials missing")
        
        return
    
    # Confirm action
    if not args.skip_supabase or not args.skip_pinecone or not args.skip_neo4j:
        print("\n⚠️  WARNING: This will delete ALL data in the selected databases! ⚠️\n")
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        
        if confirm.lower() != "yes":
            logger.info("Operation cancelled by user")
            return
    
    # Reset Supabase
    if not args.skip_supabase:
        if args.supabase_url and args.supabase_key:
            if reset_supabase(args.supabase_url, args.supabase_key):
                logger.info("Supabase reset completed successfully")
            else:
                logger.error("Supabase reset failed")
        else:
            logger.error("Supabase URL and key required for reset")
    
    # Reset Pinecone
    if not args.skip_pinecone:
        if args.pinecone_key and args.pinecone_index:
            if reset_pinecone(args.pinecone_key, args.pinecone_index):
                logger.info("Pinecone reset completed successfully")
            else:
                logger.error("Pinecone reset failed")
        else:
            logger.error("Pinecone API key and index name required for reset")
    
    # Reset Neo4j
    if not args.skip_neo4j:
        if args.neo4j_uri and args.neo4j_user and args.neo4j_pass:
            if reset_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_pass):
                logger.info("Neo4j reset completed successfully")
            else:
                logger.error("Neo4j reset failed")
        else:
            logger.error("Neo4j URI, username, and password required for reset")
    
    logger.info("Reset operations completed")

if __name__ == "__main__":
    main()