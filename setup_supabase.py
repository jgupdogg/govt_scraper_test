#!/usr/bin/env python3
"""
Setup script for the government data pipeline Supabase integration.
Creates tables and RLS policies in Supabase.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from supabase import create_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# SQL for creating tables and policies
SETUP_SQL = """
-- Sources table
CREATE TABLE IF NOT EXISTS govt_sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    base_url TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Subsources table
CREATE TABLE IF NOT EXISTS govt_subsources (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES govt_sources(id),
    name VARCHAR(255) NOT NULL,
    url_pattern TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Documents table
CREATE TABLE IF NOT EXISTS govt_documents (
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
CREATE INDEX IF NOT EXISTS idx_govt_documents_content_tsvector
ON govt_documents USING gin(to_tsvector('english', content));

-- Set up RLS (Row Level Security) for the tables
ALTER TABLE govt_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE govt_subsources ENABLE ROW LEVEL SECURITY;
ALTER TABLE govt_documents ENABLE ROW LEVEL SECURITY;

-- Create policies for allowing access from service role
CREATE POLICY "Allow service role full access to govt_sources"
    ON govt_sources FOR ALL
    USING (auth.role() = 'service_role');
    
CREATE POLICY "Allow service role full access to govt_subsources"
    ON govt_subsources FOR ALL
    USING (auth.role() = 'service_role');
    
CREATE POLICY "Allow service role full access to govt_documents"
    ON govt_documents FOR ALL
    USING (auth.role() = 'service_role');

-- Create stored procedure for full-text search
CREATE OR REPLACE FUNCTION search_documents(search_query TEXT, max_results INT)
RETURNS TABLE (
  id INT,
  url TEXT,
  title TEXT,
  source_name TEXT,
  subsource_name TEXT,
  highlight TEXT,
  rank FLOAT
) AS $$
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
$$ LANGUAGE plpgsql;
"""

def setup_supabase(supabase_url, supabase_key):
    """
    Set up Supabase tables and policies.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Initializing Supabase client")
        supabase = create_client(supabase_url, supabase_key)
        
        # Note: We can't directly execute SQL commands using the Python client
        # Instead, provide instructions to run SQL in the Supabase dashboard
        logger.info("Cannot automatically create tables through Python client.")
        logger.info("Please execute the following SQL in the Supabase SQL Editor:")
        print("\n" + "-"*80)
        print("SQL TO RUN IN SUPABASE DASHBOARD:")
        print("-"*80)
        print(SETUP_SQL)
        print("-"*80 + "\n")
        
        # Attempt to verify if tables exist (this doesn't create them)
        logger.info("Checking if tables already exist...")
        
        # Create storage bucket for documents if needed
        logger.info("Setting up storage bucket")
        try:
            # Check if bucket exists
            buckets = supabase.storage.list_buckets()
            bucket_exists = any(b['name'] == 'documents' for b in buckets)
            
            if not bucket_exists:
                logger.info("Creating 'documents' storage bucket")
                supabase.storage.create_bucket("documents", {"public": False})
            
            logger.info("Storage bucket setup complete")
        except Exception as e:
            logger.error(f"Error setting up storage bucket: {e}")
            
        # Verify setup
        logger.info("Verifying setup")
        try:
            # Check if tables exist (this doesn't create them, just verifies)
            try:
                result = supabase.table("govt_documents").select("id").limit(1).execute()
                logger.info("govt_documents table exists")
                
                result = supabase.table("govt_sources").select("id").limit(1).execute()
                logger.info("govt_sources table exists")
                
                result = supabase.table("govt_subsources").select("id").limit(1).execute()
                logger.info("govt_subsources table exists")
                
                # Test full-text search function if it exists
                try:
                    result = supabase.rpc("search_documents", {"search_query": "test", "max_results": 1}).execute()
                    logger.info("Full-text search function exists")
                except Exception as e:
                    logger.warning(f"Full-text search function may not exist yet: {e}")
                    logger.info("Please run the SQL above in the Supabase SQL Editor to create it")
            except Exception as e:
                logger.warning(f"Tables may not exist yet: {e}")
                logger.info("Please run the SQL above in the Supabase SQL Editor to create the tables")
            
            logger.info("Supabase setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error verifying setup: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error setting up Supabase: {e}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Set up Supabase for government data pipeline')
    parser.add_argument('--supabase-url', default=os.getenv('SUPABASE_URL'), help='Supabase project URL')
    parser.add_argument('--supabase-key', default=os.getenv('SUPABASE_KEY'), help='Supabase API key')
    parser.add_argument('--print-sql-only', action='store_true', help='Only print the SQL, do not attempt to connect to Supabase')
    
    args = parser.parse_args()
    
    # Check if only printing SQL
    if args.print_sql_only:
        print("\n" + "-"*80)
        print("SQL TO RUN IN SUPABASE DASHBOARD:")
        print("-"*80)
        print(SETUP_SQL)
        print("-"*80 + "\n")
        logger.info("SQL printed. Copy and run this in the Supabase SQL Editor.")
        return
    
    # Check Supabase credentials
    if not args.supabase_url or not args.supabase_key:
        logger.error("Supabase URL and key not provided. Set SUPABASE_URL and SUPABASE_KEY environment variables or use --supabase-url and --supabase-key")
        sys.exit(1)
    
    if setup_supabase(args.supabase_url, args.supabase_key):
        logger.info("Supabase setup preparation completed successfully")
        logger.info("Remember to run the SQL commands in the Supabase SQL Editor to complete setup")
    else:
        logger.error("Supabase setup preparation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()