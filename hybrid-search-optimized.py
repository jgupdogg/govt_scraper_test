#!/usr/bin/env python3
"""
Optimized Hybrid Search Implementation for Government Data Pipeline.
Combines vector search and knowledge graph while fetching full summaries from Supabase.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import required components
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from knowledge_graph_manager import KnowledgeGraphManager
    from core import SupabaseManager  # Import your SupabaseManager
except ImportError as e:
    logger.error(f"Could not import required libraries: {e}")
    logger.error("Make sure all dependencies are installed and PYTHONPATH is set correctly.")
    sys.exit(1)

class OptimizedHybridSearch:
    """
    Optimized hybrid search engine that combines vector search and knowledge graph querying,
    then fetches full summaries from Supabase.
    """
    
    def __init__(
        self,
        anthropic_api_key: str = None,
        openai_api_key: str = None,
        pinecone_api_key: str = None,
        pinecone_index: str = None,
        pinecone_namespace: str = None,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        supabase_url: str = None,
        supabase_key: str = None
    ):
        """Initialize the hybrid search engine with API keys and connection details."""
        # Store configuration from parameters or environment variables
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_index = pinecone_index or os.getenv("PINECONE_INDEX_NAME", "govt-scrape-index")
        self.pinecone_namespace = pinecone_namespace or os.getenv("PINECONE_NAMESPACE", "govt-content")
        
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        # Validate required credentials
        self._validate_credentials()
        
        # Initialize components lazily when needed
        self._llm = None
        self._embedding_model = None
        self._vector_store = None
        self._kg_manager = None
        self._db_manager = None
        
    def _validate_credentials(self):
        """Validate that all required credentials are available."""
        missing = []
        
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.pinecone_index:
            missing.append("PINECONE_INDEX_NAME")
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_username:
            missing.append("NEO4J_USERNAME")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_key:
            missing.append("SUPABASE_KEY")
        
        if missing:
            raise ValueError(f"Missing required credentials: {', '.join(missing)}")
    
    def _init_llm(self):
        """Initialize LLM if not already done."""
        if self._llm is None:
            self._llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                anthropic_api_key=self.anthropic_api_key,
                temperature=0.3
            )
            logger.info("Initialized Anthropic Claude model")
    
    def _init_embedding_model(self):
        """Initialize embedding model if not already done."""
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )
            logger.info("Initialized OpenAI embedding model")
    
    def _init_vector_store(self):
        """Initialize Pinecone vector store if not already done."""
        if self._vector_store is None:
            # Initialize embedding model first
            self._init_embedding_model()
            
            # Set up Pinecone environment
            import pinecone
            pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
            
            # Initialize the vector store
            self._vector_store = PineconeVectorStore(
                index_name=self.pinecone_index,
                embedding=self._embedding_model,
                text_key="content",
                namespace=self.pinecone_namespace
            )
            
            logger.info(f"Initialized Pinecone vector store with index: {self.pinecone_index}")
    
    def _init_kg_manager(self):
        """Initialize knowledge graph manager if not already done."""
        if self._kg_manager is None:
            self._kg_manager = KnowledgeGraphManager(
                uri=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
            logger.info("Initialized Neo4j knowledge graph manager")
    
    def _init_db_manager(self):
        """Initialize Supabase database manager if not already done."""
        if self._db_manager is None:
            self._db_manager = SupabaseManager(
                self.supabase_url,
                self.supabase_key
            )
            logger.info("Initialized Supabase database manager")
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entities from the query for knowledge graph search.
        
        Args:
            query: Search query
            
        Returns:
            List of extracted entity names
        """
        self._init_llm()
        
        # Create a prompt to extract entities
        entities_prompt = f"""
        Extract all entities (people, organizations, agencies, policies, programs, laws, etc.) 
        mentioned in this query. Return ONLY the entity names as a comma-separated list.
        If no entities are found, return "NONE".
        
        Query: {query}
        
        Entities:
        """
        
        try:
            # Get response from Claude
            response = self._llm.invoke(entities_prompt)
            extracted_text = response.content.strip()
            
            # Parse comma-separated entities
            if extracted_text.upper() == "NONE":
                return []
            
            entities = [
                entity.strip() for entity in extracted_text.split(',')
                if entity.strip() and entity.strip().lower() not in ["none", "n/a", "no entities"]
            ]
            
            # Also check for line breaks
            if not entities:
                entities = [
                    entity.strip() for entity in extracted_text.split('\n')
                    if entity.strip() and entity.strip().lower() not in ["none", "n/a", "no entities"]
                ]
            
            logger.info(f"Extracted entities from query: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector search using embeddings.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of search results with basic metadata
        """
        self._init_vector_store()
        
        try:
            logger.info(f"Performing vector search for: '{query}'")
            
            # Use similarity_search_with_score to get documents and scores
            results = self._vector_store.similarity_search_with_score(query, k=k)
            
            # Format results - only include basic metadata, not full summaries
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "doc_id": doc.metadata.get("doc_id", None),
                    "title": doc.metadata.get("title", "Untitled"),
                    "url": doc.metadata.get("url", ""),
                    "source": doc.metadata.get("source", ""),
                    "subsource": doc.metadata.get("subsource", ""),
                    "similarity_score": score,
                    "search_type": "vector"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def knowledge_graph_search(self, entities: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents related to entities using the knowledge graph.
        
        Args:
            entities: List of entity names
            limit: Maximum number of results
            
        Returns:
            List of document results with basic metadata
        """
        if not entities:
            logger.info("No entities provided for knowledge graph search")
            return []
        
        self._init_kg_manager()
        
        try:
            logger.info(f"Performing knowledge graph search for entities: {entities}")
            
            results = []
            for entity in entities:
                # Use KnowledgeGraphManager to find related documents
                entity_results = self._kg_manager.search_related_documents(
                    [entity], 
                    limit=limit
                )
                
                logger.info(f"Found {len(entity_results)} documents related to entity '{entity}' in knowledge graph")
                
                # Add search type and entity information
                for result in entity_results:
                    result["search_type"] = "knowledge_graph"
                    result["matched_entity"] = entity
                
                results.extend(entity_results)
            
            logger.info(f"Total knowledge graph results: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in knowledge graph search: {e}", exc_info=True)
            return []
    
    def fetch_summaries_from_supabase(self, doc_ids: List[str], urls: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch full document summaries from Supabase by IDs and/or URLs.
        
        Args:
            doc_ids: List of document IDs
            urls: List of document URLs as fallback
            
        Returns:
            List of documents with full summaries
        """
        self._init_db_manager()
        
        
        try:
            if not doc_ids and not urls:
                return []
            
            documents = []
            
            # Try to fetch by IDs first
            if doc_ids:
                logger.info(f"Fetching {len(doc_ids)} document summaries from Supabase by IDs")
                id_documents = self._db_manager.get_documents_by_ids(doc_ids)
                documents.extend(id_documents)
                logger.info(f"Retrieved {len(id_documents)} document summaries from Supabase by IDs")
            
            # If URLs are provided, fetch documents that weren't found by ID
            if urls:
                # Find URLs that we haven't already got documents for
                found_urls = {doc.get("url") for doc in documents if doc.get("url")}
                missing_urls = [url for url in urls if url and url not in found_urls]
                
                if missing_urls:
                    logger.info(f"Fetching {len(missing_urls)} document summaries from Supabase by URLs")
                    # We need a custom Supabase query for this
                    try:
                        # Create a query that uses 'in' operator for URLs
                        result = self._db_manager.supabase.table("govt_documents").select(
                            "id", "title", "url", "summary", "source_name", "subsource_name", "content_hash"
                        ).in_("url", missing_urls).execute()
                        
                        if result.data:
                            url_documents = result.data
                            documents.extend(url_documents)
                            logger.info(f"Retrieved {len(url_documents)} document summaries from Supabase by URLs")
                        else:
                            logger.warning(f"No documents found in Supabase for the provided URLs")
                    except Exception as e:
                        logger.error(f"Error getting documents by URLs: {e}")
            
            logger.info(f"Retrieved {len(documents)} total document summaries from Supabase")
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching document summaries: {e}")
            return []
    
    def hybrid_search(
        self, 
        query: str, 
        limit: int = 10, 
        vector_weight: float = 0.5,
        merge_method: str = "weighted",
        debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with optimized summary fetching.
        
        Args:
            query: Search query
            limit: Maximum number of results
            vector_weight: Weight for vector search results (0.0 to 1.0)
            merge_method: How to combine results ('interleave', 'weighted', or 'separate')
            debug: Show additional debug information
            
        Returns:
            List of search results with full summaries
        """
        # Extract entities for knowledge graph search
        entities = self.extract_entities_from_query(query)
        
        # Perform vector search
        vector_results = self.vector_search(query, k=limit)
        
        # Perform knowledge graph search if entities were found
        graph_results = []
        if entities:
            graph_results = self.knowledge_graph_search(entities, limit=limit)
        
        # Log search result counts
        if debug:
            logger.debug(f"Vector search returned {len(vector_results)} results")
            logger.debug(f"Knowledge graph search returned {len(graph_results)} results")
        
        # If one search method returned no results, return results from the other method
        if not vector_results and not graph_results:
            logger.info("No results from either search method")
            return []
        
        if not vector_results:
            logger.info("No vector search results, using only knowledge graph results")
            result_docs = graph_results
        elif not graph_results:
            logger.info("No knowledge graph results, using only vector search results")
            result_docs = vector_results
        else:
            # Combine results based on the specified method
            if merge_method == "separate":
                logger.info("Using separate merge method, returning results separately")
                # Keep results separate - handle this at a higher level
                return {
                    "vector_results": vector_results,
                    "graph_results": graph_results
                }
            
            elif merge_method == "weighted":
                logger.info(f"Using weighted merge method with vector_weight={vector_weight}")
                # Combine and rank by weighted scores
                combined_results = []
                
                # Process vector results
                vector_doc_ids = set()
                for result in vector_results:
                    doc_id = result.get("doc_id")
                    if not doc_id:
                        continue
                        
                    vector_doc_ids.add(doc_id)
                    # Convert similarity score to a normalized score (higher is better)
                    vector_score = 1.0 - min(result.get("similarity_score", 0), 1.0)
                    combined_results.append({
                        **result,
                        "combined_score": vector_score * vector_weight
                    })
                
                # Process graph results
                for result in graph_results:
                    doc_id = result.get("doc_id")
                    if not doc_id:
                        continue
                        
                    # Use relevance score directly
                    graph_score = result.get("relevance_score", 0.5)
                    
                    # Check if this document is already in combined results
                    existing_idx = next(
                        (i for i, r in enumerate(combined_results) if r.get("doc_id") == doc_id),
                        None
                    )
                    
                    if existing_idx is not None:
                        # Update existing result
                        combined_results[existing_idx]["knowledge_graph"] = True
                        combined_results[existing_idx]["graph_context"] = result.get("context", "")
                        combined_results[existing_idx]["matched_entity"] = result.get("matched_entity", "")
                        combined_results[existing_idx]["combined_score"] += graph_score * (1 - vector_weight)
                    else:
                        # Add new result
                        combined_results.append({
                            **result,
                            "knowledge_graph": True,
                            "combined_score": graph_score * (1 - vector_weight)
                        })
                
                # Sort by combined score (descending)
                combined_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
                
                # Limit results
                result_docs = combined_results[:limit]
                
            else:  # Default to interleave
                logger.info("Using interleave merge method")
                # Interleave results, removing duplicates
                combined_results = []
                seen_doc_ids = set()
                
                # Get iterators
                vector_iter = iter(vector_results)
                graph_iter = iter(graph_results)
                
                # Interleave until we have enough results or run out
                while len(combined_results) < limit:
                    # Try to get next vector result
                    try:
                        vector_result = next(vector_iter)
                        doc_id = vector_result.get("doc_id")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            combined_results.append(vector_result)
                            
                            if len(combined_results) >= limit:
                                break
                    except StopIteration:
                        pass
                    
                    # Try to get next graph result
                    try:
                        graph_result = next(graph_iter)
                        doc_id = graph_result.get("doc_id")
                        if doc_id and doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            combined_results.append(graph_result)
                            
                            if len(combined_results) >= limit:
                                break
                    except StopIteration:
                        pass
                    
                    # If both iterators are exhausted, break the loop
                    if len(seen_doc_ids) >= len(vector_results) + len(graph_results):
                        break
                
                result_docs = combined_results
        

            # Extract doc IDs from the results and collect URLs as fallback
            doc_ids = [doc.get("doc_id") for doc in result_docs if doc.get("doc_id")]
            doc_urls = [doc.get("url") for doc in result_docs if doc.get("url")]

            if not doc_ids and not doc_urls:
                logger.warning("No valid document IDs or URLs found in search results")
                return result_docs

            # Fetch full document summaries from Supabase
            documents = self.fetch_summaries_from_supabase(doc_ids, doc_urls)

            # Create a lookup dictionary for quick access
            doc_id_lookup = {str(doc.get("id")): doc for doc in documents if doc.get("id")}
            doc_url_lookup = {doc.get("url"): doc for doc in documents if doc.get("url")}

            # Enhance results with full summaries
            for result in result_docs:
                doc_id = result.get("doc_id")
                url = result.get("url")
                
                # Try to look up by ID first
                if doc_id and str(doc_id) in doc_id_lookup:
                    # Add summary and any other fields from Supabase
                    doc = doc_id_lookup[str(doc_id)]
                    result["summary"] = doc.get("summary")
                    
                    # If title wasn't already set, get it from Supabase
                    if not result.get("title") and doc.get("title"):
                        result["title"] = doc.get("title")
                        
                # Fall back to URL lookup if ID lookup failed
                elif url and url in doc_url_lookup:
                    doc = doc_url_lookup[url]
                    result["summary"] = doc.get("summary")
                    
                    # If ID wasn't set, get it from Supabase
                    if not result.get("doc_id") and doc.get("id"):
                        result["doc_id"] = doc.get("id")
                        
                    # If title wasn't already set, get it from Supabase
                    if not result.get("title") and doc.get("title"):
                        result["title"] = doc.get("title")
                    
                    logger.info(f"Returning {len(result_docs)} results with full summaries")
                    return result_docs
                
    def format_results(self, results, output_format="text", show_source_counts=True):
        """
        Format search results for display.
        
        Args:
            results: Search results
            output_format: Format for output ('text', 'json', or 'table')
            show_source_counts: Show counts of results by search type
            
        Returns:
            Formatted results string
        """
        if not results:
            return "No results found."
        
        if output_format == "json":
            # Return JSON format
            return json.dumps(results, indent=2)
        
        elif output_format == "table":
            # Format as table
            table_data = []
            for i, result in enumerate(results, 1):
                search_type = result.get("search_type", "unknown")
                score_type = "Similarity" if search_type == "vector" else "Relevance"
                score_value = result.get("similarity_score" if search_type == "vector" else "relevance_score", 0)
                
                context = ""
                if search_type == "knowledge_graph":
                    context = f"Entity: {result.get('matched_entity', 'Unknown')}"
                    if result.get("context"):
                        context += f" - {result.get('context')}"
                
                # Get summary snippet
                summary = result.get("summary", "")
                summary_snippet = summary[:50] + "..." if summary and len(summary) > 50 else summary
                
                table_data.append([
                    i,
                    result.get("title", "Untitled")[:40] + ("..." if len(result.get("title", "")) > 40 else ""),
                    search_type,
                    f"{score_value:.3f}",
                    context[:40] + ("..." if len(context) > 40 else ""),
                    summary_snippet
                ])
            
            return tabulate(table_data, headers=["#", "Title", "Search Type", "Score", "Context", "Summary"], tablefmt="grid")
        
        else:  # Default to text
            # Format as human-readable text
            output = [f"===== Found {len(results)} Results ====="]
            
            # Show counts by search type if requested
            if show_source_counts:
                vector_count = sum(1 for r in results if r.get("search_type") == "vector")
                kg_count = sum(1 for r in results if r.get("search_type") == "knowledge_graph")
                output.append(f"Vector Search Results: {vector_count}")
                output.append(f"Knowledge Graph Results: {kg_count}")
                output.append("")
            
            for i, result in enumerate(results, 1):
                search_type = result.get("search_type", "unknown")
                
                output.append(f"[{i}] {result.get('title', 'Untitled')}")
                output.append(f"    Source: {result.get('source', result.get('source_name', ''))}")
                output.append(f"    URL: {result.get('url', '')}")
                
                if search_type == "vector":
                    output.append(f"    Search Type: Vector (Similarity: {result.get('similarity_score', 0):.3f})")
                else:
                    output.append(f"    Search Type: Knowledge Graph (Relevance: {result.get('relevance_score', 0):.3f})")
                    output.append(f"    Matched Entity: {result.get('matched_entity', 'Unknown')}")
                    if result.get("context"):
                        output.append(f"    Context: {result.get('context', '')}")
                
                # Add combined score if available
                if "combined_score" in result:
                    output.append(f"    Combined Score: {result.get('combined_score', 0):.3f}")
                
                # Add summary
                if result.get("summary"):
                    summary = result.get("summary", "")
                    output.append(f"    Summary: {summary[:300]}..." if len(summary) > 300 else f"    Summary: {summary}")
                
                output.append("")  # Empty line between results
            
            return "\n".join(output)


def main():
    """Main function for the script."""
    parser = argparse.ArgumentParser(description='Optimized Hybrid Search for Government Data Pipeline')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of results to return')
    parser.add_argument('--vector-weight', type=float, default=0.5, help='Weight for vector search results (0.0 to 1.0)')
    parser.add_argument('--merge-method', choices=['interleave', 'weighted', 'separate'], default='weighted',
                      help='Method for merging results: interleave, weighted ranking, or separate')
    parser.add_argument('--format', choices=['text', 'json', 'table'], default='text',
                      help='Output format: text, JSON, or table')
    parser.add_argument('--output-file', help='Write results to file instead of stdout')
    parser.add_argument('--debug', action='store_true', help='Show detailed debugging information')
    parser.add_argument('--no-counts', action='store_true', help="Don't show result counts by search type")
    
    # Options for specific search types
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--vector-only', action='store_true', help='Only perform vector search')
    group.add_argument('--graph-only', action='store_true', help='Only perform knowledge graph search')
    
    args = parser.parse_args()
    
    try:
        # Configure logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            for handler in logging.getLogger().handlers:
                handler.setLevel(logging.DEBUG)
        
        # Initialize search engine
        search_engine = OptimizedHybridSearch()
        
        # Perform search based on options
        if args.vector_only:
            logger.info("Performing vector-only search")
            vector_results = search_engine.vector_search(args.query, k=args.limit)
            
            # Fetch summaries
            doc_ids = [r.get("doc_id") for r in vector_results if r.get("doc_id")]
            documents = search_engine.fetch_summaries_from_supabase(doc_ids)
            doc_lookup = {str(doc.get("id")): doc for doc in documents}
            
            # Enhance results with summaries
            for result in vector_results:
                doc_id = result.get("doc_id")
                if doc_id and str(doc_id) in doc_lookup:
                    result["summary"] = doc_lookup[str(doc_id)].get("summary")
            
            results = vector_results
            
        elif args.graph_only:
            logger.info("Performing knowledge graph-only search")
            entities = search_engine.extract_entities_from_query(args.query)
            if not entities:
                print("No entities found in query for knowledge graph search.")
                sys.exit(1)
                
            graph_results = search_engine.knowledge_graph_search(entities, limit=args.limit)
            
            # Fetch summaries
            doc_ids = [r.get("doc_id") for r in graph_results if r.get("doc_id")]
            documents = search_engine.fetch_summaries_from_supabase(doc_ids)
            doc_lookup = {str(doc.get("id")): doc for doc in documents}
            
            # Enhance results with summaries
            for result in graph_results:
                doc_id = result.get("doc_id")
                if doc_id and str(doc_id) in doc_lookup:
                    result["summary"] = doc_lookup[str(doc_id)].get("summary")
            
            results = graph_results
            
        else:
            logger.info("Performing hybrid search")
            results = search_engine.hybrid_search(
                args.query, 
                limit=args.limit, 
                vector_weight=args.vector_weight,
                merge_method=args.merge_method,
                debug=args.debug
            )
        
        # Format results
        formatted_results = search_engine.format_results(
            results, 
            output_format=args.format,
            show_source_counts=not args.no_counts
        )
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(formatted_results)
            print(f"Results written to {args.output_file}")
        else:
            print(formatted_results)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()