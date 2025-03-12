"""
Knowledge Graph Manager for government data pipeline.
Handles entity extraction, normalization, and storage in Neo4j.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """
    Manages interactions with Neo4j knowledge graph.
    Handles entity resolution and relationship mapping.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (bolt://localhost:7687 or neo4j+s://xxx.databases.neo4j.io)
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        try:
            self._connect()
            self._setup_database()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise
    
    def _setup_database(self):
        """Create constraints for the database."""
        with self.driver.session() as session:
            # Create constraints to ensure uniqueness
            try:
                # Create constraints (Neo4j 4.x+ syntax)
                session.run("""
                CREATE CONSTRAINT entity_constraint IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE
                """)
                
                session.run("""
                CREATE CONSTRAINT document_constraint IF NOT EXISTS
                FOR (d:Document) REQUIRE d.url IS UNIQUE
                """)
                
                logger.info("Database constraints created successfully")
            except Neo4jError as e:
                # If constraint already exists or older Neo4j version
                if "already exists" in str(e) or "SyntaxError" in str(e):
                    # Try older Neo4j syntax
                    try:
                        session.run("""
                        CREATE CONSTRAINT ON (e:Entity) ASSERT e.canonical_name IS UNIQUE
                        """)
                        
                        session.run("""
                        CREATE CONSTRAINT ON (d:Document) ASSERT d.url IS UNIQUE
                        """)
                        
                        logger.info("Database constraints created successfully (legacy syntax)")
                    except Neo4jError as e2:
                        if "already exists" not in str(e2):
                            logger.warning(f"Could not create constraints: {e2}")
                else:
                    logger.warning(f"Could not create constraints: {e}")
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def add_document_node(self, document):
        """
        Add a document node to the knowledge graph.
        
        Args:
            document: Document object to add
        """
        with self.driver.session() as session:
            # Create document node
            session.run("""
            MERGE (d:Document {url: $url})
            ON CREATE SET 
                d.title = $title,
                d.source_name = $source_name,
                d.subsource_name = $subsource_name,
                d.created_at = $created_at,
                d.doc_id = $doc_id
            ON MATCH SET
                d.title = $title,
                d.updated_at = $updated_at
            """, {
                "url": document.url,
                "title": document.title,
                "source_name": document.source_name,
                "subsource_name": document.subsource_name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "doc_id": str(document.doc_id)
            })
    
    def add_entities_and_relationships(self, document, entities, relationships):
        """
        Add entities and relationships to the knowledge graph.
        
        Args:
            document: Source document
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
        """
        with self.driver.session() as session:
            # Add entities
            for entity in entities:
                # Skip if missing critical information
                if not entity.get("canonical_name") or not entity.get("entity_type"):
                    continue
                
                # Create canonical entity node
                session.run("""
                MERGE (e:Entity {canonical_name: $canonical_name})
                ON CREATE SET 
                    e.type = $entity_type,
                    e.created_at = $created_at
                ON MATCH SET
                    e.type = $entity_type,
                    e.updated_at = $updated_at
                """, {
                    "canonical_name": entity["canonical_name"],
                    "entity_type": entity["entity_type"],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
                
                # Create mention node and link to both canonical entity and document
                mention_text = entity.get("mention", entity["canonical_name"])
                session.run("""
                MATCH (e:Entity {canonical_name: $canonical_name})
                MATCH (d:Document {url: $doc_url})
                MERGE (m:Mention {
                    text: $mention,
                    document_url: $doc_url
                })
                MERGE (m)-[:REFERS_TO]->(e)
                MERGE (m)-[:APPEARS_IN]->(d)
                """, {
                    "canonical_name": entity["canonical_name"],
                    "mention": mention_text,
                    "doc_url": document.url
                })
            
            # Add relationships
            for relationship in relationships:
                # Skip if missing critical information
                if not relationship.get("source_canonical") or not relationship.get("target_canonical") or not relationship.get("relation"):
                    continue
                
                # Create relationship between canonical entities
                # FIX: Create relationship properties including document reference, not a relationship to document
                session.run("""
                MATCH (source:Entity {canonical_name: $source_canonical})
                MATCH (target:Entity {canonical_name: $target_canonical})
                MATCH (d:Document {url: $doc_url})
                MERGE (source)-[r:`$relation`]->(target)
                SET r.updated_at = $updated_at,
                    r.document_url = $doc_url,
                    r.document_id = $doc_id
                """.replace("`$relation`", f"`{relationship['relation']}`"), {
                    "source_canonical": relationship["source_canonical"],
                    "target_canonical": relationship["target_canonical"],
                    "relation": relationship["relation"],
                    "doc_url": document.url,
                    "doc_id": str(document.doc_id) if document.doc_id else None,
                    "updated_at": datetime.now().isoformat()
                })
                
                # REMOVED: The problematic line was here, trying to create a relationship FROM a relationship
                # MERGE (r)-[:MENTIONED_IN]->(d)
                
                # ALTERNATIVE: Create explicit mention of the relationship if needed
                session.run("""
                MATCH (source:Entity {canonical_name: $source_canonical})
                MATCH (target:Entity {canonical_name: $target_canonical})
                MATCH (d:Document {url: $doc_url})
                MERGE (rm:RelationMention {
                    source: $source_canonical,
                    target: $target_canonical,
                    relation: $relation,
                    document_url: $doc_url
                })
                MERGE (rm)-[:APPEARS_IN]->(d)
                """, {
                    "source_canonical": relationship["source_canonical"],
                    "target_canonical": relationship["target_canonical"],
                    "relation": relationship["relation"],
                    "doc_url": document.url
                })
    
    def search_related_documents(self, entity_names, limit=5, max_path_length=2):
        """
        Find documents related to specified entities using knowledge graph.
        
        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of results to return
            max_path_length: Maximum path length for graph traversal
            
        Returns:
            List of document dictionaries with relationship context
        """
        results = []
        
        with self.driver.session() as session:
            for entity_name in entity_names:
                # First try exact canonical name match
                query = """
                MATCH (e:Entity)
                WHERE e.canonical_name = $entity_name OR e.canonical_name CONTAINS $entity_name
                MATCH path = (e)-[*1..2]-(d:Document)
                RETURN d.url as url, d.title as title, d.source_name as source_name, 
                       d.subsource_name as subsource_name, d.doc_id as doc_id,
                       [r IN relationships(path) | type(r)] as relationship_types,
                       length(path) as path_length
                ORDER BY path_length
                LIMIT $limit
                """
                result = session.run(query, {
                    "entity_name": entity_name,
                    "limit": limit
                })
                
                canonical_docs = self._process_search_results(result, entity_name)
                results.extend(canonical_docs)
                
                # Then try mention match if we haven't reached the limit
                if len(results) < limit:
                    query = """
                    MATCH (m:Mention)
                    WHERE m.text CONTAINS $entity_name
                    MATCH (m)-[:REFERS_TO]->(e:Entity)
                    MATCH (m)-[:APPEARS_IN]->(d:Document)
                    RETURN d.url as url, d.title as title, d.source_name as source_name, 
                           d.subsource_name as subsource_name, d.doc_id as doc_id,
                           [e.canonical_name] as mentioned_entities,
                           1 as path_length
                    LIMIT $limit
                    """
                    result = session.run(query, {
                        "entity_name": entity_name,
                        "limit": limit - len(results)
                    })
                    
                    mention_docs = self._process_search_results(result, entity_name)
                    results.extend(mention_docs)
            
            # Deduplicate results by URL
            unique_results = []
            seen_urls = set()
            
            for doc in results:
                if doc["url"] not in seen_urls:
                    seen_urls.add(doc["url"])
                    unique_results.append(doc)
                    
                    if len(unique_results) >= limit:
                        break
            
            return unique_results
    
    def _process_search_results(self, result, query_entity):
        """Process Neo4j search results into a standardized format."""
        docs = []
        
        for record in result:
            # Build context information based on available data
            context = ""
            if "relationship_types" in record and record["relationship_types"]:
                relationship_str = " -> ".join(record["relationship_types"])
                context = f"Connected to '{query_entity}' via: {relationship_str}"
            elif "mentioned_entities" in record and record["mentioned_entities"]:
                entities_str = ", ".join(record["mentioned_entities"])
                context = f"Mentions entities: {entities_str}"
            
            docs.append({
                "url": record["url"],
                "title": record["title"],
                "source_name": record["source_name"],
                "subsource_name": record["subsource_name"],
                "doc_id": record["doc_id"],
                "context": context,
                "relevance_score": 1.0 / (record["path_length"] if "path_length" in record else 1)
            })
        
        return docs
    
    def get_entity_info(self, entity_name):
        """
        Get detailed information about an entity.
        
        Args:
            entity_name: Name of the entity to lookup
            
        Returns:
            Dictionary with entity information and relationships
        """
        with self.driver.session() as session:
            # First try canonical name match
            query = """
            MATCH (e:Entity)
            WHERE e.canonical_name = $entity_name OR e.canonical_name CONTAINS $entity_name
            OPTIONAL MATCH (e)-[r]->(other:Entity)
            RETURN e.canonical_name as name, e.type as type,
                  collect(distinct {relation: type(r), target: other.canonical_name, target_type: other.type}) as outgoing_relations
            """
            result = session.run(query, {"entity_name": entity_name})
            
            record = result.single()
            if not record:
                # Try mention match
                query = """
                MATCH (m:Mention)-[:REFERS_TO]->(e:Entity)
                WHERE m.text CONTAINS $entity_name
                OPTIONAL MATCH (e)-[r]->(other:Entity)
                RETURN e.canonical_name as name, e.type as type,
                      collect(distinct {relation: type(r), target: other.canonical_name, target_type: other.type}) as outgoing_relations
                """
                result = session.run(query, {"entity_name": entity_name})
                record = result.single()
            
            if record:
                # Filter out null relationships
                outgoing = [r for r in record["outgoing_relations"] if r["target"] is not None]
                
                # Get incoming relationships
                query = """
                MATCH (other:Entity)-[r]->(e:Entity)
                WHERE e.canonical_name = $entity_name
                RETURN collect(distinct {relation: type(r), source: other.canonical_name, source_type: other.type}) as incoming_relations
                """
                incoming_result = session.run(query, {"entity_name": record["name"]})
                incoming_record = incoming_result.single()
                incoming = incoming_record["incoming_relations"] if incoming_record else []
                
                # Get mentions
                query = """
                MATCH (m:Mention)-[:REFERS_TO]->(e:Entity)
                WHERE e.canonical_name = $entity_name
                RETURN collect(distinct m.text) as mentions
                """
                mentions_result = session.run(query, {"entity_name": record["name"]})
                mentions_record = mentions_result.single()
                mentions = mentions_record["mentions"] if mentions_record else []
                
                return {
                    "name": record["name"],
                    "type": record["type"],
                    "mentions": mentions,
                    "outgoing_relationships": outgoing,
                    "incoming_relationships": incoming
                }
            
            return None
    
    def reset_database(self):
        """Clear all data from the Neo4j database."""
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j database reset complete")

    def get_statistics(self):
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with counts of different node and relationship types
        """
        with self.driver.session() as session:
            # Count entities by type
            entity_query = """
            MATCH (e:Entity)
            RETURN e.type as type, count(e) as count
            """
            entity_result = session.run(entity_query)
            entity_counts = {record["type"]: record["count"] for record in entity_result}
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            """
            rel_result = session.run(rel_query)
            rel_counts = {record["type"]: record["count"] for record in rel_result}
            
            # Count documents
            doc_query = "MATCH (d:Document) RETURN count(d) as count"
            doc_result = session.run(doc_query)
            doc_count = doc_result.single()["count"]
            
            # Count mentions
            mention_query = "MATCH (m:Mention) RETURN count(m) as count"
            mention_result = session.run(mention_query)
            mention_count = mention_result.single()["count"]
            
            return {
                "document_count": doc_count,
                "entity_count": sum(entity_counts.values()),
                "entity_types": entity_counts,
                "mention_count": mention_count,
                "relationship_count": sum(rel_counts.values()),
                "relationship_types": rel_counts
            }