import os
import logging
import json
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document as LCDocument
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from knowledge_graph_manager import KnowledgeGraphManager

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedProcessor:
    """
    Enhanced processor that combines all functionality:
    - LangChain for efficient document processing
    - Knowledge graph entity extraction and storage
    - Vector embeddings for semantic search
    """
    
    def __init__(
        self, 
        anthropic_api_key: str = None, 
        openai_api_key: str = None,
        pinecone_api_key: str = None,
        pinecone_index: str = "govt-scrape-index",
        pinecone_namespace: str = "govt-content",
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None
    ):
        """Initialize the enhanced processor with API keys."""
        # Store configuration
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_index = pinecone_index or os.getenv("PINECONE_INDEX_NAME", "govt-scrape-index")
        self.pinecone_namespace = pinecone_namespace or os.getenv("PINECONE_NAMESPACE", "govt-content")
        
        # Neo4j credentials
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize components lazily when needed
        self._llm = None
        self._embedding_model = None
        self._vector_store = None
        self._kg_manager = None
        
        # Check for knowledge graph availability
        self.kg_available = all([self.neo4j_uri, self.neo4j_username, self.neo4j_password])
        if not self.kg_available:
            logger.warning("Neo4j credentials not complete. Knowledge graph features will be disabled.")
    
    def _init_llm(self):
        """Initialize LLM if not already done."""
        if self._llm is None:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key is required but not provided")
                
            self._llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                anthropic_api_key=self.anthropic_api_key,
                temperature=0.3
            )
            logger.info("Initialized Anthropic Claude model")
    
    def _init_embedding_model(self):
        """Initialize embedding model if not already done."""
        if self._embedding_model is None:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required but not provided")
                
            self._embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )
            logger.info("Initialized OpenAI embedding model")
    
    def _init_vector_store(self):
        """Initialize Pinecone vector store if not already done."""
        if self._vector_store is None:
            try:
                # Initialize embedding model first if needed
                self._init_embedding_model()
                
                if not self.pinecone_api_key:
                    raise ValueError("Pinecone API key is required but not provided")
                
                # Import the recommended langchain_pinecone package
                from langchain_pinecone import PineconeVectorStore
                
                # Set up Pinecone environment
                import pinecone
                pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
                
                # Check if index exists
                existing_indexes = pc.list_indexes().names()
                
                if self.pinecone_index not in existing_indexes:
                    logger.warning(f"Index {self.pinecone_index} not found, creating...")
                    # Create the index with appropriate dimensions for the embedding model
                    from pinecone import ServerlessSpec
                    pc.create_index(
                        name=self.pinecone_index,
                        dimension=1536,  # Dimension for text-embedding-3-small
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-west-2"
                        )
                    )
                
                # Initialize the vector store with LangChain
                self._vector_store = PineconeVectorStore(
                    index_name=self.pinecone_index,
                    embedding=self._embedding_model,
                    text_key="content",  # The key in metadata containing the text to embed
                    namespace=self.pinecone_namespace
                )
                
                logger.info(f"Successfully initialized Pinecone vector store with index: {self.pinecone_index}")
            
            except ImportError as ie:
                logger.error(f"Import error: {ie}. Make sure you have langchain-pinecone package installed.")
                raise
            except Exception as e:
                logger.error(f"Error initializing vector store: {e}", exc_info=True)
                raise
    
    def _init_kg_manager(self):
        """Initialize the knowledge graph manager if not already done and available."""
        if self._kg_manager is None and self.kg_available:
            try:
                self._kg_manager = KnowledgeGraphManager(
                    uri=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password
                )
                logger.info("Successfully initialized Neo4j knowledge graph manager")
                return True
            except Exception as e:
                logger.error(f"Error initializing knowledge graph manager: {e}", exc_info=True)
                self.kg_available = False
                return False
        return self.kg_available and self._kg_manager is not None
    
    def summarize(self, document):
        """
        Generate a structured summary for a document, using the appropriate method
        based on document length.
        
        Args:
            document: Document to summarize
            
        Returns:
            Structured summary text
        """
        if not document.content:
            raise ValueError("Document has no content to summarize")
        
        # Check if document is large enough to need chunking
        is_large_document = len(document.content) > 12000
        
        if is_large_document:
            return self._summarize_large_document(document)
        else:
            return self._summarize_small_document(document)
    
    def _summarize_small_document(self, document):
        """Use direct prompting for smaller documents."""
        self._init_llm()
        
        # Use the structured format prompt
        prompt = f"""Generate a structured summary of the following government website content.

Input:
Title: {document.title}
Source: {document.source_name} - {document.subsource_name}
URL: {document.url}

Content:
{document.content[:8000]}

Output Format:
1. TITLE: A clear, direct title that captures the main topic (no more than 10 words)
2. FACTS: 3-5 bullet points with the most important and relevant information 
3. SENTIMENT: One bullet point expressing the overall sentiment (positive, negative, or neutral) of the content
4. TAGS: 5-7 relevant keywords/tags separated by commas

Example:
TITLE: Federal Grant Program Launches for Rural Communities

• $50 million in federal funding allocated to support infrastructure in rural communities
• Applications will be accepted from April 1 to June 30, 2025
• Eligible counties must have populations under 50,000 residents
• Priority given to projects addressing water quality and broadband access
• Overall sentiment is positive, with program expected to benefit approximately 200 rural counties nationwide

TAGS: rural infrastructure, federal grants, funding opportunity, application deadline, eligibility requirements, water quality, broadband
"""
        
        response = self._llm.invoke(prompt)
        summary = response.content.strip()
        
        return summary
    
    def _summarize_large_document(self, document):
        """
        Use LangChain's map-reduce summarization for larger documents,
        maintaining the structured output format.
        """
        self._init_llm()
        
        # Define map prompt (for individual chunks)
        map_template = """Summarize the key information from this portion of a government document:

{text}

Include the most important facts, statistics, dates, and policies in your summary.
Write in a clear, factual manner.
"""
        
        # Define combine prompt (for final summary)
        combine_template = """Generate a structured summary of this government content based on these extracted details:

{text}

Output Format:
1. TITLE: A clear, direct title that captures the main topic (no more than 10 words)
2. FACTS: 3-5 bullet points with the most important and relevant information 
3. SENTIMENT: One bullet point expressing the overall sentiment (positive, negative, or neutral) of the content
4. TAGS: 5-7 relevant keywords/tags separated by commas

Example:
TITLE: Federal Grant Program Launches for Rural Communities

• $50 million in federal funding allocated to support infrastructure in rural communities
• Applications will be accepted from April 1 to June 30, 2025
• Eligible counties must have populations under 50,000 residents
• Priority given to projects addressing water quality and broadband access
• Overall sentiment is positive, with program expected to benefit approximately 200 rural counties nationwide

TAGS: rural infrastructure, federal grants, funding opportunity, application deadline, eligibility requirements, water quality, broadband
"""
        
        # Create the map-reduce chain
        map_prompt = PromptTemplate.from_template(map_template)
        combine_prompt = PromptTemplate.from_template(combine_template)
        
        # Set up the map-reduce chain
        chain = load_summarize_chain(
            self._llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        # Split the document text into chunks
        text_chunks = self.text_splitter.split_text(document.content)
        
        # Convert to LangChain documents
        lc_docs = [
            LCDocument(
                page_content=chunk,
                metadata={
                    "title": document.title, 
                    "source": document.source_name,
                    "subsource": document.subsource_name,
                    "url": document.url
                }
            ) 
            for chunk in text_chunks
        ]
        
        # Run the chain
        summary = chain.invoke(lc_docs)["output_text"]
        
        return summary
    
    def extract_entities(self, document):
        """
        Extract entities and relationships from document content for knowledge graph.
        
        Args:
            document: Document to extract entities from
            
        Returns:
            Dict with entities and relationships
        """
        self._init_llm()
        
        # Limit content length for extraction
        content = document.content[:6000]
        
        prompt = f"""Extract entities and their relationships from this government document content.
    For each entity, provide both the exact mention in the text AND a canonical (standardized) name.

    Title: {document.title}
    Source: {document.source_name} - {document.subsource_name}

    Content:
    {content}

    IMPORTANT GUIDELINES:
    1. Focus ONLY on meaningful entities like Agencies, People, Organizations, Laws, Programs, Assets, Resources, and Policies.
    2. DO NOT include years, dates, percentages, or numeric values as entities.
    3. DO NOT include general concepts, statistics or measurements as entities.
    4. Each entity should represent a specific named organization, person, program, or policy.

    Format your response as a structured JSON:

    ```json
    {{
    "entities": [
        {{
        "mention": "The exact text as it appears",
        "canonical_name": "The standardized name",
        "entity_type": "Person/Agency/Policy/Program/Law/Organization/etc."
        }}
    ],
    "relationships": [
        {{
        "source_mention": "The source entity mention",
        "source_canonical": "The source entity canonical name",
        "relation": "OVERSEES/IMPLEMENTS/PART_OF/LOCATED_IN/etc.",
        "target_mention": "The target entity mention",
        "target_canonical": "The target entity canonical name"
        }}
    ]
    }}
    ```

    EXAMPLES OF VALID ENTITIES:
    ✓ "Federal Reserve" (Agency)
    ✓ "Jerome Powell" (Person)
    ✓ "Clean Water Act" (Law)
    ✓ "Paycheck Protection Program" (Program)

    EXAMPLES OF INVALID ENTITIES (DO NOT INCLUDE THESE):
    ✗ "2025" (year)
    ✗ "8.9%" (percentage)
    ✗ "Q1" (time period)
    ✗ "inflation" (concept)
    ✗ "$200 million" (monetary value)
    """
        
        try:
            response = self._llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Extract JSON part
            import json
            import re
            
            # First try to find JSON block in markdown format
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without the markdown code block
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    logger.warning("Could not extract JSON from LLM response")
                    return {"entities": [], "relationships": []}
            
            # Clean up the JSON string - fix common issues
            json_str = json_str.strip()
            
            # Fix trailing commas (common JSON error)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Try to parse JSON
            try:
                extraction_result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}. Attempting to recover...")
                
                # Manual parsing for recovery - create basic structure
                extraction_result = {"entities": [], "relationships": []}
                
                # Try to extract entities with regex
                entity_matches = re.findall(r'"mention":\s*"([^"]+)"[^}]+?"canonical_name":\s*"([^"]+)"[^}]+?"entity_type":\s*"([^"]+)"', json_str)
                for mention, canonical, entity_type in entity_matches:
                    extraction_result["entities"].append({
                        "mention": mention,
                        "canonical_name": canonical,
                        "entity_type": entity_type
                    })
                
                # Try to extract relationships with regex
                rel_matches = re.findall(r'"source_canonical":\s*"([^"]+)"[^}]+?"relation":\s*"([^"]+)"[^}]+?"target_canonical":\s*"([^"]+)"', json_str)
                for source, relation, target in rel_matches:
                    extraction_result["relationships"].append({
                        "source_canonical": source,
                        "relation": relation,
                        "target_canonical": target,
                        "source_mention": source,  # Fallback
                        "target_mention": target   # Fallback
                    })
            
            # Additional filtering for numeric values and years
            filtered_entities = []
            for entity in extraction_result.get("entities", []):
                canonical_name = entity.get("canonical_name", "")
                mention = entity.get("mention", "")
                
                # Skip if entity is numeric, a year, or a percentage
                if (re.match(r'^[0-9]+(\.[0-9]+)?%?$', canonical_name) or 
                    re.match(r'^(19|20)\d{2}$', canonical_name) or  # Years like 1999, 2025
                    re.match(r'^Q[1-4]$', canonical_name) or        # Quarters like Q1, Q2
                    canonical_name.lower() in ["inflation", "recession", "recovery", "growth"] or  # Generic economic concepts
                    len(canonical_name) < 3):  # Very short strings
                    logger.debug(f"Filtered out entity: {canonical_name} ({entity.get('entity_type', '')})")
                    continue
                    
                filtered_entities.append(entity)
            
            # Update entities with filtered list
            extraction_result["entities"] = filtered_entities
            
            # Filter relationships that reference filtered entities
            valid_entities = {entity["canonical_name"] for entity in filtered_entities}
            filtered_relationships = []
            
            for rel in extraction_result.get("relationships", []):
                source = rel.get("source_canonical", "")
                target = rel.get("target_canonical", "")
                
                if source in valid_entities and target in valid_entities:
                    filtered_relationships.append(rel)
            
            # Update relationships with filtered list
            extraction_result["relationships"] = filtered_relationships
            
            logger.info(f"Extracted {len(extraction_result.get('entities', []))} entities and {len(extraction_result.get('relationships', []))} relationships")
            
            return extraction_result
        
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "relationships": []}
    
    def store_embedding(self, document) -> str:
        """
        Store document embedding in Pinecone using LangChain.
        
        Args:
            document: Document with summary
            
        Returns:
            Embedding ID
        """
        if not document.summary:
            raise ValueError("Document has no summary to embed")
        
        self._init_vector_store()
        
        # Create a unique ID
        embedding_id = f"gov-{hashlib.md5(document.url.encode()).hexdigest()[:12]}"
        
        try:
            # Create LangChain Document
            lc_doc = LCDocument(
                page_content=document.summary,
                metadata={
                    "url": document.url,
                    "title": document.title,
                    "source": document.source_name, 
                    "subsource": document.subsource_name,
                    "content": document.summary,  # This is used as text_key for embedding
                    "processed_at": datetime.now().isoformat()
                }
            )
            
            # Store in vector store
            ids = self._vector_store.add_documents([lc_doc], ids=[embedding_id])
            
            logger.info(f"Successfully stored document in vector store with ID: {ids[0]}")
            return ids[0]
        
        except Exception as e:
            logger.error(f"Error storing embedding: {e}", exc_info=True)
            raise
    
    def process_document(self, document) -> bool:
        """
        Process a document end-to-end:
        1. Generate summary
        2. Create embedding
        3. Store in vector database
        4. Extract entities and add to knowledge graph
        
        Args:
            document: Document to process
            
        Returns:
            bool: True if successful
        """
        try:
            # Skip if already processed unless forced
            if document.embedding_id and document.status == "processed":
                logger.info(f"Document already processed with embedding_id: {document.embedding_id}")
                return True
                
            # Generate summary if needed
            if not document.summary:
                document.summary = self.summarize(document)
                logger.info(f"Generated summary for document: {document.url}")
            
            # Create and store embedding
            document.embedding_id = self.store_embedding(document)
            
            # Extract entities and add to knowledge graph if available
            kg_initialized = self._init_kg_manager()
            if kg_initialized:
                logger.info(f"Extracting entities from document: {document.url}")
                extraction_result = self.extract_entities(document)
                
                if extraction_result.get("entities") or extraction_result.get("relationships"):
                    # Store in knowledge graph
                    self._kg_manager.add_document_node(document)
                    self._kg_manager.add_entities_and_relationships(
                        document, 
                        extraction_result.get("entities", []), 
                        extraction_result.get("relationships", [])
                    )
                    logger.info(f"Added document and entities to knowledge graph: {document.url}")
            
            # Update status
            document.status = "processed"
            document.process_time = datetime.now()
            
            logger.info(f"Successfully processed document: {document.url}")
            return True
        
        except Exception as e:
            logger.error(f"Error processing document {document.url}: {e}", exc_info=True)
            document.status = "error_processing"
            return False
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query using vector search.
        
        Args:
            query: Text query
            k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        self._init_vector_store()
        
        try:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "title": doc.metadata.get("title", "Untitled"),
                    "url": doc.metadata.get("url", ""),
                    "source": doc.metadata.get("source", ""),
                    "subsource": doc.metadata.get("subsource", ""),
                    "summary": doc.page_content,
                    "similarity_score": score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}", exc_info=True)
            return []
    
    def hybrid_search(self, query, limit=5):
        """
        Perform a hybrid search using both vector search and knowledge graph.
        
        Args:
            query: Text query
            limit: Number of results
            
        Returns:
            List of search results with documents and relevant context
        """
        try:
            # 1. Extract entities from query for knowledge graph search
            entities_prompt = f"""
            Extract all entities (people, organizations, policies, programs, etc.) 
            mentioned in this query. List only the entity names without explanations.
            
            Query: {query}
            """
            
            self._init_llm()
            entities_response = self._llm.invoke(entities_prompt)
            extracted_entities = [
                e.strip() for e in entities_response.content.strip().split('\n') 
                if e.strip() and not e.strip().startswith('Entities:')
            ]
            
            # 2. Perform vector search
            vector_results = self.search_similar_documents(query, k=limit)
            
            # 3. If knowledge graph is available, perform graph search
            graph_results = []
            if self._init_kg_manager() and extracted_entities:
                graph_results = self._kg_manager.search_related_documents(
                    extracted_entities, 
                    limit=limit
                )
            
            # 4. Combine and rank results
            # For now, we'll use a simple approach of prioritizing graph results
            # and then adding vector results until we reach the limit
            combined_results = []
            seen_urls = set()
            
            # Add graph results first
            for result in graph_results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    combined_results.append({
                        **result, 
                        "source": "knowledge_graph",
                        "graph_context": result.get("context", "")
                    })
                
                if len(combined_results) >= limit:
                    break
            
            # Add vector results
            for result in vector_results:
                if result["url"] not in seen_urls and len(combined_results) < limit:
                    seen_urls.add(result["url"])
                    combined_results.append({
                        **result, 
                        "source": "vector_search",
                    })
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            return []