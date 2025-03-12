# Government Data Pipeline

A comprehensive system for scraping, processing, summarizing, and storing government documents with knowledge graph integration.

## Overview

This pipeline automates the entire process of gathering content from government websites, generating structured summaries, creating vector embeddings for search, and building a knowledge graph of entities and relationships.

The system is designed to be:
- **Efficient**: Automatically skips previously processed content
- **Intelligent**: Uses AI to generate summaries and extract entities
- **Scalable**: Works with various government websites and document types
- **Integrated**: Combines vector search and knowledge graph capabilities

## Architecture

![Architecture Diagram](https://mermaid.ink/img/pako:eNp1kk9PwzAMxb9KlBNI7dRfadMd9sNhHJAQBw6-JF6JljYiyYTG-O7YabcJDXpJnPfs92zHB2GsRZGLFsseO8WbyjVgfvSQHlE36Db73GOFJnP8TK2paa4rPRfwUGuv29TAIwx2dI3Vrtd2_GvrGjDG2t4sZM72QHsM6Hw4z9v2YAfs-2hNlxJetkMfuUiIE0yElhXtMcLjhcVx4vqeomfQ89oO2YlHE8k-3gf6_cYGOX1P1OQ4n0eewvUYrgGGEqMiDd_pU47LVqOm4KlmxuPJkHjnZ6PY6Dv0nkydImeqoUDRswzXU-UbVGQdjsHBZzZdrUDkeGDrODFc6a3QHlsUXHMnzuFUjppA8W1G0zg0KT-xexQFO9QJLlluUBww8cYG0I-A6NXDnzhD3r7AoQq_MQvLSKI4GxrpF02q3jSmnLXGXRzpZ73TUDTQaOKTcazYsUXBN4_HXdDw-geVvsvB?type=png)

## Components & Packages

### 1. Web Scraping
- **ScraperAdapter**: Custom adapter for browser automation
- **Packages**:
  - `selenium`: Browser automation
  - `BeautifulSoup4`: HTML parsing
  - `lxml`: XPath processing
  - `requests`: HTTP requests for non-JS sites

### 2. Content Extraction
- **ContentExtractor**: Base class with CSS and XPath implementations
- **Packages**:
  - `BeautifulSoup4`: Content extraction
  - `lxml`: XPath extraction

### 3. Document Processing
- **EnhancedProcessor**: Unified processor for document processing
- **Packages**:
  - `langchain`: Framework for AI chains
  - `langchain_anthropic`: Claude integration
  - `langchain.text_splitter`: Document chunking
  - `langchain.chains.summarize`: Map-reduce summarization

### 4. AI Integration
- **Model**: Claude 3 Haiku for summarization and entity extraction
- **Packages**:
  - `langchain_anthropic`: Claude model wrapper
  - `anthropic`: API access to Claude

### 5. Vector Embeddings & Search
- **Models**: OpenAI text-embedding-3-small
- **Packages**:
  - `langchain_openai`: OpenAI embedding wrapper
  - `openai`: API access
  - `langchain_pinecone`: Vector database integration
  - `pinecone`: Vector database API

### 6. Knowledge Graph
- **KnowledgeGraphManager**: Entity and relationship management
- **Packages**:
  - `neo4j`: Graph database driver
  - `langchain.graphs`: Graph operations (optional)

### 7. Database Storage
- **SupabaseManager**: Document storage and retrieval
- **Packages**:
  - `supabase`: Supabase client
  - `python-dotenv`: Environment variable management

## Installation

### Prerequisites
- Python 3.8+
- Supabase account
- Pinecone account
- Neo4j AuraDB account (for knowledge graph)
- Anthropic API key (for Claude)
- OpenAI API key (for embeddings)

### Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install python-dotenv supabase langchain langchain_anthropic langchain_openai 
pip install langchain_pinecone neo4j beautifulsoup4 lxml selenium requests
```

### Environment Setup

Create a `.env` file in the project root:

```
# Supabase credentials
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key

# AI model API keys
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# Pinecone configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=govt-scrape-index
PINECONE_NAMESPACE=govt-content

# Neo4j configuration
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

## Configuration

Create a `config.json` file to specify which government websites to scrape:

```json
{
  "sources": [
    {
      "name": "Federal Reserve",
      "url": "https://www.federalreserve.gov",
      "pages": [
        {
          "name": "Press Releases",
          "url_pattern": "/newsevents/pressreleases.htm",
          "extraction": {
            "type": "css",
            "document_links": "ul.list-unstyled li a",
            "content": "main"
          },
          "use_javascript": true
        }
      ]
    }
  ]
}
```

## Usage

### Database Setup

Run the SQL setup script in Supabase:

```bash
python setup_supabase.py --print-sql-only
```

Copy the output SQL and run it in the Supabase SQL Editor.

### Full Pipeline

```bash
python runner.py --config config.json
```

### Scrape Only

```bash
python runner.py --config config.json --scrape-only
```

### Process Only

```bash
python runner.py --config config.json --process-only --limit 20
```

### Reset Knowledge Graph

```bash
python runner.py --reset-kg
```

### Search Documents

```bash
python hybrid_search.py "your search query"
```

## Features

### 1. Intelligent Document Processing
- **Smart Summarization**: Automatically adapts to document length
- **Structured Output**: Consistent summaries with TITLE, FACTS, SENTIMENT, and TAGS
- **Entity Extraction**: Identifies organizations, people, programs, policies, and their relationships

### 2. Chronological Optimization
- Automatically skips older documents when newer ones are already processed
- Dramatically reduces processing time for frequently checked sources
- Smart re-processing of previously failed documents

### 3. Hybrid Search
- Combines vector similarity with knowledge graph relationships
- More comprehensive results than either approach alone
- Provides context on how entities are related

### 4. Entity Resolution
- Handles different mentions of the same entity (e.g., "EPA", "Environmental Protection Agency")
- Builds a coherent knowledge graph with standardized entity names
- Filters out numerical values, dates, and generic concepts that aren't true entities

## Troubleshooting

### Connection Issues
- Verify your environment variables are correct
- Ensure your API keys have sufficient permissions and credits
- Check that your Neo4j instance is running and accessible

### Processing Errors
- Examine logs for specific error messages
- Try running with `--debug` flag for more detailed information
- For Neo4j connection issues, use `neo4j-test.py` to verify connectivity

### Knowledge Graph Issues
- Run `neo4j-doc-test.py` to test document integration
- Use `runner.py --kg-stats` to view entity and relationship counts
- Consider `--reset-kg` to start fresh if the graph becomes corrupted