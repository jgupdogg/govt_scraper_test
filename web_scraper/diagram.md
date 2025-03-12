graph TD
    subgraph Scraping
        A[Government Websites] -->|ScraperAdapter| B[Document HTML]
        B -->|ContentExtractor| C[Raw Text]
    end
    
    subgraph Processing
        C -->|EnhancedProcessor| D[Document Processing]
        D -->|LangChain Text Splitter| E[Document Chunks]
        E -->|Claude 3| F[Structured Summary]
        E -->|Claude 3| G[Entity Extraction]
    end
    
    subgraph Storage
        F -->|Pinecone| H[Vector Database]
        F -->|OpenAI Embeddings| H
        F -->|SupabaseManager| I[Document Storage]
        G -->|KnowledgeGraphManager| J[Knowledge Graph]
    end
    
    subgraph Retrieval
        K[User Query] -->|Hybrid Search| L[Results]
        H -->|Vector Similarity| L
        J -->|Entity Relationships| L
    end
    
    style Scraping fill:#f9f9f9,stroke:#333,stroke-width:1px
    style Processing fill:#e6f7ff,stroke:#333,stroke-width:1px
    style Storage fill:#f0f9eb,stroke:#333,stroke-width:1px
    style Retrieval fill:#fff2e8,stroke:#333,stroke-width:1px