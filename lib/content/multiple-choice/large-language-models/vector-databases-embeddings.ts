export const vectorDatabasesEmbeddingsMC = {
  title: 'Vector Databases & Embeddings Quiz',
  id: 'vector-databases-embeddings-mc',
  sectionId: 'vector-databases-embeddings',
  questions: [
    {
      id: 1,
      question:
        'What is HNSW (Hierarchical Navigable Small World) used for in vector databases?',
      options: [
        'Compressing embeddings',
        'Approximate nearest neighbor search with high recall',
        'Generating embeddings',
        'Data encryption',
      ],
      correctAnswer: 1,
      explanation:
        'HNSW is a graph-based algorithm for approximate nearest neighbor (ANN) search. It provides high recall (95-99%) with logarithmic search complexity, making it much faster than exact search for large datasets while maintaining quality.',
    },
    {
      id: 2,
      question:
        'What is the primary advantage of modern transformer-based embeddings over Word2Vec?',
      options: [
        'Smaller embedding dimensions',
        'Faster computation',
        'Contextual embeddings that vary based on surrounding words',
        'No training required',
      ],
      correctAnswer: 2,
      explanation:
        'Word2Vec creates static embeddings—"bank" always has the same vector. Transformer-based embeddings (BERT, sentence-transformers) are contextual—"bank" has different representations in "river bank" vs "bank account" based on context.',
    },
    {
      id: 3,
      question:
        'In a vector database with 100 million vectors, which search algorithm provides the best balance of speed and recall?',
      options: [
        'Exact k-NN (brute force)',
        'HNSW',
        'Linear scan with filters',
        'Random sampling',
      ],
      correctAnswer: 1,
      explanation:
        'HNSW provides excellent recall (95-99%) with sub-millisecond latency even at 100M+ scale. Exact search would be too slow (hundreds of milliseconds), while other methods sacrifice too much recall. HNSW is the industry standard for this scale.',
    },
    {
      id: 4,
      question: 'What is "hybrid search" in the context of vector databases?',
      options: [
        'Searching multiple vector databases simultaneously',
        'Combining vector similarity search with keyword/filter search',
        'Using two different distance metrics',
        'Searching with multiple query vectors',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid search combines vector similarity (semantic search) with traditional keyword search and metadata filters. This provides both semantic understanding and precise filtering, e.g., "find similar documents about AI from 2023" combines vector and filter.',
    },
    {
      id: 5,
      question:
        'What is the typical dimensionality of modern text embeddings (like OpenAI embeddings)?',
      options: [
        '50-100 dimensions',
        '300-400 dimensions',
        '768-1536 dimensions',
        '10,000+ dimensions',
      ],
      correctAnswer: 2,
      explanation:
        "Modern text embeddings typically use 768 dimensions (BERT, sentence-transformers) or 1536 dimensions (OpenAI ada-002). Higher dimensions capture more information but increase memory/compute costs. There's a sweet spot around 768-1536.",
    },
  ],
};
