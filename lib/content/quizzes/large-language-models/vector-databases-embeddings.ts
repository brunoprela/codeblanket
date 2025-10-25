export const vectorDatabasesEmbeddingsQuiz = {
  title: 'Vector Databases & Embeddings Discussion',
  id: 'vector-databases-embeddings-quiz',
  sectionId: 'vector-databases-embeddings',
  questions: [
    {
      id: 1,
      question:
        'Compare different vector search algorithms: exact nearest neighbor (brute force), HNSW, IVF, and LSH. Analyze the tradeoffs between recall, latency, memory usage, and indexing time. For a production application with 100M vectors, which algorithm would you choose and why?',
      expectedAnswer:
        "Should cover: exact search guarantees but O(n) complexity being prohibitive at scale, HNSW's graph- based approach offering high recall(95 - 99 %) with logarithmic search, IVF's clustering approach trading recall for speed, LSH's probabilistic guarantees, memory requirements(HNSW needs full graph in memory), build time considerations, hybrid approaches, parameter tuning for recall - speed tradeoff, and practical production recommendations based on scale and requirements.",
    },
    {
      id: 2,
      question:
        'Discuss the evolution of embedding models from Word2Vec to modern transformer-based embeddings (sentence-transformers, E5, OpenAI embeddings). What makes modern embeddings superior? How do you evaluate embedding quality, and what metrics matter for different applications (search, clustering, classification)?',
      expectedAnswer:
        "Should discuss: Word2Vec\'s context-free limitations, contextual embeddings from transformers, sentence - level representations vs token - level, contrastive learning(SimCSE, sentence - transformers), fine - tuning on diverse tasks improving generalization, dimensionality considerations(768 - 1536), evaluation metrics (cosine similarity correlation, retrieval benchmarks), task - specific fine - tuning benefits, multilingual embeddings, and domain adaptation strategies.",
    },
    {
      id: 3,
      question:
        'Analyze the architecture decisions for a production vector database system: sharding strategy, replication, consistency model, and query routing. How do systems like Pinecone, Weaviate, and Qdrant differ in their approaches? What are the scaling bottlenecks and how do you address them?',
      expectedAnswer:
        'Should cover: horizontal scaling through sharding, replication for availability and read throughput, eventual consistency tradeoffs in distributed vector search, query routing to relevant shards, filtering capabilities and their performance impact, hybrid search combining vector and metadata filters, backup and disaster recovery for vector indices, cost implications of managed vs self-hosted, benchmarking methodologies, and choosing between specialized vector DBs vs adding vector support to existing databases.',
    },
  ],
};
