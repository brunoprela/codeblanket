/**
 * Quiz questions for File Embedding & Semantic Search section
 */

export const fileembeddingsemanticsearchQuiz = [
  {
    id: 'fpdu-embed-q-1',
    question:
      'Design a semantic code search system like Cursor that can find relevant files based on natural language queries. How would you handle indexing, search, and updates?',
    hint: 'Consider chunking, embedding generation, vector storage, incremental updates, and ranking.',
    sampleAnswer:
      'Semantic code search: (1) Indexing: chunk large files by functions/classes, generate embeddings for each chunk, store in vector DB with metadata (file path, function name). (2) Search: embed query, find similar chunks via cosine similarity, rank by relevance, return file paths with context. (3) Updates: watch file changes, re-embed modified files, incremental index updates. (4) Optimization: cache embeddings, batch embedding generation, use efficient vector DB (Chroma, Pinecone). (5) Ranking: combine semantic similarity + keyword match + file importance.',
    keyPoints: [
      'Chunk files by functions/classes',
      'Generate embeddings per chunk',
      'Store in vector DB with metadata',
      'Incremental updates on file changes',
      'Combine semantic + keyword search',
    ],
  },
  {
    id: 'fpdu-embed-q-2',
    question:
      'Compare different embedding models (OpenAI, open-source) for file search. What are the trade-offs?',
    hint: 'Consider quality, cost, speed, deployment, and offline capability.',
    sampleAnswer:
      'Embedding comparison: (1) OpenAI (text-embedding-3-small): High quality, API-based, costs per request, requires internet, fast. Best for cloud apps. (2) Open-source (sentence-transformers): Free, runs locally, slower on CPU, offline capable, customizable. Best for privacy/offline. (3) Trade-offs: OpenAI is easier but costs scale, open-source is free but requires GPU for speed. Choose based on: deployment (cloud vs local), budget, latency requirements, privacy needs.',
    keyPoints: [
      'OpenAI: high quality, API-based, costs scale',
      'Open-source: free, local, GPU for speed',
      'Choose based on deployment and budget',
      'OpenAI for cloud, open-source for local',
      'Consider latency and privacy needs',
    ],
  },
  {
    id: 'fpdu-embed-q-3',
    question:
      'How would you optimize embedding costs when indexing thousands of files?',
    hint: 'Think about caching, chunking, selective embedding, and batch processing.',
    sampleAnswer:
      'Cost optimization: (1) Cache embeddings with file hash - only re-embed if content changes. (2) Chunk smartly - embed only important chunks not entire files. (3) Batch requests - embed multiple texts in single API call. (4) Selective embedding - index only relevant files (exclude tests, generated code). (5) Use cheaper models - text-embedding-3-small vs large. (6) Incremental indexing - embed new/changed files only. (7) Periodic re-indexing during off-peak.',
    keyPoints: [
      'Cache embeddings by file hash',
      'Chunk files intelligently',
      'Batch API requests',
      'Index relevant files only',
      'Use cost-effective models',
      'Incremental updates',
    ],
  },
];
