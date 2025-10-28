export const ragFundamentalsMC = {
  title: 'Rag Fundamentals - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'Which of the following best describes the primary purpose of rag fundamentals?',
      options: [
        'To improve retrieval accuracy by understanding semantic meaning',
        'To reduce API costs through caching',
        'To increase database query speed',
        'To simplify code implementation',
      ],
      correctAnswer: 0,
      explanation:
        'The primary purpose is to improve retrieval accuracy through semantic understanding. While other benefits may exist, semantic accuracy is the core goal.',
      difficulty: 'easy' as const,
      category: 'Concepts',
    },
    {
      id: 2,
      question:
        'In a production RAG system using rag fundamentals, what is the most critical metric to monitor?',
      options: [
        'Total number of API calls',
        'Retrieval precision and recall',
        'Database connection pool size',
        'Code deployment frequency',
      ],
      correctAnswer: 1,
      explanation:
        "Retrieval quality (precision and recall) directly impacts the system's ability to provide accurate, relevant responses. This is the most critical metric for RAG system effectiveness.",
      difficulty: 'intermediate' as const,
      category: 'Production',
    },
    {
      id: 3,
      question:
        'When implementing rag fundamentals, which approach typically provides the best balance of accuracy and performance?',
      options: [
        'Exact search with no optimization',
        'Approximate nearest neighbor search with HNSW indexing',
        'Linear scan of all documents',
        'Random sampling of documents',
      ],
      correctAnswer: 1,
      explanation:
        'HNSW (Hierarchical Navigable Small World) indexing provides excellent accuracy (~99%) with logarithmic time complexity, making it ideal for production systems.',
      difficulty: 'intermediate' as const,
      category: 'Implementation',
    },
    {
      id: 4,
      question:
        'What is the primary challenge when scaling rag fundamentals to millions of documents?',
      options: [
        'Finding enough storage space',
        'Maintaining fast query times while ensuring accuracy',
        'Writing more code',
        'Hiring more developers',
      ],
      correctAnswer: 1,
      explanation:
        'The key challenge is maintaining sub-second query times across millions of vectors while preserving retrieval accuracy. This requires sophisticated indexing and caching strategies.',
      difficulty: 'advanced' as const,
      category: 'Scaling',
    },
    {
      id: 5,
      question:
        'Which of the following is NOT a recommended practice for rag fundamentals in production?',
      options: [
        'Implementing comprehensive monitoring and logging',
        'Using a single global embedding for all documents',
        'Caching frequent queries',
        'Regular evaluation of retrieval quality',
      ],
      correctAnswer: 1,
      explanation:
        'Using a single global embedding would lose all semantic information. Each document needs its own embedding to enable semantic search. The other options are all best practices.',
      difficulty: 'easy' as const,
      category: 'Best Practices',
    },
  ],
};
