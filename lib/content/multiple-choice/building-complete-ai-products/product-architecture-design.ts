import { MultipleChoiceQuestion } from '../../../types';

export const productArchitectureDesignMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-pad-mc-1',
      question:
        'What is the primary advantage of using Redis for caching LLM responses in an AI application?',
      options: [
        'Redis is cheaper than PostgreSQL',
        'In-memory storage enables sub-millisecond lookup times',
        'Redis automatically generates embeddings',
        'Redis has better security than other databases',
      ],
      correctAnswer: 1,
      explanation:
        'Redis stores data in-memory, providing sub-millisecond lookup times (vs 10-100ms for PostgreSQL). For semantic caching (matching similar prompts to cached responses), fast lookup is critical to avoid adding latency. Redis also supports TTL (time-to-live) for automatic cache expiration.',
    },
    {
      id: 'bcap-pad-mc-2',
      question:
        'For an AI product with chat, image generation, and document processing features, why would you choose microservices over a monolith?',
      options: [
        'Microservices are always faster than monoliths',
        'Different resource requirements (GPU for images, CPU for docs) justify separate services',
        'Microservices are easier to develop',
        'Monoliths cannot handle multiple features',
      ],
      correctAnswer: 1,
      explanation:
        'Image generation requires expensive GPU instances, while document processing needs CPU workers, and chat needs low-latency API servers. A monolith would require GPU instances for everything (expensive waste). Microservices allow independent scaling and resource optimization per feature.',
    },
    {
      id: 'bcap-pad-mc-3',
      question: 'What is semantic caching for LLM responses?',
      options: [
        'Caching responses by exact prompt match',
        'Embedding prompts and returning cached responses for similar prompts',
        'Storing only the first token of responses',
        'Compressing responses to save storage',
      ],
      correctAnswer: 1,
      explanation:
        'Semantic caching embeds prompts into vectors, then uses similarity search to find similar previous prompts. If similarity > threshold (e.g., 0.95), return the cached response. This works for paraphrased or slightly different prompts, achieving 30-40% cache hit rates vs 5-10% for exact matching.',
    },
    {
      id: 'bcap-pad-mc-4',
      question:
        'Why is PostgreSQL preferred over MongoDB for most AI applications?',
      options: [
        'MongoDB cannot store JSON data',
        'PostgreSQL has JSONB support, stronger consistency (ACID), and better join performance',
        'PostgreSQL is free while MongoDB costs money',
        'MongoDB does not support indexes',
      ],
      correctAnswer: 1,
      explanation:
        'PostgreSQL offers: (1) JSONB for flexible schema (like MongoDB), (2) ACID transactions, (3) Strong JOIN support for relational queries, (4) Better aggregation performance, (5) More mature ecosystem. MongoDB advantages (flexible schema) are now matched by PostgreSQL JSONB, while PostgreSQL maintains superior consistency and query capabilities.',
    },
    {
      id: 'bcap-pad-mc-5',
      question:
        'What is the recommended approach for handling context that exceeds the LLM token limit?',
      options: [
        'Truncate all context to fit within limits',
        'Always include: recent messages, relevant context (vector search), summary of older messages',
        "Only send the user's last message",
        'Increase the token limit by paying more',
      ],
      correctAnswer: 1,
      explanation:
        'Optimal context management: (1) Always include recent messages (last 10-20) for immediate context, (2) Use vector search to retrieve semantically relevant older messages, (3) Summarize very old messages (>50 messages ago) into a condensed form. This maintains quality while fitting within token limits and reducing costs by 60-80%.',
    },
  ];
