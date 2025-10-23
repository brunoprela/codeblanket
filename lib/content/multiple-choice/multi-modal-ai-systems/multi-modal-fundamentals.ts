export const multimodalfundamentalsMultipleChoice = [
  {
    id: 'mmas-fundamentals-mc-1',
    question:
      'Which approach provides the best balance of cost and flexibility for a multi-modal system that processes images, videos, and text?',
    options: [
      'Early fusion with a single unified model processing all modalities together',
      'Late fusion with separate models per modality and smart routing based on input',
      "Always process all modalities regardless of whether they're provided",
      'Use a single text-only model and convert all other modalities to text first',
    ],
    correctAnswer: 1,
    explanation:
      'Late fusion with smart routing offers the best balance because it allows modular development, processes only necessary modalities (reducing cost), enables parallel processing, and makes it easy to upgrade individual components. You avoid the overhead of processing unused modalities while maintaining flexibility.',
  },
  {
    id: 'mmas-fundamentals-mc-2',
    question:
      'What is the most effective way to optimize costs in a multi-modal AI system?',
    options: [
      'Always use the cheapest models available',
      'Cache results aggressively, optimize image sizes, batch similar requests, and use appropriate model quality settings based on use case',
      'Process everything synchronously to avoid queue overhead',
      'Store all raw media files without preprocessing',
    ],
    correctAnswer: 1,
    explanation:
      'Cost optimization requires a multi-faceted approach: caching prevents re-processing identical content, optimizing image sizes reduces token costs, batching improves efficiency, and using appropriate model quality (high for critical, standard for routine) balances quality and cost. This comprehensive strategy provides the best results.',
  },
  {
    id: 'mmas-fundamentals-mc-3',
    question:
      'Which embedding model is most appropriate for unified text-image retrieval in a multi-modal RAG system?',
    options: [
      'Separate text embeddings and image embeddings in different vector spaces',
      'CLIP embeddings that place text and images in a shared semantic space',
      'Standard word embeddings (Word2Vec)',
      'No embeddings - use direct pixel comparison',
    ],
    correctAnswer: 1,
    explanation:
      'CLIP embeddings are specifically designed for multi-modal applications, creating a shared semantic space where semantically similar images and text have similar embeddings. This enables cross-modal retrieval (text query → image results) and unified search across modalities, which is essential for multi-modal RAG systems.',
  },
  {
    id: 'mmas-fundamentals-mc-4',
    question:
      'What is the recommended detail level for GPT-4 Vision when processing documents?',
    options: [
      "Always use 'low' detail to save costs",
      "Always use 'high' detail for maximum accuracy",
      "Use 'high' detail for documents where accuracy is critical, 'low' for simple images or when speed matters",
      "Detail level doesn't matter for documents",
    ],
    correctAnswer: 2,
    explanation:
      "Detail level should match the task requirements. Documents with important text, tables, or charts need 'high' detail for accurate extraction. Simple images, thumbnails, or situations where speed is prioritized can use 'low' detail. This selective approach balances accuracy, cost, and latency based on actual needs.",
  },
  {
    id: 'mmas-fundamentals-mc-5',
    question:
      'How should you handle rate limits when building production multi-modal AI applications?',
    options: [
      'Ignore rate limits and let requests fail',
      'Implement exponential backoff, request queuing, and graceful degradation strategies',
      'Always wait 60 seconds between requests',
      'Switch to a different API provider immediately',
    ],
    correctAnswer: 1,
    explanation:
      'Production applications need robust rate limit handling: exponential backoff for transient rate limits, request queuing to handle bursts, graceful degradation when limits are reached (cached responses, reduced functionality), and monitoring to predict limit issues. This ensures reliability even under high load or API constraints.',
  },
];
