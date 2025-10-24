export const contextWindowManagementMC = {
  title: 'Context Window Management Quiz',
  id: 'context-window-management-mc',
  sectionId: 'context-window-management',
  questions: [
    {
      id: 1,
      question: 'What is the "lost in the middle" problem with long contexts?',
      options: [
        'Models forget the prompt halfway through generation',
        'Models have difficulty attending to information in the middle of long contexts',
        "The middle layers of the transformer don't work properly",
        'Training data is missing middle sections',
      ],
      correctAnswer: 1,
      explanation:
        "Research shows LLMs struggle to use information from the middle of long contexts, with better attention to the beginning and end. This position bias means simply stuffing everything into context doesn't work—you need to put important info at the edges or use retrieval.",
    },
    {
      id: 2,
      question:
        'What is the recommended overlap percentage when chunking documents?',
      options: [
        '0% (no overlap)',
        '10-20% overlap',
        '50% overlap',
        '90% overlap',
      ],
      correctAnswer: 1,
      explanation:
        "Using 10-20% overlap between chunks helps ensure important information isn't lost at boundaries. For example, with 512-token chunks, a 100-token overlap means each chunk shares 100 tokens with its neighbors, providing continuity.",
    },
    {
      id: 3,
      question:
        'What is the typical cost difference between using 10k vs 100k context tokens?',
      options: [
        "Same cost—context window doesn't affect pricing",
        '2x more expensive',
        '10x more expensive (linear with tokens)',
        '100x more expensive',
      ],
      correctAnswer: 2,
      explanation:
        'Most APIs price linearly by token count. Using 100k tokens costs 10x more than 10k tokens. This makes context compression and retrieval (only including relevant chunks) crucial for cost management at scale.',
    },
    {
      id: 4,
      question:
        'For a chatbot, what is the best strategy for managing conversation history that exceeds context limits?',
      options: [
        'Delete oldest messages (sliding window)',
        'Summarize old conversation and keep recent messages',
        'Start a new conversation and lose all history',
        'Increase context window infinitely',
      ],
      correctAnswer: 1,
      explanation:
        'Summarizing old messages while keeping recent ones preserves long-term context (key facts, user preferences) while maintaining detailed recent context. This hybrid approach works better than pure sliding window (loses long-term info) or summarizing everything (loses nuance).',
    },
    {
      id: 5,
      question:
        'What is hierarchical summarization (map-reduce pattern) used for?',
      options: [
        "Summarizing models's hidden layers",
        'Recursively summarizing document chunks until reaching a final summary',
        'Creating multiple summaries at different detail levels',
        'Distributing summarization across multiple models',
      ],
      correctAnswer: 1,
      explanation:
        'Map-reduce summarization: (1) Split long document into chunks, (2) Summarize each chunk (map), (3) Combine and summarize the summaries (reduce), (4) Repeat until single summary fits. This enables summarizing documents far beyond context limits.',
    },
  ],
};
