/**
 * Multiple choice questions for File Embedding & Semantic Search section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const fileembeddingsemanticsearchMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fpdu-embed-mc-1',
      question: 'What is the purpose of file embeddings?',
      options: [
        'Compress files',
        'Convert files to vector representations for semantic search',
        'Encrypt file content',
        'Index file metadata',
      ],
      correctAnswer: 1,
      explanation:
        'Embeddings convert text into vector representations that capture semantic meaning, enabling similarity search based on meaning not just keywords.',
    },
    {
      id: 'fpdu-embed-mc-2',
      question:
        'Which metric is commonly used to compare embedding similarity?',
      options: [
        'Euclidean distance',
        'Manhattan distance',
        'Cosine similarity',
        'Hamming distance',
      ],
      correctAnswer: 2,
      explanation:
        'Cosine similarity is the standard metric for comparing embeddings as it measures angle between vectors, capturing semantic similarity regardless of magnitude.',
    },
    {
      id: 'fpdu-embed-mc-3',
      question: 'What is a vector database?',
      options: [
        'A database for storing vectors like arrays',
        'A database optimized for similarity search over embeddings',
        'A database using vector graphics',
        'A linear algebra database',
      ],
      correctAnswer: 1,
      explanation:
        'Vector databases (like Chroma, Pinecone, Weaviate) are specialized for efficiently storing and searching high-dimensional vectors (embeddings).',
    },
    {
      id: 'fpdu-embed-mc-4',
      question: 'Why cache file embeddings?',
      options: [
        'To make files load faster',
        'To avoid regenerating expensive embeddings for unchanged files',
        'To compress file storage',
        'To improve file security',
      ],
      correctAnswer: 1,
      explanation:
        'Caching embeddings saves API costs and time by avoiding regeneration for unchanged files. Check file hash to detect changes.',
    },
    {
      id: 'fpdu-embed-mc-5',
      question:
        'What is the recommended OpenAI embedding model for cost-effectiveness?',
      options: [
        'text-embedding-3-large',
        'text-embedding-3-small',
        'text-embedding-ada-002',
        'gpt-4-embeddings',
      ],
      correctAnswer: 1,
      explanation:
        'text-embedding-3-small offers excellent quality at lower cost compared to the large variant. It is suitable for most applications.',
    },
  ];
