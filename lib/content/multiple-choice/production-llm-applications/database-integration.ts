import { MultipleChoiceQuestion } from '../../../types';

export const databaseIntegrationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-db-mc-1',
    question: 'What is the purpose of connection pooling?',
    options: [
      'Faster queries',
      'Reuse database connections instead of creating new ones (expensive)',
      'Save memory',
      'Better security',
    ],
    correctAnswer: 1,
    explanation:
      'Connection pooling reuses existing connections (expensive to create: 100ms+). Pool of 20 connections handles thousands of requests efficiently.',
  },
  {
    id: 'pllm-db-mc-2',
    question: 'When should you use pgvector vs dedicated vector databases?',
    options: [
      'Always use pgvector',
      'Always use dedicated',
      'pgvector for <1M vectors, dedicated for >10M vectors and advanced features',
      'No difference',
    ],
    correctAnswer: 2,
    explanation:
      'pgvector good for <1M vectors, simpler architecture, lower cost. Dedicated vector DBs (Pinecone, Weaviate) better at scale with advanced filtering.',
  },
  {
    id: 'pllm-db-mc-3',
    question: 'How should you store conversation history?',
    options: [
      'In memory',
      'Messages table with conversation_id foreign key, indexed',
      'Text files',
      'No storage needed',
    ],
    correctAnswer: 1,
    explanation:
      'Store in Messages table with conversation_id FK, indexed on conversation_id and created_at for efficient retrieval of conversation history.',
  },
  {
    id: 'pllm-db-mc-4',
    question: 'What is the purpose of database indexes?',
    options: [
      'Make queries slower',
      'Speed up queries on indexed columns at cost of write performance',
      'Save space',
      'Required for foreign keys',
    ],
    correctAnswer: 1,
    explanation:
      'Indexes speed up SELECT queries on indexed columns (user_id, created_at) by orders of magnitude, with slight overhead on INSERT/UPDATE.',
  },
  {
    id: 'pllm-db-mc-5',
    question: 'How do you handle growing conversation data?',
    options: [
      'Delete old data',
      'Archive conversations older than 90 days to cold storage, partition tables',
      'Nothing needed',
      'Use bigger server',
    ],
    correctAnswer: 1,
    explanation:
      'Archive old conversations to cheaper storage, partition tables by date (monthly), keep only active data in main tables for performance.',
  },
];
