/**
 * Multiple choice questions for GraphQL Schema Design section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const graphqlschemadesignMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'graphql-q1',
    question: 'What is the main advantage of GraphQL over REST APIs?',
    options: [
      'GraphQL is faster because it uses binary protocol',
      'Clients can request exactly the data they need, avoiding over/under-fetching',
      'GraphQL automatically generates database queries',
      "GraphQL doesn't require authentication",
    ],
    correctAnswer: 1,
    explanation:
      "GraphQL's main advantage is that clients specify exactly what data they need in their query, eliminating over-fetching (getting unnecessary data) and under-fetching (needing multiple requests). GraphQL uses HTTP (not binary), doesn't auto-generate database queries, and still requires authentication.",
    difficulty: 'easy',
  },
  {
    id: 'graphql-q2',
    question: 'In GraphQL schema, what does the "!" symbol mean?',
    options: [
      'The field is deprecated',
      'The field requires authentication',
      'The field is non-nullable (required)',
      'The field is unique',
    ],
    correctAnswer: 2,
    explanation:
      'The "!" symbol indicates a field is non-nullable, meaning it must always have a value. "String!" means a required string, "String" is optional. "[Post!]!" means a non-null array containing non-null Posts.',
    difficulty: 'easy',
  },
  {
    id: 'graphql-q3',
    question: 'What is the N+1 query problem in GraphQL, and how is it solved?',
    options: [
      'Making N+1 database queries for a list; solved with DataLoader batching',
      'Requesting N+1 fields; solved with query depth limiting',
      'N+1 concurrent requests; solved with rate limiting',
      'N+1 schema versions; solved with versioning',
    ],
    correctAnswer: 0,
    explanation:
      'N+1 problem: Fetching a list of N items, then making 1 query per item for related data (N+1 total queries). Example: fetching 10 posts, then 10 separate queries for each author. DataLoader batches and caches these queries, turning N+1 into 2 queries.',
    difficulty: 'hard',
  },
  {
    id: 'graphql-q4',
    question:
      'Which pagination approach follows the Relay specification for GraphQL?',
    options: [
      'Offset-based with page numbers',
      'Cursor-based with edges, nodes, and pageInfo',
      'Limit-offset with totalCount',
      'Page-based with hasNextPage',
    ],
    correctAnswer: 1,
    explanation:
      'Relay specification uses cursor-based pagination with specific structure: edges (array of {cursor, node}), nodes (actual data), and pageInfo (hasNextPage, hasPreviousPage, startCursor, endCursor). This provides consistent, efficient pagination.',
    difficulty: 'medium',
  },
  {
    id: 'graphql-q5',
    question: 'When should you make a GraphQL field nullable vs non-nullable?',
    options: [
      'Always make fields non-nullable for data integrity',
      'Make fields nullable to allow schema evolution without breaking clients',
      'Nullable only for optional user input',
      'Non-nullable only for IDs and timestamps',
    ],
    correctAnswer: 1,
    explanation:
      'Nullable fields allow schema evolution - you can add new fields without breaking existing clients. If you add a required field later, old data might not have it. Core fields (id, createdAt) can be non-nullable, but most fields should be nullable for flexibility. Only inputs should match business requirements strictly.',
    difficulty: 'medium',
  },
];
