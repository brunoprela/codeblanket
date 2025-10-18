/**
 * Multiple choice questions for GraphQL Performance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const graphqlperformanceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'graphql-perf-q1',
    question:
      'What is the N+1 query problem in GraphQL, and how does DataLoader solve it?',
    options: [
      'N+1 network requests; DataLoader combines them into one HTTP request',
      'N+1 database queries for fetching related data; DataLoader batches them',
      'N+1 schema definitions; DataLoader merges them',
      'N+1 validation errors; DataLoader catches them',
    ],
    correctAnswer: 1,
    explanation:
      'N+1 problem: Fetching N items then making 1 query per item for related data (N+1 total queries). Example: 100 posts → 1 query for posts + 100 queries for authors. DataLoader batches these into 2 queries total: one for posts, one batched query for all authors.',
    difficulty: 'medium',
  },
  {
    id: 'graphql-perf-q2',
    question: "Why can't GraphQL leverage HTTP caching as easily as REST APIs?",
    options: [
      "GraphQL doesn't support HTTP headers",
      "GraphQL typically uses POST requests which aren't cached by default",
      'GraphQL responses are always unique per user',
      'GraphQL uses WebSockets instead of HTTP',
    ],
    correctAnswer: 1,
    explanation:
      "GraphQL typically uses POST requests (to send query in body), and POST requests aren't cached by browsers/CDNs by default. REST GET requests have built-in HTTP caching. GraphQL requires custom caching solutions like APQ (Automatic Persisted Queries) or Redis.",
    difficulty: 'medium',
  },
  {
    id: 'graphql-perf-q3',
    question: 'What is the purpose of query complexity analysis in GraphQL?',
    options: [
      'To validate query syntax',
      'To prevent expensive queries by assigning cost limits',
      'To optimize database indexes automatically',
      'To generate TypeScript types',
    ],
    correctAnswer: 1,
    explanation:
      'Query complexity analysis assigns "cost" to each field and rejects queries exceeding a threshold. This prevents malicious or accidentally expensive queries (e.g., fetching 1000 posts × 1000 comments = 1M records). Each field gets a complexity score.',
    difficulty: 'easy',
  },
  {
    id: 'graphql-perf-q4',
    question:
      'When implementing cursor-based pagination, why fetch limit + 1 items?',
    options: [
      'To have a backup item in case one is deleted',
      "To determine if there's a next page (hasNextPage)",
      'To calculate total count efficiently',
      'To improve query performance',
    ],
    correctAnswer: 1,
    explanation:
      'Fetching limit + 1 items lets you check hasNextPage: if you get more than requested, there are more pages. Return only "limit" items to client, but the extra item tells you hasNextPage = true. Avoids expensive COUNT queries.',
    difficulty: 'hard',
  },
  {
    id: 'graphql-perf-q5',
    question: 'What is Automatic Persisted Queries (APQ) and why is it useful?',
    options: [
      'Automatically saves queries to database for audit',
      'Caches queries by hash, reducing bandwidth on repeat requests',
      'Persists subscriptions across server restarts',
      'Automatically generates query documentation',
    ],
    correctAnswer: 1,
    explanation:
      'APQ sends query hash instead of full query text after first request. Server caches query by hash. Subsequent requests send only hash (small), saving bandwidth especially for mobile. First request: full query + hash. Subsequent: just hash.',
    difficulty: 'medium',
  },
];
