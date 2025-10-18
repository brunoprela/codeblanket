/**
 * Multiple choice questions for GraphQL section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const graphqlMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'graphql-vs-rest-fetching',
    question:
      'What is the primary advantage of GraphQL over REST in terms of data fetching?',
    options: [
      'GraphQL is faster because it uses binary encoding',
      'GraphQL allows clients to request exactly the data they need, avoiding over-fetching and under-fetching',
      'GraphQL automatically caches responses more efficiently',
      'GraphQL requires fewer servers to operate',
    ],
    correctAnswer: 1,
    explanation:
      "GraphQL's primary advantage is that clients can specify exactly which fields they need in the query, receiving neither more nor less data. REST endpoints return fixed data structures, often resulting in over-fetching (getting unused fields) or under-fetching (needing multiple requests). GraphQL uses JSON (not binary), caching is actually more complex than REST, and server requirements are comparable.",
  },
  {
    id: 'graphql-n-plus-one',
    question:
      'You have a GraphQL query that fetches 100 posts and the author for each post. Without DataLoader, how many database queries would typically be executed?',
    options: [
      '1 query (GraphQL automatically batches)',
      '2 queries (1 for posts, 1 for all authors)',
      '101 queries (1 for posts, 100 for each author)',
      '100 queries (GraphQL optimizes away the posts query)',
    ],
    correctAnswer: 2,
    explanation:
      'This is the classic N+1 problem. Without DataLoader batching, you would execute 1 query to fetch all posts, then 100 separate queries to fetch each author individually. DataLoader solves this by batching the author queries into a single query: SELECT * FROM users WHERE id IN (id1, id2, ..., id100). Always use DataLoader to avoid this performance issue.',
  },
  {
    id: 'graphql-subscriptions',
    question:
      'How do GraphQL subscriptions typically communicate real-time updates to clients?',
    options: [
      'HTTP long polling',
      'Server-Sent Events (SSE)',
      'WebSocket connections',
      'Regular HTTP requests with short intervals',
    ],
    correctAnswer: 2,
    explanation:
      'GraphQL subscriptions typically use WebSocket connections for bidirectional, real-time communication. When a client subscribes, a WebSocket connection is established, and the server pushes updates whenever relevant data changes. While SSE could work for server-to-client updates, WebSocket is the standard for GraphQL subscriptions and is supported by all major GraphQL implementations like Apollo.',
  },
  {
    id: 'graphql-caching-challenge',
    question: 'Why is caching more difficult in GraphQL compared to REST APIs?',
    options: [
      'GraphQL responses are larger and harder to store',
      'GraphQL uses POST requests to a single endpoint, bypassing standard HTTP caching',
      'GraphQL servers are stateful and cannot be cached',
      'GraphQL responses contain metadata that prevents caching',
    ],
    correctAnswer: 1,
    explanation:
      "GraphQL typically uses POST requests to a single `/graphql` endpoint with the query in the request body. HTTP caching (like CDN caching, browser caching) works primarily with GET requests to different URLs. This makes standard HTTP caching ineffective. Solutions include: persisted queries (converting to GET requests with query hashes), response caching at the GraphQL server level, or client-side caching (like Apollo Client's normalized cache).",
  },
  {
    id: 'graphql-use-case',
    question: 'Which scenario is the BEST fit for choosing GraphQL over REST?',
    options: [
      'A public API consumed by thousands of third-party developers who need stable, well-documented endpoints',
      'A mobile application that displays complex screens with data from multiple resources and has bandwidth constraints',
      'A simple file storage service focused on uploading and downloading binary files',
      'A high-performance internal microservice handling millions of requests per second',
    ],
    correctAnswer: 1,
    explanation:
      'GraphQL excels for mobile applications because: (1) It minimizes data transfer by letting clients request only needed fields, (2) Complex screens requiring data from multiple resources can be fetched in a single request, (3) Bandwidth is often limited on mobile networks. For public APIs, REST is more familiar; for file storage, REST is simpler; for high-performance internal services, gRPC is typically better than GraphQL.',
  },
];
