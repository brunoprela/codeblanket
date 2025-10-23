import { MultipleChoiceQuestion } from '../../../types';

export const apiIntegrationToolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question: 'What is exponential backoff in the context of API retries?',
    options: [
      'Decreasing retry delay over time',
      'Increasing retry delay exponentially with each attempt',
      'Random retry delays',
      'Fixed delay between retries',
    ],
    correctAnswer: 1,
    explanation:
      'Exponential backoff increases the delay between retries exponentially (1s, 2s, 4s, 8s, etc.) to avoid overwhelming the API during outages or rate limiting.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What does a 429 HTTP status code indicate?',
    options: [
      'Server error',
      'Authentication failure',
      'Rate limit exceeded',
      'Invalid request',
    ],
    correctAnswer: 2,
    explanation:
      '429 Too Many Requests indicates that the client has sent too many requests in a given time period (rate limiting).',
  },
  {
    id: 'fcfc-mc-3',
    question: 'Why is jitter added to retry delays?',
    options: [
      'To make retries slower',
      'To prevent thundering herd problem where many clients retry simultaneously',
      'To reduce costs',
      'To improve accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Jitter adds randomness to retry delays to prevent many clients from retrying at exactly the same time (thundering herd), which would overwhelm the service.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'What is the purpose of OAuth 2.0 token refresh?',
    options: [
      'To reduce API costs',
      'To obtain new access tokens when they expire without re-authentication',
      'To make requests faster',
      'To validate user permissions',
    ],
    correctAnswer: 1,
    explanation:
      'OAuth 2.0 refresh tokens allow obtaining new access tokens when they expire without requiring the user to re-authenticate, improving user experience.',
  },
  {
    id: 'fcfc-mc-5',
    question: 'What is semantic caching in API tools?',
    options: [
      'Caching based on exact query strings',
      'Caching based on semantic similarity of queries',
      'Caching API keys',
      'Caching only successful responses',
    ],
    correctAnswer: 1,
    explanation:
      'Semantic caching uses embeddings to find semantically similar queries and return cached results, even when query wording differs.',
  },
];
