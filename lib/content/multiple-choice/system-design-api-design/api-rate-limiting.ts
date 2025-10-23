/**
 * Multiple choice questions for API Rate Limiting Strategies section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apiratelimitingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ratelimit-q1',
    question: 'What is the main problem with Fixed Window rate limiting?',
    options: [
      'It uses too much memory to store request timestamps',
      'It allows bursts at window boundaries (e.g., 200 requests in 1 second across 2 windows)',
      'It requires Redis for distributed systems',
      "It doesn't provide accurate rate limiting over time",
    ],
    correctAnswer: 1,
    explanation:
      'Fixed Window allows bursts at boundaries: user sends 100 requests at t=59s (end of window 1) and 100 at t=60s (start of window 2) = 200 requests in 1 second, bypassing the limit. Sliding Window Counter solves this by weighting counts across windows.',
  },
  {
    id: 'ratelimit-q2',
    question:
      'Which rate limiting algorithm is most memory efficient while providing smooth rate limiting?',
    options: [
      'Fixed Window (stores one counter per window)',
      'Sliding Window Log (stores every request timestamp)',
      'Sliding Window Counter (stores two counters with weighted calculation)',
      'Token Bucket (stores token count and last refill time)',
    ],
    correctAnswer: 2,
    explanation:
      'Sliding Window Counter is most efficient: stores only 2 counters (current + previous window) and uses weighted calculation. Sliding Window Log stores ALL timestamps (memory intensive). Token Bucket is efficient but allows bursts. Fixed Window has burst issues.',
  },
  {
    id: 'ratelimit-q3',
    question:
      'What HTTP status code should be returned when a client exceeds rate limits?',
    options: [
      '403 Forbidden',
      '503 Service Unavailable',
      '429 Too Many Requests',
      '400 Bad Request',
    ],
    correctAnswer: 2,
    explanation:
      '429 Too Many Requests is the standard HTTP status code for rate limiting. 403 is for authorization failures, 503 is for server unavailability, 400 is for invalid requests. Always return 429 with Retry-After header.',
  },
  {
    id: 'ratelimit-q4',
    question:
      'Why is cost-based rate limiting useful for APIs with diverse operations?',
    options: [
      'It reduces server costs by limiting expensive operations',
      'It allows assigning different "costs" to operations (e.g., search costs 10x more than read)',
      'It encrypts expensive operations for security',
      'It automatically scales backend resources based on cost',
    ],
    correctAnswer: 1,
    explanation:
      'Cost-based rate limiting assigns different costs to operations based on resource usage. Expensive operations (analytics, search) cost more points than simple reads. User has point budget (e.g., 10,000/hour). This prevents abuse of expensive endpoints without overly restricting cheap operations.',
  },
  {
    id: 'ratelimit-q5',
    question:
      'What is the purpose of the Retry-After header in rate limit responses?',
    options: [
      'To tell the client how many retries are allowed',
      'To specify how many seconds the client should wait before retrying',
      'To indicate the retry strategy (exponential backoff vs linear)',
      'To automatically retry the request on the client side',
    ],
    correctAnswer: 1,
    explanation:
      'Retry-After header tells client how many seconds to wait before retrying. This prevents clients from spam-retrying immediately, which would worsen the situation. Standard practice: return Retry-After with 429 status code.',
  },
];
