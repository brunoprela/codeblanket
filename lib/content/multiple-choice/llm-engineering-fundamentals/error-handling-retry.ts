/**
 * Multiple choice questions for Error Handling & Retry Logic section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const errorhandlingretryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What HTTP status code indicates a rate limit has been exceeded?',
    options: [
      '400 Bad Request',
      '401 Unauthorized',
      '429 Too Many Requests',
      '500 Internal Server Error',
    ],
    correctAnswer: 2,
    explanation:
      '429 Too Many Requests indicates rate limiting. This is a transient error that should be retried with exponential backoff. Wait times should increase (1s, 2s, 4s, 8s) to give the API time to recover.',
  },
  {
    id: 'mc2',
    question: 'Which error should NEVER be retried?',
    options: [
      '429 Rate Limit',
      '500 Internal Server Error',
      '401 Unauthorized',
      '503 Service Unavailable',
    ],
    correctAnswer: 2,
    explanation:
      '401 Unauthorized indicates invalid credentials - retrying will never succeed. Fix the API key instead. 429, 500, and 503 are transient errors worth retrying. Never retry authentication errors, bad requests, or content policy violations.',
  },
  {
    id: 'mc3',
    question: 'What is exponential backoff?',
    options: [
      'Reducing cost exponentially over time',
      'Doubling wait time between retries (1s, 2s, 4s, 8s)',
      'Increasing temperature exponentially',
      'Logging errors in exponential format',
    ],
    correctAnswer: 1,
    explanation:
      "Exponential backoff doubles the wait time between retries: 1s, 2s, 4s, 8s, 16s. This gives the API progressively more time to recover from overload and prevents hammering a failing service. It's the industry-standard retry strategy.",
  },
  {
    id: 'mc4',
    question: 'What is "jitter" in retry logic?',
    options: [
      'Random noise in responses',
      'Random variance added to wait times to prevent thundering herd',
      'Shaking the server rack',
      'A type of error message',
    ],
    correctAnswer: 1,
    explanation:
      'Jitter adds random variance to retry delays (e.g., wait 2s ± 0.5s random) to desynchronize retries from multiple clients. Without jitter, all clients retry simultaneously, potentially re-triggering the same issue. Jitter spreads load over time.',
  },
  {
    id: 'mc5',
    question: 'When should a circuit breaker open?',
    options: [
      'After every error',
      'After a threshold of failures (e.g., 5 in 10 requests)',
      'When cost exceeds budget',
      'Circuit breakers are not used with LLMs',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breakers open after a failure threshold (e.g., 5 failures in last 10 requests or 50% error rate). Opening blocks all requests for a timeout period, preventing cascading failures and giving the API time to recover. After timeout, it enters half-open state to test recovery.',
  },
];
