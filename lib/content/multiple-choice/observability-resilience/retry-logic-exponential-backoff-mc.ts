/**
 * Multiple choice questions for Retry Logic & Exponential Backoff section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const retryLogicMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is exponential backoff, and why is jitter important?',
    options: [
      'Waiting a fixed time between retries, jitter adds randomness',
      'Doubling the wait time between retries (1s, 2s, 4s), jitter adds randomness to prevent thundering herd',
      'Retrying immediately without waiting',
      'Waiting exponentially longer than the previous success',
    ],
    correctAnswer: 1,
    explanation:
      'Exponential backoff doubles the delay between retries (1s, 2s, 4s, 8s) to give the service time to recover and reduce load. Jitter adds randomness (e.g., wait 0-2s instead of exactly 2s) to prevent thundering herd: Without jitter, 1000 clients all retry at exact same time (synchronized) → overwhelm service. With jitter, retries spread across time window → service can recover. AWS recommends full jitter: delay = random(0, baseDelay × 2^attempt).',
  },
  {
    id: 'mc2',
    question: 'Which HTTP status codes should generally NOT be retried?',
    options: [
      '2xx success codes',
      '4xx client errors (except 429 Too Many Requests)',
      '5xx server errors',
      'All status codes should be retried',
    ],
    correctAnswer: 1,
    explanation:
      "4xx client errors should NOT be retried (except 429) because they indicate permanent failures: 400 Bad Request (invalid input), 401 Unauthorized (invalid credentials), 404 Not Found (resource doesn't exist). Retrying won't fix these. SHOULD retry: 5xx server errors (500, 502, 503, 504 - might be transient), network timeouts, connection errors. Exception: 429 Too Many Requests SHOULD be retried with backoff (rate limit will clear).",
  },
  {
    id: 'mc3',
    question: 'What is idempotency, and why is it important for retry logic?',
    options: [
      'Making operations faster',
      'An operation producing the same result no matter how many times executed, essential for safe retries',
      'Retry logic without delays',
      'Distributing load across servers',
    ],
    correctAnswer: 1,
    explanation:
      'Idempotency means an operation produces the same result when executed multiple times. Essential for retries because network failures can cause duplicate requests. Example: POST /payment {amount: 100} is NOT idempotent (10 retries = $1000 charge!). PUT /payment/order-123 {amount: 100} IS idempotent (10 retries = single $100 charge). Solution: Idempotency keys - server caches results, returns cached response for duplicate keys. Stripe requires idempotency keys for all payments.',
  },
  {
    id: 'mc4',
    question: 'What is a retry budget, and why is it important?',
    options: [
      'Money allocated for retry logic',
      'Limit on total retries (e.g., max 10% of requests can be retries) to prevent retry amplification',
      'Number of retry attempts allowed',
      'Time allowed for retries',
    ],
    correctAnswer: 1,
    explanation:
      "Retry budget limits total retries to prevent retry amplification. Example without budget: Service failing → 10,000 req/s → All retry 3 times → 30,000 req/s total → Service overwhelmed → Can't recover → Retry storm. With budget (10%): Allow max 1,000 retries/s. If exceeded, fail fast without retry. This prevents retries from making the problem worse and allows service to recover. Netflix, AWS, Google use strict retry budgets.",
  },
  {
    id: 'mc5',
    question: 'When is the best time to give up retrying?',
    options: [
      'After 1 retry',
      'After 3-5 retries with exponential backoff, or when receiving non-retriable error (4xx)',
      'Never give up, retry forever',
      'After 100 retries',
    ],
    correctAnswer: 1,
    explanation:
      'Give up after: (1) 3-5 retry attempts with exponential backoff (most transient issues resolve by then), (2) Non-retriable error received (4xx client error), (3) Total time exceeds threshold (e.g., 60 seconds total), (4) Retry budget exhausted. Retrying forever wastes resources and delays error reporting. Typical config: 3-5 retries with backoff (1s, 2s, 4s, 8s, 16s), max 60s total, fail fast on 4xx. Balance: Enough retries to handle transient failures, not so many that it masks permanent failures.',
  },
];
