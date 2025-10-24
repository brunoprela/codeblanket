/**
 * Multiple choice questions for Rate Limiting Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const ratelimitingalgorithmsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of Token Bucket algorithm over Leaky Bucket?',
    options: [
      'Uses less memory',
      'Allows bursts of requests when tokens are available',
      'Provides smoother output rate',
      'Easier to implement',
    ],
    correctAnswer: 1,
    explanation:
      'Token Bucket allows bursts up to the bucket capacity by consuming accumulated tokens. If a user has not made requests for a while, tokens accumulate, allowing a burst of requests. This provides better user experience than Leaky Bucket which enforces constant output rate. Stripe and AWS use Token Bucket for this reason.',
  },
  {
    id: 'mc2',
    question:
      'What is the boundary burst problem in Fixed Window Counter algorithm?',
    options: [
      'Requests are always rejected at window boundaries',
      'Users can make 2x the limit by making requests at the end of one window and start of next',
      'The algorithm stops working at midnight',
      'Memory usage doubles at boundaries',
    ],
    correctAnswer: 1,
    explanation:
      'Fixed Window resets the counter at window boundaries. A user can make 100 requests at 00:59 (end of window 1) and 100 at 01:00 (start of window 2), totaling 200 requests in 1 minute—double the limit! Sliding Window Counter solves this with weighted averaging.',
  },
  {
    id: 'mc3',
    question:
      'Which rate limiting algorithm is recommended for production systems?',
    options: [
      'Fixed Window Counter',
      'Sliding Window Counter',
      'No rate limiting needed',
      'Simple request counter without windows',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding Window Counter is production-standard (used by Cloudflare, Kong). It provides: (1) Accurate enforcement without boundary bursts, (2) Memory efficiency (only 2 counters), (3) Smooth rate limiting. Better than Fixed Window (boundary issue) and Sliding Log (memory intensive).',
  },
  {
    id: 'mc4',
    question:
      'How do you implement distributed rate limiting across multiple servers?',
    options: [
      'Each server maintains independent rate limits',
      'Use Redis with atomic operations for centralized state',
      'No solution exists',
      'Require all servers to communicate constantly',
    ],
    correctAnswer: 1,
    explanation:
      'Redis provides centralized rate limit state with atomic operations (INCR, EXPIRE). Multiple servers check/update the same Redis keys. Lua scripts ensure atomic read-modify-write. This is production-proven (Cloudflare, AWS API Gateway). Memory efficient and scales to millions of requests/second.',
  },
  {
    id: 'mc5',
    question: 'Which production systems implement rate limiting?',
    options: [
      'Only academic research systems',
      'Stripe API, GitHub API, Twitter API, Cloudflare',
      'Only government systems',
      'Only internal systems',
    ],
    correctAnswer: 1,
    explanation:
      'Rate limiting is industry-standard: Stripe (token bucket, 100/sec), GitHub (fixed window, 5000/hour), Twitter (sliding window, 300/15min), Cloudflare (sliding window counter), AWS API Gateway (token bucket). Every public API implements rate limiting to prevent abuse and ensure fairness.',
  },
];
