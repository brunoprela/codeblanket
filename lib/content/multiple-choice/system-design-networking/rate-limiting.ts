/**
 * Multiple choice questions for Rate Limiting & Throttling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const ratelimitingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'rate-limit-algorithm',
    question:
      'Which rate limiting algorithm allows burst traffic while maintaining a long-term average rate?',
    options: [
      'Fixed Window Counter',
      'Leaky Bucket',
      'Token Bucket',
      'Sliding Window Log',
    ],
    correctAnswer: 2,
    explanation:
      "Token Bucket allows burst traffic because clients can accumulate tokens up to the bucket capacity and use them all at once. However, the long-term rate is limited by the refill rate. Leaky Bucket processes at a constant rate (no bursts). Fixed Window has boundary burst problems. Sliding Window Log is accurate but doesn't specifically enable bursts.",
  },
  {
    id: 'rate-limit-distributed',
    question:
      'Why is a Lua script preferred over multiple Redis commands for distributed rate limiting?',
    options: [
      'Lua scripts are faster than Redis commands',
      'Lua scripts provide atomic execution, preventing race conditions across multiple API servers',
      'Lua scripts use less memory than Redis data structures',
      'Lua scripts automatically distribute across Redis cluster nodes',
    ],
    correctAnswer: 1,
    explanation:
      "Lua scripts execute atomically in Redis, meaning all operations complete without interruption. This prevents race conditions when multiple API servers check and update rate limit counters simultaneously. Without atomicity, two servers could both check the counter (e.g., 99 requests), both see it's below the limit (100), and both allow the request, resulting in 101 requests. Lua scripts ensure the check-and-increment happens atomically.",
  },
  {
    id: 'rate-limit-429',
    question:
      'A user receives a 429 Too Many Requests response. Which HTTP header should the API return to indicate when the user can retry?',
    options: [
      'X-RateLimit-Reset',
      'Retry-After',
      'X-RateLimit-Remaining',
      'Cache-Control',
    ],
    correctAnswer: 1,
    explanation:
      'Retry-After (RFC 6585) specifically indicates when the client should retry after being rate limited, either as seconds (e.g., "60") or HTTP date. X-RateLimit-Reset shows when the rate limit window resets, X-RateLimit-Remaining shows remaining requests, and Cache-Control is for caching, not rate limiting.',
  },
  {
    id: 'rate-limit-placement',
    question:
      'Where should rate limiting middleware be placed in the request processing pipeline to best protect against authentication brute-force attacks?',
    options: [
      'After authentication, to only limit authenticated users',
      'Before authentication, to limit all login attempts including invalid ones',
      'After database queries, to limit only successful logins',
      'In a separate microservice, to avoid impacting API performance',
    ],
    correctAnswer: 1,
    explanation:
      'Rate limiting must occur BEFORE authentication to protect against brute-force attacks. If rate limiting happens after authentication, an attacker can make unlimited login attempts with invalid credentials, potentially guessing passwords or causing a DDoS. Rate limiting before authentication (by IP address) limits all login attempts, preventing such attacks.',
  },
  {
    id: 'rate-limit-boundary-problem',
    question:
      'The Fixed Window Counter algorithm has a "boundary problem" where users can potentially send double the rate limit in a short period. Which algorithm solves this issue while remaining memory efficient?',
    options: [
      'Leaky Bucket',
      'Token Bucket',
      'Sliding Window Counter',
      'Fixed Window Log',
    ],
    correctAnswer: 2,
    explanation:
      "Sliding Window Counter solves the boundary problem by using a weighted average of the previous and current window counts, providing accurate rate limiting without bursts at boundaries. It's memory efficient (stores only 2 counters per user). Leaky Bucket prevents bursts but queues requests. Token Bucket allows bursts. Fixed Window Log is accurate but memory intensive (stores all timestamps).",
  },
];
