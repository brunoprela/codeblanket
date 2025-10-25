/**
 * Multiple choice questions for Latency vs Throughput section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const latencyvsthroughputMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is latency?',
    options: [
      'The number of requests processed per second',
      'The time taken to complete a single operation',
      'The total capacity of the system',
      'The number of concurrent users',
    ],
    correctAnswer: 1,
    explanation:
      'Latency is the time taken to complete a single operation, measured in milliseconds or seconds. It represents how fast an individual request completes. This is different from throughput, which measures how many operations complete per unit time.',
  },
  {
    id: 'mc2',
    question: 'Which system should prioritize throughput over latency?',
    options: [
      'Real-time video game server',
      'REST API for mobile app',
      'Nightly batch job processing 10M records',
      'Stock trading platform',
    ],
    correctAnswer: 2,
    explanation:
      'Batch jobs that process large datasets should prioritize throughput (total work completed) over latency (time per item). No user is waiting for individual records. Video games, mobile APIs, and trading platforms are latency-critical because users are waiting for responses.',
  },
  {
    id: 'mc3',
    question:
      "According to Little\'s Law, if you want 1,000 req/s throughput with 200ms latency per request, how many concurrent requests must you handle?",
    options: [
      '50 concurrent requests',
      '100 concurrent requests',
      '200 concurrent requests',
      '500 concurrent requests',
    ],
    correctAnswer: 2,
    explanation:
      "Little\'s Law: Concurrency = Throughput × Latency = 1,000 req/s × 0.2s = 200 concurrent requests. This means you need to handle 200 requests in flight simultaneously to achieve your throughput goal at that latency.",
  },
  {
    id: 'mc4',
    question: 'Why should you measure P99 latency instead of average latency?',
    options: [
      'P99 is easier to calculate',
      'Average latency hides outliers that affect user experience',
      'P99 is always lower than average',
      'Average latency is never used in production',
    ],
    correctAnswer: 1,
    explanation:
      'Average latency can be misleading because it hides outliers. If 1% of requests take 10 seconds while 99% take 50ms, the average might look good, but 1 in 100 users gets a terrible experience. P99 tells you the latency that 99% of users experience or better.',
  },
  {
    id: 'mc5',
    question: 'What technique increases throughput at the cost of latency?',
    options: ['Caching', 'Indexing', 'Batching', 'CDN'],
    correctAnswer: 2,
    explanation:
      'Batching increases throughput by processing multiple operations together, reducing per-operation overhead. However, individual operations have higher latency because they wait for the batch to fill. This is ideal for background jobs but not for user-facing operations.',
  },
];
