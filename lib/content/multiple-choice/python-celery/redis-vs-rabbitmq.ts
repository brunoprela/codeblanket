/**
 * Multiple choice questions for Redis vs RabbitMQ section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const redisVsRabbitmqMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Which broker is faster?',
    options: [
      'RabbitMQ (disk-backed)',
      'Redis (in-memory)',
      'Both same speed',
      'Depends on task type',
    ],
    correctAnswer: 1,
    explanation:
      "Redis is ~2× faster than RabbitMQ because it's in-memory. RabbitMQ writes to disk (persistence) which adds latency. Redis: <1ms latency. RabbitMQ: ~10ms latency. Trade-off: Speed (Redis) vs Reliability (RabbitMQ).",
  },
  {
    id: 'mc2',
    question: 'Which broker is more reliable?',
    options: [
      'Redis (can lose tasks on crash)',
      'RabbitMQ (disk-backed persistence)',
      'Both equally reliable',
      'Neither is reliable',
    ],
    correctAnswer: 1,
    explanation:
      "RabbitMQ more reliable because it's disk-backed (persistent). Redis is in-memory - tasks lost if Redis crashes (unless AOF enabled). For critical tasks (payments, orders), use RabbitMQ. For non-critical (emails), Redis acceptable.",
  },
  {
    id: 'mc3',
    question: 'When should you use RabbitMQ over Redis?',
    options: [
      'Never, Redis is always better',
      'For critical tasks where reliability > speed',
      'Only for small projects',
      'When you need faster performance',
    ],
    correctAnswer: 1,
    explanation:
      'Use RabbitMQ when: (1) Critical tasks (payments, orders), (2) Reliability > speed, (3) >1M tasks/day, (4) Complex routing needed, (5) Multi-DC. RabbitMQ guarantees task delivery. Redis faster but can lose tasks.',
  },
  {
    id: 'mc4',
    question: 'What is a hybrid Redis + RabbitMQ approach?',
    options: [
      'Use both brokers simultaneously for different task types',
      'Use Redis as backup for RabbitMQ',
      'Use RabbitMQ as backup for Redis',
      'Not possible to use both',
    ],
    correctAnswer: 0,
    explanation:
      'Hybrid approach: Use Redis for non-critical tasks (emails, analytics) and RabbitMQ for critical tasks (payments, orders). Create two Celery apps with different brokers. Benefits: Speed for non-critical + reliability for critical + cost optimization.',
  },
  {
    id: 'mc5',
    question: 'Which broker is easier to set up?',
    options: [
      'RabbitMQ (requires Erlang + RabbitMQ)',
      'Redis (just redis-server)',
      'Both equally easy',
      'Neither is easy',
    ],
    correctAnswer: 1,
    explanation:
      'Redis much easier: Install redis-server, run it, done (5 minutes). RabbitMQ: Install Erlang, install RabbitMQ, configure, enable plugins (30 minutes). For MVPs/startups, Redis simpler. For production with reliability needs, worth RabbitMQ complexity.',
  },
];
