/**
 * Multiple choice questions for Back-of-the-Envelope Estimations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const backofenvelopeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Approximately how many seconds are in one day? (Choose the best approximation)',
    options: [
      '10,000 seconds',
      '86,400 seconds (~100K)',
      '1 million seconds',
      '10 million seconds',
    ],
    correctAnswer: 1,
    explanation:
      "1 day = 24 hours × 60 minutes × 60 seconds = 86,400 seconds. For back-of-envelope calculations, round to ~100K seconds. This makes math easier: If you have 1M requests/day, that's 1M / 100K ≈ 10 requests/second. Memorize: 1 day ≈ 100K sec, 1 month ≈ 2.5M sec, 1 year ≈ 30M sec.",
  },
  {
    id: 'mc2',
    question:
      'Instagram has 500M daily active users, each viewing 100 photos per day. What is the approximate read QPS (queries per second)?',
    options: ['5,000 QPS', '50,000 QPS', '500,000 QPS', '5,000,000 QPS'],
    correctAnswer: 2,
    explanation:
      'Calculation: 500M users × 100 photos = 50 billion reads/day. 50B / 86,400 seconds ≈ 50B / 100K = 500K QPS (average). During peak hours (2-3× average), could be 1-1.5M QPS. This high read rate drives architectural decisions: heavy caching (Redis, CDN), read replicas, eventual consistency acceptable.',
  },
  {
    id: 'mc3',
    question:
      'Twitter stores 500M tweets per day. Each tweet is 280 characters (~280 bytes). How much storage is needed per day for text only (no media)?',
    options: ['14 MB', '140 MB', '140 GB', '1.4 TB'],
    correctAnswer: 2,
    explanation:
      'Calculation: 500M tweets × 280 bytes = 140,000 MB = 140 GB per day (text only). With metadata (user_id, timestamp, etc.), probably 2-3× this. With media (photos/videos), total storage is much higher - typically 10-50 TB/day. This small text size (140 GB) is why Twitter can store all historical tweets, but media requires CDN and tiered storage.',
  },
  {
    id: 'mc4',
    question:
      'Why should you design systems for PEAK traffic rather than AVERAGE traffic?',
    options: [
      'To waste resources and increase costs',
      "Because averages don't matter in production",
      'Systems crash if designed for average but peak exceeds capacity',
      'Peak and average are always the same',
    ],
    correctAnswer: 2,
    explanation:
      'Systems must handle peak load or they crash when most needed. Example: E-commerce designed for 10K average QPS crashes when Black Friday brings 100K QPS. Result: Lost sales, angry customers, brand damage. Solution: Design for peak (with buffer), use auto-scaling to reduce costs during average periods. Peak is typically 2-10× average depending on application type.',
  },
  {
    id: 'mc5',
    question:
      'A database can handle 1,000 writes per second. Your system needs 50,000 writes/sec. How many database shards do you need minimum?',
    options: ['5 shards', '10 shards', '50 shards', '100 shards'],
    correctAnswer: 2,
    explanation:
      'Minimum: 50,000 / 1,000 = 50 shards. In practice, add 30-50% buffer for: (1) Uneven distribution (some shards may be hotter). (2) Peak traffic spikes. (3) Maintenance (taking shards offline). (4) Future growth. So deploy 65-75 shards. This calculation shows why Twitter/Instagram use NoSQL databases (Cassandra) that handle 10-100× more writes per node than traditional SQL.',
  },
];
