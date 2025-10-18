/**
 * Multiple choice questions for In-Memory vs Persistent Storage section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const inmemoryvspersistentstorageMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'What is the primary advantage of in-memory storage over persistent storage?',
      options: [
        'Data durability across restarts',
        'Lower cost per gigabyte',
        'Extremely fast access (100x faster)',
        'Support for complex SQL queries',
      ],
      correctAnswer: 2,
      explanation:
        'In-memory storage (RAM) is 100-1000x faster than persistent storage (disk), with latencies of 0.1-1ms vs 10-100ms. This speed comes at the cost of volatility (data loss on restart), higher cost per GB, and limited capacity. Persistent storage offers durability, lower cost, and complex queries.',
    },
    {
      id: 'mc2',
      question: 'Which data should NOT be stored solely in an in-memory cache?',
      options: [
        'User session tokens',
        'API rate limit counters',
        'Financial transaction records',
        'Product catalog details',
      ],
      correctAnswer: 2,
      explanation:
        'Financial transaction records must be stored in persistent storage (database) because they require durability and cannot be lost. Session tokens, rate limit counters, and product catalog can be cached because they can be regenerated or have acceptable loss risk. Always use persistent storage as source of truth for critical data.',
    },
    {
      id: 'mc3',
      question:
        'What is the typical cache hit rate target for a well-configured cache?',
      options: ['50%', '70%', '90%', '99%'],
      correctAnswer: 2,
      explanation:
        "A well-configured cache should achieve >90% hit rate, meaning 90%+ of requests are served from cache. If hit rate is <90%, the cache is likely too small, TTL is too short, or you're caching the wrong data. Monitor cache hit rate as a key metric.",
    },
    {
      id: 'mc4',
      question:
        'What is the main disadvantage of Redis AOF (Append-Only File) compared to RDB?',
      options: [
        'More data loss on crash',
        'Larger file size and slower restarts',
        'Cannot be used in production',
        'Requires more expensive hardware',
      ],
      correctAnswer: 1,
      explanation:
        "AOF produces larger files (every operation logged) and has slower restarts (must replay all operations), compared to RDB's compact snapshots. However, AOF has much less data loss (1 second max vs minutes). RDB is faster but loses more data. Many production systems use both (hybrid).",
    },
    {
      id: 'mc5',
      question: 'What is the cache-aside pattern?',
      options: [
        'Writing to cache and database simultaneously',
        'Application checks cache first, queries database on miss, and populates cache',
        'Database automatically populates cache',
        'Cache automatically writes to database',
      ],
      correctAnswer: 1,
      explanation:
        'Cache-aside (lazy loading) means the application: (1) Checks cache first, (2) On cache miss, queries database, (3) Populates cache with result, (4) Returns data. The application explicitly manages the cache. This is different from write-through (write to both) or database-managed caching.',
    },
  ];
