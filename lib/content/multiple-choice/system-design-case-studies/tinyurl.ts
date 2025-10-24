/**
 * Multiple choice questions for Design TinyURL section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const tinyurlMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'You are designing a URL shortener that needs to support 60 billion URLs over 10 years. Using base62 encoding (0-9, a-z, A-Z), what is the minimum short URL length required?',
    options: [
      '5 characters (62^5 = 916 million)',
      '6 characters (62^6 = 56 billion)',
      '7 characters (62^7 = 3.5 trillion)',
      '8 characters (62^8 = 218 trillion)',
    ],
    correctAnswer: 2,
    explanation:
      'Base62 provides 62^n possible combinations. 62^6 = 56 billion is close but insufficient for 60 billion URLs (no safety margin). 62^7 = 3.5 trillion provides 50x headroom, which is ideal for growth and avoiding conflicts. In production, always size for 10-100x headroom, so 7 characters is the correct choice.',
  },
  {
    id: 'mc2',
    question:
      'Your URL shortener uses hash-based short URL generation (MD5 + first 6 chars). As the database grows to 10 billion URLs, what problem will you increasingly face?',
    options: [
      'Slower hash computation',
      'Increased collision rate requiring more database checks',
      'Hash function will run out of outputs',
      'URLs will become longer',
    ],
    correctAnswer: 1,
    explanation:
      'With hash-based generation, collision probability increases as the table fills (birthday paradox). At 10B URLs with 6-char keys (62^6 = 56B possible), collision rate becomes significant. Each collision requires: (1) DB query to check existence, (2) Regenerate hash with suffix/counter, (3) Repeat until unique. This adds latency and DB load. This is why base62 encoding with auto-increment IDs is preferred - zero collisions guaranteed.',
  },
  {
    id: 'mc3',
    question:
      'Your URL shortener is serving 20,000 redirects/second with 80% cache hit rate. Your database can handle 5,000 QPS. What happens if Redis cache goes down?',
    options: [
      'System continues normally (cache is not critical)',
      'Database is immediately overloaded with 20,000 QPS and system goes down',
      'Load balancer will queue requests until cache returns',
      'Application servers will automatically reduce traffic',
    ],
    correctAnswer: 1,
    explanation:
      'With 80% cache hit rate, only 20% × 20K = 4K QPS hits the database. If cache fails, all 20K QPS hits the database, which can only handle 5K QPS. Database gets overloaded, queries slow down, timeouts occur, system cascades to failure. MITIGATIONS: (1) Redis replication (3-5 replicas), (2) Circuit breaker pattern to fail fast, (3) Application-level cache as fallback, (4) Rate limiting to protect database. This demonstrates why cache is a critical dependency, not just an optimization.',
  },
  {
    id: 'mc4',
    question: 'When should you shard your URL shortener database?',
    options: [
      'Immediately at design time (always shard for distributed systems)',
      'When single database cannot handle write/read load or storage',
      'Never - URL shorteners should use NoSQL instead',
      'When you have more than 1 million URLs',
    ],
    correctAnswer: 1,
    explanation:
      'Start with single database + read replicas. Shard only when: (1) Write load exceeds single DB capacity (~10K writes/sec for MySQL), (2) Storage exceeds single server (tens of TBs), or (3) Read replicas cannot handle read load. For the given scenario (200 writes/sec, 20K reads/sec, 36 TB over 10 years), single primary + 5-10 read replicas is sufficient. Premature sharding adds massive operational complexity. The 1M URL threshold is arbitrary and far too early.',
  },
  {
    id: 'mc5',
    question:
      'A viral celebrity shares one of your shortened URLs, causing traffic to spike from 20,000 to 200,000 requests/sec for that specific URL. What is your best immediate mitigation?',
    options: [
      'Add more database replicas',
      'Switch from 302 to 301 redirects',
      'Cache hit rate increases to ~99.9% for that URL, naturally handling the spike',
      'Implement rate limiting per URL',
    ],
    correctAnswer: 2,
    explanation:
      'This is the beauty of caching! For a single hot URL, first request loads it into Redis, then ALL subsequent requests (199,999 of them) are served from cache with sub-millisecond latency. Even if cache hit rate for entire system is 80%, a viral URL approaches 99.9%+ cache hit rate. Redis can handle millions of QPS for a single key. Database sees only cache misses from OTHER URLs. Adding DB replicas or switching to 301 does nothing for the immediate spike. Rate limiting would hurt user experience. This is why caching is critical for URL shorteners - viral URLs naturally stay hot in cache.',
  },
];
