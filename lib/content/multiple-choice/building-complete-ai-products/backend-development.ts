import { MultipleChoiceQuestion } from '../../../types';

export const backendDevelopmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-bd-mc-1',
    question:
      'Why is a hybrid API architecture (REST + SSE + async jobs) recommended for AI applications?',
    options: [
      'REST handles everything',
      'Different needs: REST (CRUD), SSE (streaming chat), async jobs (long tasks like image gen)',
      'Only use WebSocket for everything',
      'Avoid APIs entirely',
    ],
    correctAnswer: 1,
    explanation:
      'Hybrid optimizes for each use case: (1) REST: CRUD operations (users, documents), simple, cacheable, (2) SSE: Streaming (chat), unidirectional, HTTP-based, auto-reconnect, (3) Async jobs: Long-running (image/video gen), return job_id immediately, poll for status. Using only REST would block API servers. Only WebSocket is overkill (bidirectional not needed). Hybrid provides best performance and UX per feature type.',
  },
  {
    id: 'bcap-bd-mc-2',
    question:
      'How should LLM provider failures be handled with multiple providers (OpenAI, Anthropic, Cohere)?',
    options: [
      'Fail immediately if primary fails',
      'Circuit breaker: 5 failures → open (skip) for 60s → half-open (test) → closed, automatic fallback',
      'Never retry',
      'Always use all providers simultaneously',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breaker pattern: (1) Track failures per provider, (2) Closed state: Normal operation, (3) After 5 consecutive failures → Open: Skip provider for 60s (give time to recover), (4) Half-open: Try 1 request after 60s, if success → Closed, if failure → Open again, (5) Automatic fallback: Primary fails → try secondary → tertiary. This prevents cascading failures and wasted retry attempts on failing providers.',
  },
  {
    id: 'bcap-bd-mc-3',
    question:
      'What is the best database partitioning strategy for message tables with 1M+ messages/day?',
    options: [
      'No partitioning needed',
      'Partition by month (created_at), archive old partitions to S3 after 90 days',
      'Partition by user_id',
      'Partition randomly',
    ],
    correctAnswer: 1,
    explanation:
      'Time-based partitioning: (1) Partition messages by created_at (monthly), (2) Recent data (last 3 months) stays in PostgreSQL (fast queries), (3) Archive old partitions to S3 (cheap storage), (4) Benefits: Fast queries on recent data (only scan relevant partition), easy archival, manage table size. User-based partitioning causes uneven distribution. 1M msgs/day = 30M/month = manageable with partitioning, unmanageable without.',
  },
  {
    id: 'bcap-bd-mc-4',
    question:
      'How should rate limiting be implemented for different user tiers (free, pro, enterprise)?',
    options: [
      'No rate limiting',
      'Redis sliding window: track requests in last minute/hour, check count < tier_limit before processing',
      'Count in PostgreSQL',
      'Unlimited for everyone',
    ],
    correctAnswer: 1,
    explanation:
      'Redis sliding window: (1) Store request timestamps in Redis (sorted set), (2) Before request: count timestamps in last minute/hour, (3) If count >= tier_limit → 429 error, (4) Tiers: free (10/min, 100/hr), pro (100/min, 1000/hr), enterprise (unlimited), (5) Fast (sub-millisecond), (6) Automatic cleanup (TTL). PostgreSQL too slow for rate limiting (10-50ms per check vs <1ms Redis). This protects infrastructure while enabling tiered pricing.',
  },
  {
    id: 'bcap-bd-mc-5',
    question:
      'What is the recommended approach for tracking costs per request in an AI application?',
    options: [
      "Don't track costs",
      "Middleware logs: user_id, endpoint, tokens, cost_usd, async insert to DB (don't block request)",
      'Track only monthly totals',
      'Store in text files',
    ],
    correctAnswer: 1,
    explanation:
      "Async cost tracking middleware: (1) After each LLM call, calculate cost (input_tokens × rate + output_tokens × rate), (2) Log: user_id, endpoint, model, tokens, cost_usd, timestamp, (3) Async insert to database (don't block request), (4) Aggregate: Daily summary per user, monthly trends, cost per feature, (5) Alert if: Daily spend >2x expected, user costs >$50/day. This enables: per-user billing, cost optimization, profitability analysis without adding latency.",
  },
];
