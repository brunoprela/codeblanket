export const backendDevelopmentQuiz = [
  {
    id: 'bcap-bd-q-1',
    question:
      'Design the API architecture for an AI application with: streaming chat, document upload, image generation (2-5min wait), and analytics. Requirements: (1) Support 10k concurrent connections, (2) Handle long-running tasks, (3) Rate limit by user tier, (4) Track costs per request. Should you use: REST, GraphQL, gRPC, or a hybrid? Include load balancing and caching strategies.',
    sampleAnswer:
      "Hybrid architecture: (1) REST for CRUD (user, documents, settings) - simple, cacheable, standard. (2) SSE for streaming (chat) - unidirectional, HTTP-based, auto-reconnect. (3) Async jobs for long tasks (image gen) - return job_id immediately, poll status. Infrastructure: (1) Load balancer (Nginx): Round-robin across API servers, sticky sessions for SSE (connection affinity). (2) API servers (FastAPI): Stateless, horizontally scalable (10+ instances), handle REST + SSE endpoints. (3) Worker pool (Celery): Separate from API, scales independently, processes background jobs (image gen, document processing). (4) Redis: Session store, rate limiting, pub/sub for SSE. (5) PostgreSQL: Persistent data (users, jobs, analytics). Streaming: API server holds SSE connection, subscribes to Redis channel (chat:{conversation_id}), LLM tokens published to channel, API streams to client. Disconnection: Client reconnects with Last-Event-ID, server replays missed events. Long tasks: POST /images/generate → Create job in Redis → Return {job_id} → Worker picks up → Client polls GET /jobs/{job_id}. Alternative: WebSocket for job updates (avoid polling). Rate limiting: Redis-based sliding window (store request timestamps). Check before processing: if count > tier_limit in last hour → 429 error. Cost tracking: Middleware logs: user_id, endpoint, tokens (input/output), cost_usd. Async insert to DB (don't block request). Caching: (1) Redis for LLM responses (semantic cache: hash prompt → response, 1hr TTL). (2) CDN for static (images, assets). (3) Application cache for user profiles (5min TTL). Concurrency: 10k connections = 10k SSE streams. Each API server handles ~1k connections (event loop). Use 10-20 servers.",
    keyPoints: [
      'Hybrid: REST (CRUD), SSE (streaming), async jobs (long tasks)',
      'Separate API servers (stateless) from workers (background jobs)',
      'Redis: rate limiting, session store, pub/sub for SSE',
      'Semantic cache: hash prompts, store responses (1hr TTL), 70%+ hit rate',
      'Concurrency: each server handles ~1k SSE connections, scale horizontally',
    ],
  },
  {
    id: 'bcap-bd-q-2',
    question:
      'Your AI application calls multiple LLM providers (OpenAI, Anthropic, Cohere) with automatic fallback. Design the provider management system: (1) How do you route requests, (2) Handle rate limits per provider, (3) Implement circuit breaker for failing providers, (4) Track costs and latency per provider, (5) Support A/B testing models? Include failover strategy.',
    sampleAnswer:
      'Multi-provider orchestration: (1) Routing strategy: Priority-based with fallback chain: [Anthropic (primary), OpenAI (fallback), Cohere (last resort)]. Route based on: model capability (coding→Claude, general→GPT), cost (try cheaper first if quality acceptable), latency (fastest provider). (2) Rate limiting: Track per-provider requests in Redis: provider:{name}:minute → count. Before call, check if < provider_limit (Anthropic: 1000 RPM, OpenAI: 3000 RPM). If exceeded, skip to next provider. (3) Circuit breaker: State machine per provider: Closed (healthy) → Open (failing) → Half-Open (testing recovery). Closed: normal operation. On 5 consecutive failures → Open. Open: skip provider for 60s. After 60s → Half-Open: try 1 request. Success → Closed. Failure → Open again. (4) Provider abstraction: Unified interface: class LLMProvider { async generate (prompt, options): response }. Implementations: AnthropicProvider, OpenAIProvider, CohereProvider. Normalize responses to common format. (5) Cost tracking: Log per request: provider, model, input_tokens, output_tokens, cost_usd. Aggregate daily: which provider is cheapest for workload. (6) Latency tracking: Measure time-to-first-token, total latency. Use p95 for provider selection. (7) A/B testing: 10% traffic → experimental model, 90% → production. Compare: quality (thumbs up rate), cost, latency. Failover: If primary fails → immediate fallback, log event, alert if failure rate >5%. Retry: 3x with exponential backoff before declaring failure.',
    keyPoints: [
      'Priority-based routing with fallback chain (primary → backup → last resort)',
      'Track rate limits per provider in Redis, skip if exceeded',
      'Circuit breaker: 5 failures → open for 60s → half-open (test recovery)',
      'Unified interface: abstract provider differences, normalize responses',
      'Track cost + latency per provider, use data for routing decisions',
    ],
  },
  {
    id: 'bcap-bd-q-3',
    question:
      'Design the database schema for an AI chat application with: users, conversations, messages, usage tracking, and billing. Requirements: (1) Efficiently query conversation history, (2) Calculate monthly usage per user, (3) Support message search, (4) Handle 1M+ messages/day, (5) Audit trail for compliance. Compare: PostgreSQL, MongoDB, and a hybrid approach.',
    sampleAnswer:
      'Hybrid: PostgreSQL (primary) + Elasticsearch (search) + S3 (archive). PostgreSQL schema: users (id, email, tier, credits), conversations (id, user_id, title, created_at), messages (id, conversation_id, role, content, tokens, cost, created_at), usage_logs (id, user_id, date, tokens, cost). Indexes: (1) messages (conversation_id, created_at) - fast conversation retrieval. (2) messages (conversation_id, role, created_at DESC) - last assistant message. (3) usage_logs (user_id, date) - monthly aggregation. (4) Partitioning: Partition messages by created_at (monthly partitions), old partitions moved to S3 (after 90 days). Conversation history: Query messages WHERE conversation_id = X ORDER BY created_at LIMIT 20 (fast with index). For full history (1000+ messages): Paginate, lazy load older messages. Monthly usage: Pre-aggregate: Insert/update usage_logs daily (cron job), calculate: SUM(tokens), SUM(cost) per user per day. Monthly query: Simple SUM over 30-day partition. Message search: Sync messages to Elasticsearch (async worker), full-text search on content. Query: "user:123 AND content:keyword", return message IDs, fetch from PostgreSQL. Scale: 1M messages/day = ~12 inserts/sec (easy for PostgreSQL). Use connection pooling (PgBouncer), read replicas for analytics queries. Audit trail: Store all API calls: audit_logs (id, user_id, endpoint, request_body, response_status, created_at). Append-only table, partition monthly, retain 2 years. MongoDB comparison: Flexible schema (good for nested messages), but: weaker JOIN support, complex aggregations slower, less mature tooling. PostgreSQL: ACID, JSON support (JSONB), mature, cost-effective. Recommendation: PostgreSQL unless specific NoSQL requirement.',
    keyPoints: [
      'PostgreSQL primary: strong consistency, JSONB for flexibility, mature tooling',
      'Partition messages by month, archive to S3 after 90 days',
      'Pre-aggregate usage daily (usage_logs), monthly queries fast',
      'Elasticsearch for full-text search, PostgreSQL for structured data',
      'Scale: 1M msgs/day easy, connection pooling, read replicas for analytics',
    ],
  },
];
