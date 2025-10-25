export const productArchitectureDesignQuiz = [
  {
    id: 'bcap-pad-q-1',
    question:
      'Design the complete architecture for an AI code editor like Cursor. Include: frontend components, backend services, database schema, caching strategy, and external API integrations. How would you handle real-time collaboration, context management (100k+ line codebases), and cost optimization? Justify your technology choices.',
    sampleAnswer:
      'Architecture: Frontend (VS Code fork with Monaco editor), Backend (FastAPI with async workers), PostgreSQL (user data, projects), Redis (cache, rate limiting), Qdrant (vector DB for codebase embeddings), S3 (file storage). Real-time: WebSocket server with CRDT (Yjs) for concurrent edits + presence. Context: Parse codebase into AST, chunk by function/class, embed with Voyage, store in Qdrant. Query relevant code before each completion. Cost: Cache embeddings (regenerate only on file changes), use GPT-4o-mini for simple completions, Claude Sonnet for complex. Rate limit: 500 requests/hr for free, unlimited for pro. Horizontal scaling: Stateless API servers behind load balancer, Redis for session persistence, separate worker pool for long-running tasks (codebase indexing).',
    keyPoints: [
      'Separate concerns: API layer, business logic, data layer, worker queue',
      'Real-time requires WebSocket + CRDT for conflict resolution',
      'Context management: chunk code semantically (functions/classes), embed, vector search',
      'Cost optimization: aggressive caching, model routing, rate limiting',
      'Scale: stateless services, Redis for shared state, worker pool for background jobs',
    ],
  },
  {
    id: 'bcap-pad-q-2',
    question:
      'Compare monolithic vs microservices architectures for an AI product with 3 features: chat, image generation, and document analysis. Each has different resource requirements (chat: low latency, image: GPU, docs: CPU-heavy). Which architecture would you choose and why? How would you handle: deployment, monitoring, cost allocation, and feature velocity?',
    sampleAnswer:
      'Choose microservices for this use case. Reasoning: (1) Resource isolation - image gen needs GPU instances, chat needs low-latency API servers, docs need CPU workers. Monolith would require expensive GPU instances for everything. (2) Independent scaling - scale each service based on demand (image gen scales differently than chat). (3) Cost allocation - track costs per service. (4) Feature velocity - teams work independently. Deployment: Docker + Kubernetes with separate deployments per service. Monitoring: Centralized logging (ELK), distributed tracing (Jaeger), metrics (Prometheus). Cost: Tag cloud resources by service, track LLM API usage per feature. Trade-offs: Increased complexity (network calls, service mesh), harder local development (Docker Compose), operational overhead. For early-stage startup, might start monolithic then extract services as needed.',
    keyPoints: [
      'Microservices win when features have vastly different resource needs',
      'GPU services should be isolated to avoid expensive compute for simple tasks',
      'Independent scaling critical for cost optimization',
      'Trade-off: operational complexity vs resource efficiency',
      'Start monolithic for speed, extract services as product matures',
    ],
  },
  {
    id: 'bcap-pad-q-3',
    question:
      "You're building an AI product that needs to support 100k concurrent users with <2s response time. Current architecture: single FastAPI server, OpenAI API calls, PostgreSQL database. System is slow and expensive. Design a scalable, cost-effective architecture. Include: caching layers, database optimization, LLM provider strategy, CDN usage, and monitoring. What\'s your migration path from current to target architecture?",
    sampleAnswer:
      'Target architecture: (1) Multi-layer caching: CDN (Cloudflare) for static assets, Redis for semantic cache (hash prompts, cache responses for 1hr), application-level cache for user data. (2) Database: Add read replicas for queries, connection pooling (PgBouncer), index optimization, move analytics to separate OLAP DB (ClickHouse). (3) LLM strategy: Multi-provider routing (OpenAI/Anthropic/Cohere), model cascade (try GPT-4o-mini first, fallback to GPT-4o if needed), prompt caching (Anthropic), batch requests where possible. (4) Infrastructure: Horizontal scaling with load balancer, separate worker pool for LLM calls (prevent blocking), rate limiting per user tier, auto-scaling based on queue depth. (5) Monitoring: Track p95 latency, cache hit rate, cost per request, error rate. Migration path: Phase 1 (week 1-2): Add Redis cache, semantic caching. Phase 2 (week 3-4): Database optimization, read replicas. Phase 3 (week 5-6): Multi-provider routing, horizontal scaling. Phase 4 (week 7-8): Worker pool, advanced caching. Expect 60% cost reduction, 3x latency improvement.',
    keyPoints: [
      'Multi-layer caching: CDN → Redis → Database',
      'Semantic cache for LLM responses (massive cost savings)',
      'Database: read replicas, connection pooling, separate OLAP',
      'Multi-provider strategy with model cascade for cost optimization',
      'Phased migration minimizes risk, validates improvements incrementally',
    ],
  },
];
