export const goToMarketStrategyQuiz = [
  {
    id: 'bcap-go-to-market-strategy-q-1',
    question:
      'Design a complete architecture for go-to-market strategy. Consider scalability, cost, security, and user experience. What are the key components and how do they interact?',
    sampleAnswer:
      'A production go-to-market strategy requires multiple layers: Frontend (React/Next.js with WebSocket for real-time), API Gateway (FastAPI with rate limiting), Processing Layer (async workers with Celery/Redis queue), Database Layer (PostgreSQL with read replicas, Redis cache), AI Service Layer (OpenAI/Claude with fallbacks), Storage (S3 for files), and Monitoring (DataDog/Sentry). Key interactions: Frontend → API Gateway → Queue → Workers → AI Services. Scale horizontally with load balancers, cache aggressively, use CDN for static assets. Security: JWT auth, input validation, rate limiting, encryption at rest/transit. Cost optimization: smart model selection, caching, batching requests. Total architecture supports 100K+ concurrent users with 99.9% uptime.',
    keyPoints: [
      'Multi-layer architecture with clear separation of concerns',
      'Horizontal scaling with load balancing and caching',
      'Comprehensive security measures',
    ],
  },
  {
    id: 'bcap-go-to-market-strategy-q-2',
    question:
      'What are the key performance bottlenecks in go-to-market strategy and how would you address them? Include specific metrics and optimization strategies.',
    sampleAnswer:
      'Main bottlenecks: 1) LLM API latency (2-10s) - mitigate with streaming responses, response caching, and smart model routing (GPT-3.5 for simple tasks); 2) Database queries - add indexes, use read replicas, implement query caching with Redis; 3) File processing - async workers, parallel processing, pre-processing pipelines; 4) Frontend rendering - code splitting, lazy loading, virtual scrolling for large lists. Metrics to track: p95 response time <3s, throughput >1000 req/s, error rate <0.1%, cache hit rate >70%. Load testing reveals bottlenecks early. Use APM tools (DataDog) to identify slow queries/endpoints. Optimize hot paths first (80/20 rule). Monitor continuously and iterate.',
    keyPoints: [
      'Identify and measure key bottlenecks',
      'Multiple mitigation strategies per bottleneck',
      'Continuous monitoring and iteration',
    ],
  },
  {
    id: 'bcap-go-to-market-strategy-q-3',
    question:
      'How would you handle failure scenarios in go-to-market strategy? Design a comprehensive error handling and recovery strategy including retries, fallbacks, and user communication.',
    sampleAnswer:
      'Multi-layer error handling: 1) Network failures - exponential backoff retries (3 attempts, 2^n seconds), timeout after 30s; 2) API errors - fallback to alternative models (GPT-4 fails → Claude, Claude fails → GPT-3.5), circuit breaker pattern to prevent cascade failures; 3) Database errors - connection pooling with retries, read from replicas on primary failure; 4) Processing errors - dead letter queue for failed jobs, manual review dashboard. User communication: clear error messages (not "500 error", but "AI service temporarily unavailable, retrying..."), show progress for long operations, offer cancel option. Logging: structured logs with correlation IDs, error tracking (Sentry), alerting on error rate >1%. Recovery: automatic retries for transient errors, graceful degradation (disable feature vs entire system down), regular backup testing, documented runbooks for common issues.',
    keyPoints: [
      'Multiple layers of error handling with fallbacks',
      'Clear user communication during failures',
      'Comprehensive logging and alerting strategy',
    ],
  },
];
