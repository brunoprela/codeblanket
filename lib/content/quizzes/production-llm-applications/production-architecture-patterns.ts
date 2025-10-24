export const productionArchitecturePatternsQuiz = [
  {
    id: 'pllm-q-1-1',
    question:
      'Compare microservices, queue-based, and event-driven architectures for LLM applications. When would you choose each pattern, and how would you implement a hybrid architecture that combines all three? Consider scalability, cost, complexity, and failure handling in your answer.',
    sampleAnswer:
      'Microservices provide service isolation and independent scaling but add network overhead. Queue-based architectures excel at handling long-running LLM tasks asynchronously but require additional infrastructure. Event-driven patterns offer loose coupling and extensibility but can be harder to debug. A hybrid approach is often best: use microservices for service boundaries (API gateway, LLM service, cache service), queues for expensive operations (document processing, batch generation), and events for workflow coordination (document uploaded → extract text → summarize → notify user). This provides flexibility while managing complexity. Implement with FastAPI for APIs, Celery for queues, and RabbitMQ for events. Scale microservices horizontally, add queue workers based on depth, and use circuit breakers for failure handling.',
    keyPoints: [
      'Microservices for service boundaries and independent scaling',
      'Queues for async processing of expensive LLM operations',
      'Events for workflow coordination and loose coupling',
    ],
  },
  {
    id: 'pllm-q-1-2',
    question:
      'Design a production architecture for an LLM application that handles 10,000 concurrent users, processes 1 million API calls per day, and maintains 99.9% uptime. Include specific technologies, scaling strategies, and failure recovery mechanisms.',
    sampleAnswer:
      'Architecture: Load balancer (AWS ALB) → API Gateway (10+ instances, auto-scaled) → LLM Service (20+ instances) + Cache (Redis Cluster) + Queue (Celery with RabbitMQ) + Database (PostgreSQL with read replicas) + Vector DB (Pinecone). Scaling: Horizontal scaling for stateless services with auto-scaling based on CPU/memory. Queue workers scale based on queue depth. Database uses read replicas for read-heavy operations. Failure recovery: Health checks every 30s, automatic instance replacement, circuit breakers for external dependencies, request retries with exponential backoff, cached fallbacks when APIs fail. Multi-AZ deployment for redundancy. Rate limiting per user to prevent abuse. Monitoring with Prometheus/Grafana, logging with ELK stack, distributed tracing with Jaeger. Cost optimization through caching (50-90% hit rate goal), cheaper model routing when appropriate, and batch processing during off-peak hours.',
    keyPoints: [
      'Horizontal scaling with auto-scaling groups',
      'Multi-layer caching and redundancy for reliability',
      'Comprehensive monitoring and failure recovery mechanisms',
    ],
  },
  {
    id: 'pllm-q-1-3',
    question:
      'Explain the trade-offs between stateless and stateful service design for LLM applications. How does conversation history affect your architecture decisions, and what strategies can maintain performance while preserving state?',
    sampleAnswer:
      'Stateless services are simpler to scale and more reliable (any instance handles any request), but require external storage for conversation history. Stateful services can optimize by keeping conversation in memory but complicate scaling and failure recovery. For LLM apps with conversation history: Store all state in Redis or PostgreSQL, use session IDs to retrieve context, keep services stateless for easy scaling. Strategies to maintain performance: Cache conversation history in Redis with TTL (1-24 hours), use connection pooling for database access, compress large conversation histories, implement sliding window (keep last N messages), use CDN for static content. For high-traffic conversations: Partition by session ID (sticky sessions), use Redis Cluster for horizontal scaling of cache, implement read replicas for database. The key is external state storage (Redis/DB) with aggressive caching and efficient serialization, allowing stateless services to scale while maintaining conversation context.',
    keyPoints: [
      'Stateless services with external state storage for scalability',
      'Redis caching for conversation history performance',
      'Session management strategies for maintaining context',
    ],
  },
];
