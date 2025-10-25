/**
 * Quiz questions for Redis vs RabbitMQ section
 */

export const redisVsRabbitmqQuiz = [
  {
    id: 'q1',
    question:
      'Your startup is processing 50K tasks/day with Redis. As you scale to 5M tasks/day, you experience task loss during Redis restarts. Should you migrate to RabbitMQ? Provide a comprehensive analysis.',
    sampleAnswer:
      'REDIS TO RABBITMQ MIGRATION ANALYSIS: **Current State**: 50K → 5M tasks/day (100× growth). Task loss during Redis restarts. **Problem**: Redis in-memory = tasks lost on restart (unless AOF/RDB configured). At 5M tasks/day = critical reliability issue. **Solution: Migrate to RabbitMQ**. **Why**: (1) Disk-backed persistence (no task loss), (2) Handles 5M tasks/day easily, (3) Better reliability for scale. **Migration Plan**: Phase 1: Install RabbitMQ, test, Phase 2: Dual-run (Redis + RabbitMQ), Phase 3: Migrate non-critical tasks, Phase 4: Migrate critical tasks, Phase 5: Deprecate Redis. **Outcome**: Zero task loss, better reliability at scale.',
    keyPoints: [
      'Redis: In-memory = task loss risk',
      'RabbitMQ: Disk-backed = persistent',
      'Migration: Gradual rollout',
      'Scale: 5M tasks/day needs reliability',
      'Decision: Migrate to RabbitMQ',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Redis and RabbitMQ performance for 100K tasks/sec workload.',
    sampleAnswer:
      'PERFORMANCE COMPARISON AT 100K TASKS/SEC: **Redis**: Throughput: ~100K tasks/sec. Latency: <1ms. Memory: 10GB RAM for queue. Result: ✅ Handles load BUT risk of task loss. **RabbitMQ**: Throughput: ~50K tasks/sec. Latency: ~10ms. Disk: 100GB SSD for persistence. Result: ⚠️ Need 2 instances for 100K/sec. **Conclusion**: Redis faster BUT RabbitMQ more reliable. At 100K/sec scale, reliability > speed. Use RabbitMQ with horizontal scaling.',
    keyPoints: [
      'Redis: 2× faster but less reliable',
      'RabbitMQ: Reliable but need scaling',
      'At 100K/sec: Reliability critical',
      'Solution: RabbitMQ + horizontal scaling',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a hybrid Redis + RabbitMQ architecture where each is used appropriately.',
    sampleAnswer:
      'HYBRID ARCHITECTURE: **Redis (Non-Critical)**: Email tasks, Analytics events, Logging tasks, Cache warming. **RabbitMQ (Critical)**: Payment processing, Order creation, User registration, Data backup. **Benefits**: Speed for non-critical (Redis), Reliability for critical (RabbitMQ), Cost optimization (Redis cheaper). **Implementation**: ```python app_fast = Celery("fast", broker="redis://localhost:6379/0") app_critical = Celery("critical", broker="amqp://localhost:5672//") @app_fast.task def send_email(): pass @app_critical.task def process_payment(): pass ``` **Result**: Best of both worlds.',
    keyPoints: [
      'Redis: Non-critical tasks',
      'RabbitMQ: Critical tasks',
      'Hybrid: Best of both',
      'Separate Celery apps',
      'Cost + reliability optimization',
    ],
  },
];
