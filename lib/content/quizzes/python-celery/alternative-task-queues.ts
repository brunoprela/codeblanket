/**
 * Quiz questions for Alternative Task Queues section
 */

export const alternativeTaskQueuesQuiz = [
  {
    id: 'q1',
    question:
      'Compare Celery, RQ, Dramatiq, and Huey. When would you choose each?',
    sampleAnswer:
      'TASK QUEUE COMPARISON: **Celery**: Use for: Enterprise apps, complex workflows (chains/chords), multiple brokers, extensive monitoring. Pros: Feature-rich, Flower monitoring, large community. Cons: Complex setup, heavy. **RQ**: Use for: MVPs, prototypes, simple queues, quick setup. Pros: Very simple API, Redis-based, lightweight. Cons: Redis only, no periodic tasks, limited features. **Dramatiq**: Use for: Fast reliable processing, good error handling, medium projects. Pros: Fast, auto-retry, Redis + RabbitMQ. Cons: Smaller community, fewer integrations. **Huey**: Use for: Flask/Django, periodic tasks, small/medium projects. Pros: Lightweight, built-in periodic tasks, simple. Cons: Limited scale vs Celery. **DECISION**: Start RQ (MVP) → Huey (periodic tasks) → Celery (enterprise scale).',
    keyPoints: [
      'Celery: Enterprise, complex',
      'RQ: Simple, MVP',
      'Dramatiq: Fast, reliable',
      'Huey: Flask/Django, periodic',
      'Choose based on scale and requirements',
    ],
  },
  {
    id: 'q2',
    question:
      'Your startup uses RQ with Redis. As you scale to 1M tasks/day with complex workflows, should you migrate to Celery? Explain.',
    sampleAnswer:
      'RQ → CELERY MIGRATION DECISION: **Current**: RQ + Redis, 1M tasks/day, complex workflows needed. **Problems with RQ at scale**: (1) No canvas (chains/chords), (2) Limited monitoring, (3) No RabbitMQ support, (4) Manual periodic task management. **Benefits of Celery**: (1) Canvas workflows (chains, groups, chords), (2) Flower monitoring, (3) RabbitMQ for reliability, (4) Celery Beat for periodic tasks, (5) Better at 1M+ tasks/day. **RECOMMENDATION**: Migrate to Celery. **Migration plan**: Phase 1: Install Celery + RabbitMQ, Phase 2: Run parallel (RQ + Celery), Phase 3: Migrate non-critical tasks, Phase 4: Migrate critical tasks, Phase 5: Deprecate RQ. **Timeline**: 2-3 weeks. **Result**: Better reliability, complex workflows, proper monitoring.',
    keyPoints: [
      'RQ: Good for simple < 100K tasks/day',
      'Celery: Better for 1M+ tasks/day + complex workflows',
      'Migration: Gradual rollout',
      'Benefits: Canvas, monitoring, reliability',
      'Decision: Migrate at scale',
    ],
  },
  {
    id: 'q3',
    question:
      'Implement the same task in Celery, RQ, and Dramatiq. Compare the code.',
    sampleAnswer:
      'TASK QUEUE CODE COMPARISON: **Celery**: ```python from celery import Celery app = Celery("myapp", broker="redis://localhost:6379/0") @app.task (bind=True, max_retries=3) def send_email (self, email): try: send_mail (email) except Exception as exc: raise self.retry (exc=exc, countdown=60) send_email.delay("user@example.com") ``` **RQ**: ```python from redis import Redis from rq import Queue q = Queue (connection=Redis()) def send_email (email): send_mail (email) job = q.enqueue (send_email, "user@example.com") ``` **Dramatiq**: ```python import dramatiq @dramatiq.actor (max_retries=3) def send_email (email): send_mail (email) send_email.send("user@example.com") ``` **COMPARISON**: Code lines: RQ (5) < Dramatiq (6) < Celery (10). Retry handling: Celery (manual), Dramatiq (auto), RQ (limited). Verdict: RQ simplest, Celery most powerful.',
    keyPoints: [
      'RQ: Simplest code',
      'Dramatiq: Auto-retry',
      'Celery: Most features',
      'All solve same problem',
      'Choose based on needs',
    ],
  },
];
