import { MultipleChoiceQuestion } from '@/lib/types';

export const backgroundTasksMultipleChoice = [
  {
    id: 1,
    question:
      "When should you use FastAPI's built-in BackgroundTasks instead of Celery?",
    options: [
      "For quick, simple operations (< 10 seconds) that don't require retry logic or persistence, like logging or sending simple notifications",
      'For all asynchronous operations to avoid the complexity of setting up Celery',
      'For CPU-intensive tasks like image processing that need to scale horizontally',
      'For critical operations like payment processing that require guaranteed delivery',
    ],
    correctAnswer: 0,
    explanation:
      "FastAPI BackgroundTasks is ideal for quick, fire-and-forget operations that complete in seconds and don't need retry logic. It runs in the same process as your API server, requires no external dependencies, and is simple to use. However, it has limitations: no persistence (tasks lost if server crashes), no retry logic, no monitoring, and can't scale across multiple servers. Use Celery when you need: retry logic (payment processing), long-running tasks (report generation), persistence (can't lose tasks), horizontal scaling (multiple workers), or task monitoring. BackgroundTasks is perfect for: logging, simple analytics, non-critical notifications, and quick post-request cleanup.",
  },
  {
    id: 2,
    question:
      'Why is exponential backoff with jitter recommended for task retries instead of fixed intervals?',
    options: [
      'Exponential backoff gives services time to recover while jitter prevents thundering herd problem where all tasks retry simultaneously',
      'Exponential backoff is faster than linear backoff for task completion',
      'Jitter makes tasks complete in random order which improves throughput',
      'Fixed intervals consume more memory than exponential backoff',
    ],
    correctAnswer: 0,
    explanation:
      "Exponential backoff (10s, 20s, 40s, 80s) gives failing services progressively more time to recover, preventing you from overwhelming an already struggling service. Jitter (randomness added to backoff) solves the thundering herd problem: if 1000 tasks fail at the same time and all retry at exactly the same interval, they'll overwhelm the service again. With jitter, retries are spread out: some retry at 10s, others at 12s, 15s, etc. The pattern: delay = min(base * (2 ** retries), max_delay) + random(0, delay * 0.3). This provides: 1) Progressive backoff (more time for recovery), 2) Prevents synchronized retries (jitter), 3) Bounded delays (max_delay cap), 4) Production-proven (used by AWS SDKs, Stripe, etc). Fixed intervals fail when many tasks retry simultaneously, overwhelming the service repeatedly.",
  },
  {
    id: 3,
    question:
      'What is the purpose of idempotency keys in payment processing tasks?',
    options: [
      'To ensure that if a task is retried (due to failure or timeout), the same payment is not charged multiple times',
      'To encrypt payment data for security compliance',
      'To track which user initiated the payment transaction',
      'To speed up payment processing by caching results',
    ],
    correctAnswer: 0,
    explanation:
      'Idempotency keys prevent duplicate charges when tasks are retried. Scenario: You submit payment task, it charges the card successfully, but network fails before confirmation reaches your server. Without idempotency, retry would charge card again. With idempotency key (e.g., "order_12345"), payment gateway recognizes this is the same request and returns the original result instead of charging again. Implementation: stripe.Charge.create(amount=1000, source=token, idempotency_key=f"order_{order_id}"). The key is: 1) Deterministic (same for same order), 2) Unique per operation, 3) Included in every API call. Payment gateways cache results by idempotency key for 24 hours. This makes payment tasks safe to retry: same result whether called once or 100 times. Critical for production: network failures, timeouts, and worker restarts are common—idempotency ensures users aren\'t charged multiple times.',
  },
  {
    id: 4,
    question:
      'You have a Celery deployment where tasks are being added to the queue but not processed. What is the most likely cause?',
    options: [
      'No Celery workers are running, or workers are not configured to consume from the correct queue',
      'Redis is running out of disk space',
      'Tasks are too large and exceed the message size limit',
      'FastAPI is blocking the task queue',
    ],
    correctAnswer: 0,
    explanation:
      'The most common cause of tasks stuck in queue is no workers running or workers consuming from wrong queue. Diagnostic steps: 1) Check workers running: celery -A tasks inspect stats (returns empty if no workers), 2) Check worker queues: celery -A tasks inspect active_queues (verify workers consuming from correct queue), 3) Check queue length: redis-cli LLEN celery (shows tasks waiting). Common scenarios: workers crashed and weren\'t restarted (use supervisor/systemd for auto-restart), workers configured for queue "high_priority" but tasks sent to "default", workers exist but all stuck on long-running tasks (increase concurrency). Redis disk space (option 2) would cause write failures, not stuck tasks. Message size limits (option 3) would cause immediate errors, not queuing. FastAPI (option 4) doesn\'t interact with the queue—it only submits tasks. Fix: Start workers with correct queue configuration: celery -A tasks worker --queues=default,high_priority --concurrency=4',
  },
  {
    id: 5,
    question:
      'What is the purpose of a dead letter queue in Celery task processing?',
    options: [
      'To collect and handle tasks that have permanently failed after all retry attempts, enabling manual investigation and alerting',
      'To temporarily store tasks during worker restarts',
      'To prioritize urgent tasks over regular tasks',
      'To cache task results for faster retrieval',
    ],
    correctAnswer: 0,
    explanation:
      "A dead letter queue (DLQ) collects tasks that have permanently failed after exhausting all retries. Purpose: Some tasks fail permanently (invalid data, external service down for hours, bugs). After max retries (e.g., 5 attempts), these tasks should: 1) Not block the queue, 2) Not be silently dropped, 3) Be investigated by humans. DLQ implementation: When task fails permanently, send to separate queue/database for manual review. Example: Payment fails 5 times → move to DLQ → alert finance team → create support ticket → human investigates root cause. DLQ enables: 1) Automatic alerting (finance team notified), 2) Audit trail (all failures recorded), 3) Manual recovery (fix issue, reprocess task), 4) Root cause analysis (why did it fail?). Without DLQ, failed tasks disappear—lost revenue, angry customers, no visibility. Production pattern: on_failure() hook in Celery task sends to DLQ, which triggers alerts and creates tickets. This is different from temporary storage during restarts (option 2), task prioritization (option 3), or result caching (option 4)—it's specifically for handling permanent failures gracefully.",
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
