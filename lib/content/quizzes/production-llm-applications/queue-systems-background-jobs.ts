export const queueSystemsBackgroundJobsQuiz = [
  {
    id: 'pllm-q-4-1',
    question:
      'Design a comprehensive queue-based architecture for an LLM application that processes documents, generates summaries, and sends notifications. Include job priorities, retry logic, dead letter queues, and monitoring. How would you ensure reliability and handle failures?',
    sampleAnswer:
      'Architecture: RabbitMQ broker with 3 queues (urgent, default, batch), Celery workers pulling from queues, Redis for results, PostgreSQL for job metadata. Job flow: 1) extract_text.delay (doc_id) → default queue, 2) on success → summarize_text.delay (text) → priority based on user tier, 3) on success → send_notification.delay (user_id), 4) on any failure → move to dead letter queue after 3 retries. Priorities: urgent queue (SLA <1min) for premium users, default queue (<5min) for regular, batch queue (best effort) for background jobs. Retry logic: @task (max_retries=3, retry_backoff=True, retry_backoff_max=600, retry_jitter=True) for transient errors, immediate failure for invalid requests. Dead letter queue: after max retries, store in DLQ with error details, alert ops team, provide admin interface to retry manually. Monitoring: track queue depth per queue, worker utilization, success/failure rates, average processing time, alert on queue depth >100 or error rate >5%. Reliability: task_acks_late=True (acknowledge after completion), task_reject_on_worker_lost=True, persistent message delivery, worker health checks, automatic worker restart on failure.',
    keyPoints: [
      'Multiple priority queues with appropriate routing and SLAs',
      'Comprehensive retry logic with exponential backoff and DLQ',
      'Monitoring and alerting for queue health and worker performance',
    ],
  },
  {
    id: 'pllm-q-4-2',
    question:
      'Compare Celery and Redis Queue (RQ) for LLM background jobs. When would you choose each, and how would you migrate from RQ to Celery as your application scales? What challenges might you encounter?',
    sampleAnswer:
      "RQ: Simpler, minimal configuration, good for <10K jobs/day, no complex workflows, Python-only. Setup in minutes, easy debugging. Limitations: no task routing, basic scheduling, single Redis instance, no advanced patterns. Celery: Enterprise-ready, complex workflows (chains, groups, chords), multiple brokers (RabbitMQ, Redis, SQS), advanced routing, better monitoring, handles millions of jobs/day. Complexity: more configuration, steeper learning curve. Choose RQ for: MVP/prototype, simple background jobs, small team, <10K jobs/day, don't need advanced features. Choose Celery for: production scale, complex workflows, need reliability guarantees, multiple job types, >100K jobs/day. Migration strategy: 1) Add Celery alongside RQ (dual-write), 2) Migrate one job type at a time, 3) Test thoroughly in staging, 4) Monitor both systems, 5) Gradually route traffic to Celery, 6) Keep RQ as fallback for 2 weeks, 7) Deprecate RQ. Challenges: different task signatures, workflow patterns need rewriting, monitoring setup, worker configuration, result backend migration, handling in-flight jobs during switch. Use feature flags to control routing percentage.",
    keyPoints: [
      'RQ for simplicity and small scale, Celery for production and complex workflows',
      'Gradual migration with dual-write and percentage-based routing',
      'Challenges include API differences and in-flight job handling',
    ],
  },
  {
    id: 'pllm-q-4-3',
    question:
      'Explain how you would implement progress tracking for long-running LLM tasks (processing 100-page documents) that provides real-time updates to users. Include technical implementation, user experience considerations, and error handling.',
    sampleAnswer:
      'Implementation: Use Celery with bind=True, call self.update_state (state="PROCESSING", meta={progress: 45, current_page: 45, total_pages: 100, status: "Processing page 45..."}) after each page. Store in Redis with TTL. Frontend polls /tasks/{task_id}/progress every 2s or uses WebSocket for real-time updates. Technical details: task updates include percentage, current step, ETA (calculate from average page time), detailed status message. Store granular progress: {stage: "extraction|summary|save", progress_percent: 45, current_item: 45, total_items: 100, eta_seconds: 120, errors: []}. UX considerations: show progress bar with percentage, current status message, estimated time remaining, allow cancellation, show which page is being processed, handle progress going backwards (retries) gracefully, don\'t update too frequently (< 1s), use optimistic updates. Error handling: on error, preserve progress state, show partial results if possible, offer resume option, log detailed error with context, provide helpful error message, track error frequency per task type. WebSocket message format: {type: "progress", task_id: "123", progress: 45, message: "Page 45/100"}. Handle disconnections: store last received progress, resume from there on reconnect.',
    keyPoints: [
      'Granular state updates with meta information stored in Redis',
      'Real-time updates via polling or WebSocket with appropriate frequency',
      'Comprehensive error handling with resume capability and partial results',
    ],
  },
];
