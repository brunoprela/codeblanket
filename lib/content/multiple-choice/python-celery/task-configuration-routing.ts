/**
 * Multiple choice questions for Task Configuration & Routing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const taskConfigurationRoutingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is the purpose of task routing in Celery?',
      options: [
        'To make tasks run faster',
        'To direct specific tasks to specific queues for isolation and specialization',
        'To automatically scale workers',
        'To compress task payloads',
      ],
      correctAnswer: 1,
      explanation:
        'Task routing directs tasks to specific queues, enabling: (1) Isolation (slow tasks don\'t block fast tasks), (2) Specialization (GPU workers for video, CPU for reports), (3) Independent scaling per queue. Example: task_routes = {"tasks.send_email": {"queue": "emails"}}. Start workers for specific queues: celery -A tasks worker -Q emails. Option 1 (speed) is wrong - routing doesn\'t make tasks faster. Options 3-4 are unrelated.',
    },
    {
      id: 'mc2',
      question: 'What does worker_prefetch_multiplier=4 mean?',
      options: [
        'Worker runs 4 tasks simultaneously',
        'Worker fetches 4 tasks from queue at once and processes them sequentially',
        'Worker creates 4 child processes',
        'Worker can handle 4 connections to the broker',
      ],
      correctAnswer: 1,
      explanation:
        'prefetch_multiplier=4 means worker prefetches (locks) 4 tasks from the queue before processing them. Worker processes sequentially: Task 1 → Task 2 → Task 3 → Task 4, then fetches 4 more. Problem: Fast tasks stuck behind slow tasks. Solution: prefetch=1 (fetch one at a time) for better load balancing. Option 1 (simultaneous): Wrong - concurrency controls parallel execution. Option 3 (processes): Wrong - pool controls processes. Option 4 (connections): Wrong - unrelated.',
    },
    {
      id: 'mc3',
      question:
        'What is the difference between soft_time_limit and time_limit?',
      options: [
        'They are the same thing',
        'soft_time_limit raises an exception (graceful), time_limit sends SIGKILL (forced)',
        'soft_time_limit is for fast tasks, time_limit is for slow tasks',
        'soft_time_limit is in seconds, time_limit is in minutes',
      ],
      correctAnswer: 1,
      explanation:
        'soft_time_limit raises SoftTimeLimitExceeded exception (graceful - task can cleanup, save progress, catch exception). time_limit sends SIGKILL to worker process (forced termination - no cleanup, immediate kill). Set soft < hard: soft_time_limit=270s, time_limit=300s gives 30s for cleanup before SIGKILL. Production: Always set soft limit for graceful shutdown. Options 1, 3, 4 are incorrect.',
    },
    {
      id: 'mc4',
      question: 'Which serializer should you use in production for security?',
      options: [
        "pickle - it's the fastest",
        "json - it's safe and doesn't allow arbitrary code execution",
        "yaml - it's most readable",
        "msgpack - it's most compact",
      ],
      correctAnswer: 1,
      explanation:
        'Use JSON serialization in production for security. pickle allows arbitrary Python objects and can execute malicious code (security vulnerability with untrusted task sources). json is safe, widely supported, human-readable. msgpack is safe and faster than JSON but less common. yaml is slow and has security issues. Production config: task_serializer="json", accept_content=["json"]. Never use pickle with untrusted sources.',
    },
    {
      id: 'mc5',
      question: 'What does task_acks_late=True do?',
      options: [
        'Makes tasks run slower',
        'Acknowledges task after completion (not immediately), requeues if worker crashes',
        'Delays task execution by configured time',
        'Sends acknowledgment to the result backend instead of broker',
      ],
      correctAnswer: 1,
      explanation:
        'task_acks_late=True means worker acknowledges task AFTER completion (not immediately when received). If worker crashes mid-task, task is NOT acknowledged → broker requeues it → another worker processes it. Trade-off: Reliability (no lost tasks) vs risk of duplicates. Solution: Make tasks idempotent. Production: Use acks_late=True + task_reject_on_worker_lost=True for critical tasks. Options 1, 3, 4 are incorrect.',
    },
  ];
