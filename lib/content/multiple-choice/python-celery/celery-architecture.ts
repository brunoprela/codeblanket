/**
 * Multiple choice questions for Celery Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const celeryArchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which worker pool should you use for CPU-intensive image processing tasks that need true parallelism?',
    options: [
      'threads - because they share memory efficiently',
      'prefork - because it uses multiprocessing and bypasses the GIL',
      'gevent - because it supports massive concurrency',
      "solo - because it's simplest",
    ],
    correctAnswer: 1,
    explanation:
      "For CPU-intensive tasks, use prefork (multiprocessing) because it creates separate processes that bypass Python\'s GIL (Global Interpreter Lock), enabling true parallelism. Threads (option 1) are limited by GIL for CPU tasks. Gevent (option 3) is for I/O tasks only. Solo (option 4) has no concurrency. CPU-bound = prefork, I/O-bound = gevent/threads.",
  },
  {
    id: 'mc2',
    question:
      'Your application makes 1000s of HTTP API calls per minute. Which worker pool and concurrency setting is best?',
    options: [
      'prefork with concurrency=4',
      'threads with concurrency=50',
      'gevent with concurrency=1000',
      'solo with concurrency=1',
    ],
    correctAnswer: 2,
    explanation:
      'HTTP API calls are I/O-bound (waiting for network responses). Gevent with high concurrency (1000) is perfect: lightweight greenlets, low memory, can handle thousands of concurrent I/O operations. Prefork (option 1) wastes memory with separate processes. Threads (option 2) better than prefork but still limited by GIL. Solo (option 4) has no concurrency. I/O-bound = gevent with high concurrency.',
  },
  {
    id: 'mc3',
    question: 'What is the role of the result backend in Celery?',
    options: [
      'It executes tasks in background',
      'It stores task results and states for later retrieval',
      'It routes tasks to workers',
      'It replaces the message broker',
    ],
    correctAnswer: 1,
    explanation:
      'The result backend stores task results and states (PENDING, SUCCESS, FAILURE) so clients can retrieve them later via result.get(). Common backends: Redis, database, RPC. The broker (not backend) routes tasks. Workers (not backend) execute tasks. Backend is optional - use it if you need to retrieve results, skip it for fire-and-forget tasks.',
  },
  {
    id: 'mc4',
    question:
      'How do you route specific tasks to specific queues for specialized workers?',
    options: [
      "app.conf.task_routes = {'tasks.process_video': {'queue': 'videos'}}",
      'Use different brokers for each task',
      'Rename the task to include queue name',
      'Workers automatically detect task types',
    ],
    correctAnswer: 0,
    explanation:
      'Use task_routes configuration to map tasks to queues: app.conf.task_routes = {"tasks.process_video": {"queue": "videos"}}. Then start specialized workers: celery -A tasks worker -Q videos --concurrency=2. This enables task isolation (slow tasks don\'t block fast ones), specialized hardware (GPU workers for videos), and independent scaling. Options 2-4 are incorrect approaches.',
  },
  {
    id: 'mc5',
    question:
      'Your Celery workers crash with "MemoryError" after processing 500 tasks. What\'s the likely cause and fix?',
    options: [
      'Broker is full - switch from Redis to RabbitMQ',
      'Memory leak - set worker_max_tasks_per_child=100 to restart workers periodically',
      'Too many workers - reduce concurrency',
      'Tasks are too fast - slow them down with time.sleep()',
    ],
    correctAnswer: 1,
    explanation:
      "Memory accumulates over time even with careful coding (Python GC isn't perfect). Setting worker_max_tasks_per_child=100 restarts workers after 100 tasks, preventing memory accumulation. This is standard production practice. Option 1 (broker) doesn't affect worker memory. Option 3 (reduce concurrency) just processes slower, doesn't fix leak. Option 4 is nonsense. Other fixes: stream files (don't load entirely), set memory limits, use prefork for process isolation.",
  },
];
