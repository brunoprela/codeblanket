import { MultipleChoiceQuestion } from '../../../types';

export const queueSystemsBackgroundJobsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pllm-queue-mc-1',
      question: 'What is the purpose of a dead letter queue (DLQ)?',
      options: [
        'Store completed jobs',
        'Store jobs that failed after all retries for manual review',
        'Priority queue for urgent tasks',
        'Archive old jobs',
      ],
      correctAnswer: 1,
      explanation:
        'Dead letter queues store jobs that failed after all retry attempts. This allows manual investigation, fixes, and retry without losing information about failures.',
    },
    {
      id: 'pllm-queue-mc-2',
      question: 'When using Celery, what does task_acks_late=True do?',
      options: [
        'Delays task execution',
        'Acknowledges task only after completion (prevents loss on worker crash)',
        'Reduces performance',
        'Increases priority',
      ],
      correctAnswer: 1,
      explanation:
        'task_acks_late=True means tasks are only acknowledged after completion. If a worker crashes during execution, the task goes back to the queue instead of being lost.',
    },
    {
      id: 'pllm-queue-mc-3',
      question: 'How should you implement job priorities in a queue system?',
      options: [
        'Single queue with priority field',
        'Multiple queues (urgent, default, batch) with workers pulling in order',
        'Random selection',
        'FIFO only',
      ],
      correctAnswer: 1,
      explanation:
        'Use multiple queues with different priorities. Workers pull from urgent first, then default, then batch. This ensures high-priority jobs are processed first.',
    },
    {
      id: 'pllm-queue-mc-4',
      question: 'What is the purpose of exponential backoff in retry logic?',
      options: [
        'Make retries faster',
        'Gradually increase wait time between retries to avoid overwhelming failing services',
        'Reduce costs',
        'Improve accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'Exponential backoff (2^n seconds) gradually increases wait time between retries, giving failing services time to recover and avoiding thundering herd problems.',
    },
    {
      id: 'pllm-queue-mc-5',
      question: 'How do you track progress for long-running tasks in Celery?',
      options: [
        'You cant',
        'Use self.update_state() to publish progress updates',
        'Check logs',
        'Poll the database',
      ],
      correctAnswer: 1,
      explanation:
        'In Celery tasks with bind=True, use self.update_state(state=PROCESSING, meta={progress: 50}) to publish progress that clients can poll.',
    },
  ];
