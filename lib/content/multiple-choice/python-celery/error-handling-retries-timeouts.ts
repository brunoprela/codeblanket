/**
 * Multiple choice questions for Error Handling, Retries & Timeouts section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const errorHandlingRetriesTimeoutsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'What is the difference between soft_time_limit and time_limit?',
      options: [
        'They are the same',
        'soft_time_limit raises exception (graceful), time_limit sends SIGKILL (forced)',
        'soft_time_limit is in seconds, time_limit is in minutes',
        'soft_time_limit for fast tasks, time_limit for slow tasks',
      ],
      correctAnswer: 1,
      explanation:
        'soft_time_limit raises SoftTimeLimitExceeded exception (task can catch, cleanup, save progress). time_limit sends SIGKILL (immediate termination, no cleanup). Best practice: Set soft < hard (e.g., soft=270s, hard=300s) for 30s grace period.',
    },
    {
      id: 'mc2',
      question: 'How do you implement exponential backoff for retries?',
      options: [
        'Use countdown=60 * (2 ** self.request.retries)',
        'Use countdown=60',
        'Use retry_backoff=True',
        'Both A and C',
      ],
      correctAnswer: 3,
      explanation:
        'Exponential backoff: countdown=60 * (2 ** retries) gives 60s, 120s, 240s, 480s. Or use autoretry_for with retry_backoff=True for automatic exponential backoff. Both approaches work. Manual gives more control, auto is simpler.',
    },
    {
      id: 'mc3',
      question: 'When should a task NOT retry?',
      options: [
        'Network timeout',
        'Rate limiting (429)',
        'Resource not found (404 - permanent error)',
        'Temporary server error (503)',
      ],
      correctAnswer: 2,
      explanation:
        "404 Not Found is permanent error - resource doesn't exist, retrying won't help. Don't retry permanent errors. DO retry: timeouts (transient), rate limits (temporary), 503 errors (temporary outage). Check error type before retrying.",
    },
    {
      id: 'mc4',
      question: 'What does autoretry_for=(Exception,) do?',
      options: [
        'Prevents all retries',
        'Automatically retries task if Exception (or subclass) raised',
        'Retries only on success',
        'Disables error handling',
      ],
      correctAnswer: 1,
      explanation:
        'autoretry_for=(Exception,) automatically retries task if Exception (or any subclass) is raised. No need for manual self.retry(). Combine with retry_kwargs, retry_backoff for comprehensive auto-retry. Example: @app.task(autoretry_for=(RequestException,), retry_kwargs={"max_retries": 5}).',
    },
    {
      id: 'mc5',
      question: 'What is a dead letter queue (DLQ)?',
      options: [
        'Queue for urgent tasks',
        'Storage for permanently failed tasks after max retries',
        'Queue for deleted tasks',
        'Priority queue',
      ],
      correctAnswer: 1,
      explanation:
        'Dead Letter Queue stores tasks that failed after all retry attempts (max_retries exceeded). Allows manual inspection, debugging, and potential reprocessing. Implementation: After max retries, send task info to separate queue/database. Prevents silent failures.',
    },
  ];
