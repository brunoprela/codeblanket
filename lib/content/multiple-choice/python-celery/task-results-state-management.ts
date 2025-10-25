/**
 * Multiple choice questions for Task Results & State Management section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const taskResultsStateManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What does result.get(timeout=10) do?',
      options: [
        'Checks if task is complete without waiting',
        'Waits up to 10 seconds for task to complete, then returns result (blocks)',
        'Sets task timeout to 10 seconds',
        'Gets task result from cache',
      ],
      correctAnswer: 1,
      explanation:
        'result.get(timeout=10) BLOCKS the calling thread for up to 10 seconds waiting for task to complete. Returns result if success, raises exception if failure, raises TimeoutError if exceeds 10s. NEVER use in API endpoints (blocks web worker). Use result.ready() for non-blocking check. Alternative: Poll status endpoint asynchronously.',
    },
    {
      id: 'mc2',
      question: 'What is the purpose of result_expires configuration?',
      options: [
        'How long task can run before timing out',
        'How long result stays in backend before deletion (prevents unbounded growth)',
        'How long worker waits for task',
        'How long before task is retried',
      ],
      correctAnswer: 1,
      explanation:
        'result_expires controls how long results stay in result backend (Redis/database) before automatic deletion. Default: no expiration (unbounded growth!). Production: Set result_expires=3600 (1 hour) to prevent Redis from filling up. Without expiration, millions of results accumulate → Redis OOM. Example: app.conf.result_expires = 3600.',
    },
    {
      id: 'mc3',
      question: 'When should you use ignore_result=True?',
      options: [
        'Never, always store results',
        "For fire-and-forget tasks that don't need result retrieval (90% of tasks)",
        'Only for failed tasks',
        'For all periodic tasks',
      ],
      correctAnswer: 1,
      explanation:
        "Use ignore_result=True for fire-and-forget tasks: logging, emails, cleanup, analytics. These tasks don't need result retrieval → no reason to store in backend → saves memory. Example: @app.task(ignore_result=True) def send_email(). ~90% of tasks don't need results! Only store results for tasks where you need to retrieve output (reports, processing results). Reduces result backend load significantly.",
    },
    {
      id: 'mc4',
      question: 'What task state comes after STARTED?',
      options: [
        'Always SUCCESS',
        'SUCCESS, FAILURE, or RETRY (depending on execution)',
        'COMPLETE',
        'FINISHED',
      ],
      correctAnswer: 1,
      explanation:
        "After STARTED: SUCCESS (if succeeds), FAILURE (if fails), or RETRY (if retrying). State flow: PENDING → STARTED → SUCCESS/FAILURE/RETRY. If RETRY, goes back to STARTED for retry attempt. Custom states also possible (PROGRESS, DOWNLOADING, etc.) via self.update_state(). States COMPLETE and FINISHED don't exist in Celery.",
    },
    {
      id: 'mc5',
      question: 'How do you track progress in a long-running task?',
      options: [
        'Use print() statements',
        'Call self.update_state(state="PROGRESS", meta={...}) with progress info',
        'Write to a log file',
        'Progress tracking not supported',
      ],
      correctAnswer: 1,
      explanation:
        'Use self.update_state() to track progress: self.update_state(state="PROGRESS", meta={"current": 50, "total": 100, "percent": 50}). Frontend polls result.state and result.info to display progress bar. Example: @app.task(bind=True) def long_task(self): for i in range(100): self.update_state(state="PROGRESS", meta={"percent": i}). Enables real-time progress tracking.',
    },
  ];
