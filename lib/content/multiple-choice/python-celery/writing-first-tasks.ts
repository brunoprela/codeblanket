/**
 * Multiple choice questions for Writing Your First Tasks section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const writingFirstTasksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'You need to process a 50MB CSV file in a Celery task. What is the best approach?',
    options: [
      'Pass the file contents (50MB bytes) as a task argument',
      'Pass the S3 key or file path, download file inside the worker',
      'Split file into 1MB chunks and pass each chunk as separate arguments',
      'Store file contents in Redis, pass Redis key',
    ],
    correctAnswer: 1,
    explanation:
      'Pass reference (S3 key, file path, URL), not data itself. Option 1 (pass 50MB) bloats queue, wastes memory, slow serialization. Option 2 (correct): Worker downloads when needed, minimal queue size. Option 3 (chunking) still passes 50MB total. Option 4 (Redis) wastes Redis memory. General rule: args >100KB → pass reference, not data.',
  },
  {
    id: 'mc2',
    question: 'What is the difference between .delay() and .apply_async()?',
    options: [
      '.delay() is faster and optimized for performance',
      '.delay() is syntactic sugar for .apply_async() with no options',
      '.delay() uses Redis while .apply_async() uses RabbitMQ',
      '.delay() is deprecated, always use .apply_async()',
    ],
    correctAnswer: 1,
    explanation:
      '.delay() is convenience wrapper for .apply_async() with no advanced options. They execute identically by default. Use .delay() for simple cases (90% of time): clean syntax. Use .apply_async() when you need: countdown/eta, priority, queue routing, custom task ID, retry policy. Option 1 (speed) is wrong - same speed. Option 3 (broker) is wrong - both use configured broker. Option 4 (deprecated) is wrong - both supported.',
  },
  {
    id: 'mc3',
    question:
      'Which types are safe to pass as task arguments with JSON serialization?',
    options: [
      'int, float, str, bool, list, dict, None',
      'datetime.datetime, decimal.Decimal, bytes',
      'Custom classes, file objects, database connections',
      'All Python objects are serializable',
    ],
    correctAnswer: 0,
    explanation:
      'JSON serialization supports: int, float, str, bool, list, dict, None. Option 2 (datetime, Decimal, bytes): NOT JSON serializable - convert to string/float first. Option 3 (custom objects): NOT serializable - pass IDs/references. Option 4 (all objects): Wrong - many types not serializable. For datetime: pass as .isoformat() string. For Decimal: pass as float. For objects: pass ID, fetch in worker.',
  },
  {
    id: 'mc4',
    question:
      'How do you make a task idempotent (safe to call multiple times)?',
    options: [
      'Add @idempotent decorator',
      'Check database flag before execution, set flag after success',
      'Use .delay() instead of .apply_async()',
      'Set max_retries=1 to prevent multiple executions',
    ],
    correctAnswer: 1,
    explanation:
      'Idempotence requires checking if task already executed: database flag (email_sent=True), separate log table with unique constraint, Redis cache. Option 1: No @idempotent decorator exists. Option 2 (correct): Check flag, execute only if not done, set flag atomically. Option 3: .delay() vs .apply_async() irrelevant to idempotence. Option 4: Retries are for failures, not idempotence. Best practice: database tracking + unique constraints for race conditions.',
  },
  {
    id: 'mc5',
    question:
      'What happens if you pass datetime.datetime.now() as a task argument with JSON serialization?',
    options: [
      'It works fine, datetime is automatically converted',
      'It raises "TypeError: Object of type datetime is not JSON serializable"',
      'It serializes but loses timezone information',
      'Celery converts it to Unix timestamp automatically',
    ],
    correctAnswer: 1,
    explanation:
      "datetime objects are NOT JSON serializable. JSON supports: int, float, str, bool, list, dict, None only. Passing datetime raises TypeError. Fix: Convert to string: datetime.now().isoformat(), then parse in worker: datetime.fromisoformat (date_str). Same for Decimal (pass as float), bytes (pass as base64 string), custom objects (pass ID). Celery doesn't auto-convert - you must handle serialization explicitly.",
  },
];
