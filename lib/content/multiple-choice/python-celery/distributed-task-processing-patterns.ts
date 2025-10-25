/**
 * Multiple choice questions for Distributed Task Processing Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const distributedTaskProcessingPatternsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is the map-reduce pattern?',
      options: [
        'Process sequentially then aggregate',
        'Process in parallel (map) then aggregate results (reduce)',
        'Process once without aggregation',
        'Only for database operations',
      ],
      correctAnswer: 1,
      explanation:
        'Map-reduce: (1) Map phase - process items in parallel across multiple workers, (2) Reduce phase - aggregate all results into final output. Example: Process 1M records in 100 chunks (map), sum all results (reduce). Enables massive parallel processing.',
    },
    {
      id: 'mc2',
      question: 'Why use distributed locking?',
      options: [
        'To make tasks faster',
        'To ensure only one worker processes a resource at a time (prevents race conditions)',
        'To store task results',
        'To route tasks to queues',
      ],
      correctAnswer: 1,
      explanation:
        'Distributed locking ensures only ONE worker processes a resource at a time, preventing race conditions. Example: Only one worker processes payment for order_id=123. Use Redis lock with timeout. Without locking: Multiple workers might process same payment (duplicate charges!).',
    },
    {
      id: 'mc3',
      question: 'What does idempotent mean?',
      options: [
        'Task runs fast',
        'Task runs exactly once',
        'Task can be called multiple times with same result',
        'Task never fails',
      ],
      correctAnswer: 2,
      explanation:
        'Idempotent: Calling task N times has same result as calling once. Example: send_email() checks if already sent (idempotent) vs blindly sends (not idempotent = duplicates!). Critical for retries. Implementation: Check if already processed before executing.',
    },
    {
      id: 'mc4',
      question: 'What is fan-out/fan-in pattern?',
      options: [
        'Process tasks sequentially',
        'Distribute work to multiple workers (fan-out), combine results (fan-in)',
        'Cancel tasks',
        'Retry failed tasks',
      ],
      correctAnswer: 1,
      explanation:
        'Fan-out/fan-in: (1) Fan-out - distribute work to N workers (parallel), (2) Fan-in - combine all results into one. Celery chord implements this: chord (header)(callback). Header tasks run in parallel, callback receives all results. Perfect for parallel processing with aggregation.',
    },
    {
      id: 'mc5',
      question: 'Why chunk large tasks?',
      options: [
        'To make tasks slower',
        'To break large task into smaller pieces that fit within time limits',
        'To use more memory',
        'Chunking is not necessary',
      ],
      correctAnswer: 1,
      explanation:
        'Chunking breaks large tasks into smaller pieces: (1) Each chunk < time limit, (2) Parallel processing (100 chunks = 100× faster), (3) Better failure handling (only failed chunks retry), (4) Memory efficient. Example: Process 1M records as 100 chunks of 10K records each.',
    },
  ];
