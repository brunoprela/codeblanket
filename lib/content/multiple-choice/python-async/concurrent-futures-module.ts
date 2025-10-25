import { MultipleChoiceQuestion } from '@/lib/types';

export const concurrentFuturesModuleMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cfm-mc-1',
    question:
      'What is the main difference between ThreadPoolExecutor and ProcessPoolExecutor?',
    options: [
      'They are identical',
      'ThreadPoolExecutor uses threads (shared memory, GIL), ProcessPoolExecutor uses processes (separate memory, no GIL)',
      'ThreadPoolExecutor is faster',
      'ProcessPoolExecutor is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'ThreadPoolExecutor vs ProcessPoolExecutor: ThreadPoolExecutor: Uses threads, Shared memory (low overhead), Subject to GIL (no CPU parallelism), Good for I/O-bound work. ProcessPoolExecutor: Uses processes, Separate memory (high overhead), No GIL (true parallelism), Good for CPU-bound work. Example: CPU task: ThreadPoolExecutor = no speedup (GIL). ProcessPoolExecutor = 4× speedup (4 cores). I/O task: ThreadPoolExecutor = 10× speedup (10 threads). ProcessPoolExecutor = 10× speedup but higher overhead. When to use: ThreadPoolExecutor: I/O operations, shared state, low overhead. ProcessPoolExecutor: CPU operations, true parallelism, isolated tasks.',
  },
  {
    id: 'cfm-mc-2',
    question: 'What does executor.map() return?',
    options: [
      'A list of results immediately',
      'An iterator that yields results in the order tasks were submitted',
      'A Future object',
      'None',
    ],
    correctAnswer: 1,
    explanation:
      'executor.map() returns an iterator that yields results in submission order. Behavior: Tasks execute concurrently, Results returned in submission order (not completion order), Iterator blocks when next result not ready. Example: results = executor.map(task, [1, 2, 3]); for r in results: print(r). If task(2) finishes first, iterator still waits for task(1). Compare to submit() with as_completed(): futures = [executor.submit(task, i) for i in [1,2,3]]; for future in as_completed(futures): print(future.result()). This yields results as they complete (any order). Use map() when: Need results in order, Simple batch processing. Use submit() when: Want results ASAP, Need timeouts/cancellation.',
  },
  {
    id: 'cfm-mc-3',
    question: 'What happens when you call future.result() without a timeout?',
    options: [
      'Returns None immediately',
      'Blocks indefinitely until the task completes or raises an exception',
      'Returns a default value',
      'Cancels the task',
    ],
    correctAnswer: 1,
    explanation:
      'future.result() without timeout blocks indefinitely until task completes. Behavior: If task done: Returns result immediately. If task running: Blocks until complete. If task failed: Raises the exception from task. If task cancelled: Raises CancelledError. Example: future = executor.submit(slow_task); result = future.result() # Blocks forever if slow_task never completes. Safer: Use timeout: result = future.result(timeout=10.0). Raises TimeoutError if not done in 10s. Or check first: if future.done(): result = future.result(). Best practice: Always use timeout for external operations (network, database). Prevents indefinite hangs.',
  },
  {
    id: 'cfm-mc-4',
    question: 'What does as_completed() do?',
    options: [
      'Cancels all futures',
      'Returns an iterator that yields futures as they complete (not in submission order)',
      'Waits for all futures to complete',
      'Sorts futures by completion time',
    ],
    correctAnswer: 1,
    explanation:
      "as_completed(futures) returns iterator that yields futures as they finish (any order). Behavior: Yields futures immediately when they complete, Not in submission order (in completion order), Allows progressive processing. Example: futures = [executor.submit(task, i) for i in range(10)]; for future in as_completed(futures): result = future.result(); process(result). If task(5) finishes first, it's yielded first (even though submitted later). Benefits: Process results ASAP (don't wait for all), Show progress as tasks complete, Lower latency (start processing early results). Compare to map(): map() waits for results in order. as_completed() yields results as ready. Use as_completed() when: Want progressive results, Different task durations, Progress tracking.",
  },
  {
    id: 'cfm-mc-5',
    question: 'Can you cancel a Future that is already running?',
    options: [
      'Yes, always',
      'No, cancel() only works on pending futures; returns False if already running',
      'Yes, but only for ThreadPoolExecutor',
      'Cancellation is not supported',
    ],
    correctAnswer: 1,
    explanation:
      'future.cancel() only cancels pending futures (not started). Returns True if cancelled, False if running/done. States: Pending: Not started yet. cancel() returns True (cancelled). Running: Currently executing. cancel() returns False (can\'t cancel). Done: Completed. cancel() returns False (already done). Example: future = executor.submit(task); if future.cancel(): print("Cancelled"); else: print("Too late, already running"). Why can\'t cancel running: Thread/process already executing task, No safe way to force stop (could corrupt state). Workaround: Cooperative cancellation: Check flag in task: def task(): while not stop_flag: # work. Then: stop_flag = True (task stops at next check). Best practice: Cancel quickly after submit if needed, Use timeout instead of cancel for running tasks.',
  },
];
