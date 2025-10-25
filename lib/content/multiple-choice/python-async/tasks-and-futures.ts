import { MultipleChoiceQuestion } from '@/lib/types';

export const tasksAndFuturesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'taf-mc-1',
    question:
      'What is the key difference between creating a task and just calling an async function?',
    options: [
      'Tasks are faster than async functions',
      'Tasks start running immediately when created, coroutines only run when awaited',
      'Tasks can only be used with asyncio.gather()',
      'Tasks require Python 3.11+, coroutines work in older versions',
    ],
    correctAnswer: 1,
    explanation:
      "asyncio.create_task() wraps a coroutine and schedules it to run immediately on the event loop. Just calling an async function returns a coroutine object that doesn't execute until awaited. Example: task = asyncio.create_task(fetch_data()) starts fetching immediately (non-blocking). coro = fetch_data() creates coroutine but doesn't start execution. Key use: Create multiple tasks before awaiting to achieve concurrency. task1 = create_task(op1()); task2 = create_task(op2()) starts both immediately. await task1; await task2 waits for both (already running concurrently). Without tasks: result1 = await op1(); result2 = await op2() runs sequentially (slow).",
  },
  {
    id: 'taf-mc-2',
    question:
      'When you cancel a task with task.cancel(), what must you do when catching CancelledError?',
    options: [
      'Nothing, the task is automatically cleaned up',
      'Re-raise the CancelledError after cleanup',
      'Return None to indicate cancellation',
      'Call task.cancel() again to confirm',
    ],
    correctAnswer: 1,
    explanation:
      "You MUST re-raise CancelledError after catching it and doing cleanup. Example: try: await long_operation(); except asyncio.CancelledError: await cleanup(); raise # Critical! If you don't re-raise: (1) task.cancelled() returns False (task appears to succeed normally), (2) higher-level cancellation breaks (e.g., in gather()), (3) caller doesn't know task was cancelled. Proper pattern: catch for cleanup, then raise to propagate cancellation status. The only exception: intentionally suppressing cancellation for critical operations that must complete (rare). Always re-raise unless you have specific reason not to.",
  },
  {
    id: 'taf-mc-3',
    question:
      'What does asyncio.gather(*tasks, return_exceptions=True) do differently than the default?',
    options: [
      'It makes tasks run faster',
      'It returns a list of results/exceptions instead of raising on first exception',
      'It only gathers tasks that succeed',
      'It automatically retries failed tasks',
    ],
    correctAnswer: 1,
    explanation:
      'return_exceptions=True changes error handling: Default: await asyncio.gather(task1, task2, task3) raises the first exception immediately, stopping execution. With return_exceptions=True: All tasks run to completion, exceptions are returned as results (not raised). Example: results = await gather(task1, task2, task3, return_exceptions=True); for i, r in enumerate(results): if isinstance(r, Exception): print(f"Task {i} failed: {r}"); else: print(f"Task {i} succeeded: {r}"). Use when: (1) You want all results even if some fail, (2) Partial success is useful, (3) Need to know which specific tasks failed. Example: fetching from 100 APIs—if 5 fail, still want the 95 successes.',
  },
  {
    id: 'taf-mc-4',
    question:
      'When should you use asyncio.as_completed() instead of asyncio.gather()?',
    options: [
      'When you need results in the order they were submitted',
      'When you want to process results as soon as each task completes',
      'When you need to stop on the first error',
      'When you need the fastest possible execution',
    ],
    correctAnswer: 1,
    explanation:
      'Use asyncio.as_completed() when you want to process results immediately as each task completes (not waiting for all). Example: Dashboard that shows user data progressively—for coro in asyncio.as_completed(tasks): data = await coro; render(data) shows each section as it loads (better UX). Compare to gather(): results = await gather(*tasks); for r in results: render(r) waits for ALL tasks before showing anything. Use cases: (1) Progressive rendering (show results as available), (2) Streaming data (process each item immediately), (3) Time-sensitive (act on first completions). Both run tasks concurrently (same speed), differ in when results are available to your code. as_completed yields in completion order, gather returns in submission order.',
  },
  {
    id: 'taf-mc-5',
    question: 'What is the relationship between Tasks and Futures in asyncio?',
    options: [
      'Tasks and Futures are the same thing',
      'Task is a subclass of Future that wraps a coroutine',
      'Futures are faster than Tasks',
      'Tasks are deprecated in favor of Futures',
    ],
    correctAnswer: 1,
    explanation:
      'Task is a subclass of Future specifically for running coroutines. Future is a low-level primitive representing a value that will be available in the future. Task extends Future to: (1) Wrap and execute a coroutine, (2) Schedule it on the event loop, (3) Provide cancellation, (4) Track exceptions. Use Tasks for coroutines (high-level): task = asyncio.create_task(my_coroutine()). Use Futures for callback-based code (low-level): future = loop.create_future(); callback sets future.set_result(value). In practice: Use asyncio.create_task() for your code (high-level, convenient). Futures are mainly for library authors integrating callback-based APIs with async/await. Both can be awaited, both track completion state, Task adds coroutine-specific functionality.',
  },
];
