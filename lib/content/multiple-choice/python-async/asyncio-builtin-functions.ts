import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncioBuiltinFunctionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'abf-mc-1',
    question:
      'What is the purpose of asyncio.Semaphore and how does it differ from asyncio.Lock?',
    options: [
      'Semaphore allows N concurrent accesses, Lock allows only 1',
      'Semaphore is faster than Lock',
      'They are the same, just different names',
      'Semaphore is for threads, Lock is for coroutines',
    ],
    correctAnswer: 0,
    explanation:
      'Semaphore(N) allows N coroutines to access a resource concurrently. Lock is equivalent to Semaphore(1)—only 1 coroutine can hold it. Example: sem = Semaphore(3); async with sem: access_resource() allows 3 concurrent accesses. Lock: lock = Lock(); async with lock: critical_section() allows only 1. Use cases: Semaphore for rate limiting (max 10 API requests concurrent), connection pooling (max 50 connections), resource quotas. Lock for mutual exclusion (protect shared state, ensure only one writer). Both use same async with context manager syntax. Semaphore is generalization of Lock (Lock = Semaphore(1)).',
  },
  {
    id: 'abf-mc-2',
    question: 'What does asyncio.Queue.task_done() do and why is it important?',
    options: [
      'It removes an item from the queue',
      'It marks an item as processed, allowing queue.join() to complete',
      'It creates a new task for the item',
      'It returns the result of processing',
    ],
    correctAnswer: 1,
    explanation:
      'task_done() marks an item as processed after get() and processing. It decrements an internal counter that tracks pending items. queue.join() waits until this counter reaches zero (all items processed). Pattern: item = await queue.get(); process(item); queue.task_done(). Without task_done(), join() blocks forever (thinks items still pending). Use case: Graceful shutdown—await queue.join() waits for all items to be processed before stopping consumers. Critical: Must call task_done() exactly once per get(), no more (raises ValueError), no less (join() never completes). Common bug: Forgetting task_done() causes join() to hang.',
  },
  {
    id: 'abf-mc-3',
    question: 'Why should you never use time.sleep() in async code?',
    options: [
      "time.sleep() doesn't work in async functions",
      'time.sleep() blocks the entire event loop, preventing other tasks from running',
      'time.sleep() is slower than asyncio.sleep()',
      'time.sleep() causes memory leaks',
    ],
    correctAnswer: 1,
    explanation:
      'time.sleep() is a blocking call that freezes the entire event loop thread. While sleeping, NO other coroutines can run—defeats the purpose of async. Example: await gather(sleep_task(), other_task()). With time.sleep(5): other_task waits 5 seconds (sequential). With await asyncio.sleep(5): other_task runs during sleep (concurrent). Always use await asyncio.sleep() in async code. It yields control to the event loop, allowing other tasks to run. Same issue with all blocking I/O: requests.get() → use aiohttp, open().read() → use aiofiles, subprocess.run() → use create_subprocess_exec(). time.sleep() works syntactically but breaks concurrency.',
  },
  {
    id: 'abf-mc-4',
    question:
      'What is the difference between asyncio.wait_for() and setting a timeout in asyncio.wait()?',
    options: [
      'They are identical',
      'wait_for() times out a single operation and raises TimeoutError, wait() times out waiting for multiple and returns (done, pending)',
      'wait_for() is for coroutines, wait() is for tasks',
      'wait_for() is slower than wait()',
    ],
    correctAnswer: 1,
    explanation:
      'wait_for(coro, timeout=5) wraps a single coroutine with timeout, raises TimeoutError if exceeded, cancels the operation. Example: try: result = await wait_for(fetch(), 5); except TimeoutError: print("Timed out"). wait(tasks, timeout=5) waits for multiple tasks, returns after timeout without raising, gives you (done, pending) to decide what to do. Example: done, pending = await wait(tasks, timeout=5); partial_results = [t.result() for t in done]; for t in pending: t.cancel(). Use wait_for() for single operation with timeout. Use wait() for multiple operations where you want partial results on timeout. wait_for() is stricter (raises), wait() gives control.',
  },
  {
    id: 'abf-mc-5',
    question: 'What does asyncio.to_thread() do and when should you use it?',
    options: [
      'It creates a new thread for every async operation',
      "It runs blocking I/O in a thread pool so it doesn't block the event loop",
      'It converts regular functions to async functions',
      'It speeds up CPU-bound operations',
    ],
    correctAnswer: 1,
    explanation:
      "to_thread(blocking_func, *args) runs a blocking function in a thread pool, preventing it from blocking the event loop. Example: data = await asyncio.to_thread(requests.get, url) runs blocking requests in thread, event loop continues processing other tasks. Use when: (1) Library doesn't have async version (requests, PIL), (2) Legacy code you can't rewrite, (3) Blocking I/O that can't be made async. Limitation: Still limited by thread pool size (~32 threads), not true parallelism due to GIL. For CPU-bound work, use ProcessPoolExecutor instead (true parallelism). to_thread() doesn't make code faster, just prevents blocking the event loop.",
  },
];
