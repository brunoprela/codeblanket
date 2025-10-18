/**
 * Quiz questions for Async/Await & Asynchronous Programming section
 */

export const asyncawaitQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between concurrency and parallelism. How does async/await enable concurrency in Python?',
    hint: 'Think about single vs multiple CPU cores, and how async handles I/O wait time.',
    sampleAnswer:
      "Concurrency means multiple tasks make progress during overlapping time periods, but not necessarily simultaneously. Parallelism means tasks execute simultaneously on multiple CPU cores. Async/await enables concurrency through cooperative multitasking: when an async function awaits an I/O operation (like a network request), it yields control back to the event loop, which can then run other tasks. This is perfect for I/O-bound operations where tasks spend time waiting. However, it doesn't help with CPU-bound tasks because Python still runs on a single thread. For example, with async you can handle 1000 network requests concurrently on one CPU core because most time is spent waiting, not computing.",
    keyPoints: [
      'Concurrency: tasks make progress during overlapping periods',
      'Parallelism: tasks execute simultaneously on multiple cores',
      'Async enables concurrency via cooperative multitasking',
      'Perfect for I/O-bound, not CPU-bound operations',
      'Single-threaded but handles many tasks concurrently',
    ],
  },
  {
    id: 'q2',
    question:
      "Why can't you use regular blocking functions like time.sleep() or requests.get() in async code? What happens if you do?",
    hint: 'Consider what happens to other async tasks while one task is blocked.',
    sampleAnswer:
      'Blocking functions freeze the entire event loop, preventing all other async tasks from running. When you call time.sleep(1), the entire program pauses—no other async tasks can execute during that second. This defeats the purpose of async programming. You must use async versions: asyncio.sleep() for delays, aiohttp for HTTP requests, asyncpg for databases. These async functions yield control to the event loop during I/O, allowing other tasks to run. If you absolutely must use blocking code, use asyncio.to_thread() or loop.run_in_executor() to run it in a thread pool, preventing it from blocking the event loop.',
    keyPoints: [
      'Blocking functions freeze the entire event loop',
      'Prevents all other async tasks from running',
      'Must use async equivalents (asyncio.sleep, aiohttp)',
      'Async functions yield control during I/O',
      'Use loop.run_in_executor() for blocking code if needed',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the event loop and how does it schedule async tasks? How does this differ from threads?',
    hint: 'Think about context switching, memory overhead, and how control is yielded.',
    sampleAnswer:
      "The event loop is a single-threaded scheduler that manages async tasks. It maintains a queue of tasks and runs them cooperatively: each task explicitly yields control (at await points), allowing the loop to switch to other tasks. This is different from threads where the OS preemptively switches between threads at any time. Advantages: 1) No race conditions (tasks only switch at await), 2) Lower memory overhead (tasks are lightweight), 3) Predictable switching points. Disadvantages: 1) One blocking call ruins everything, 2) Doesn't utilize multiple CPU cores. The event loop makes async programming safer and more efficient than threads for I/O-bound workloads.",
    keyPoints: [
      'Event loop: single-threaded scheduler for async tasks',
      'Cooperative multitasking: tasks yield control at await',
      'Threads: preemptive multitasking by OS',
      'Async: lighter weight, no race conditions',
      "Trade-off: requires discipline, can't use blocking calls",
    ],
  },
];
