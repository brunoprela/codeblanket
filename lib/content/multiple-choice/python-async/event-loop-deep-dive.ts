import { MultipleChoiceQuestion } from '@/lib/types';

export const eventLoopDeepDiveMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'eld-mc-1',
    question: 'What is the primary role of the event loop in asyncio?',
    options: [
      'To create multiple threads for parallel execution',
      'To manage and schedule coroutines, monitoring I/O and switching between tasks',
      'To automatically optimize CPU-bound operations',
      'To prevent race conditions in concurrent code',
    ],
    correctAnswer: 1,
    explanation:
      "The event loop is the orchestrator of async operations—it manages coroutines by scheduling when they run, monitoring I/O operations (network, files) for completion, and switching between tasks at await points. It runs in a single thread but creates the illusion of concurrency by rapidly switching between tasks when they're waiting for I/O. Example: When you await a network request, the event loop suspends that coroutine, runs other ready coroutines, and resumes the first coroutine when the network response arrives. The event loop does NOT create threads (asyncio is single-threaded), does NOT optimize CPU-bound work (that needs multiprocessing), and while it helps prevent some race conditions, that's not its primary purpose.",
  },
  {
    id: 'eld-mc-2',
    question:
      'Which method is the recommended way to run an async program in Python 3.7+?',
    options: [
      'loop = asyncio.get_event_loop(); loop.run_forever()',
      'asyncio.run (main())',
      'loop = asyncio.new_event_loop(); loop.run_until_complete (main())',
      'await main() directly',
    ],
    correctAnswer: 1,
    explanation:
      'asyncio.run (main()) is the recommended high-level API for running async programs in Python 3.7+. It handles everything automatically: creates a new event loop, runs your coroutine until completion, and properly closes the loop. Example: asyncio.run (main()) vs manual loop = asyncio.get_event_loop(); loop.run_until_complete (main()); loop.close(). The asyncio.run() version is simpler, safer (always closes loop), and prevents common mistakes. run_forever() is for servers that should run indefinitely (not typical programs). You cannot await main() directly outside an async context—Python will throw "SyntaxError: await outside async function".',
  },
  {
    id: 'eld-mc-3',
    question:
      'What happens when you call loop.call_soon (callback) vs loop.call_later (delay, callback)?',
    options: [
      'call_soon executes immediately, call_later schedules for future',
      'call_soon schedules for next loop iteration, call_later schedules after specified delay',
      'Both are identical, just different names',
      'call_soon uses threads, call_later uses the event loop',
    ],
    correctAnswer: 1,
    explanation:
      'loop.call_soon (callback) schedules the callback to run on the next iteration of the event loop—it doesn\'t execute immediately but as soon as the current task yields control. loop.call_later (delay, callback) schedules the callback to run after the specified delay (in seconds). Example: loop.call_soon (print, "A"); loop.call_later(1.0, print, "B"); await asyncio.sleep(0) will print "A" immediately (next iteration), then "B" after 1 second. Use call_soon() to break up long operations (yield control to other tasks). Use call_later() for timeouts and scheduled operations. Neither uses threads—both run in the single event loop thread.',
  },
  {
    id: 'eld-mc-4',
    question:
      'Why would you use loop.call_soon_threadsafe() instead of loop.call_soon()?',
    options: [
      'call_soon_threadsafe is faster than call_soon',
      'call_soon_threadsafe can be called from other threads safely, call_soon cannot',
      'call_soon_threadsafe guarantees execution order',
      'call_soon_threadsafe is needed for async functions',
    ],
    correctAnswer: 1,
    explanation:
      'loop.call_soon_threadsafe() is the thread-safe version that can be called from threads other than the event loop thread. The event loop is not thread-safe by default—if you call loop.call_soon() from another thread, you risk data corruption and crashes. Example: Thread worker thread calls loop.call_soon_threadsafe (lambda: print("Done")) to notify the event loop of completion. The event loop thread then executes the callback safely. Use case: Integrating blocking operations in threads with async code. Regular call_soon() should only be called from within the event loop thread. call_soon_threadsafe() uses locks internally (slightly slower but safe). Not about execution order or async functions—about thread safety.',
  },
  {
    id: 'eld-mc-5',
    question: 'What is uvloop and why would you use it in production?',
    options: [
      'A debugging tool for finding event loop bugs',
      'A high-performance drop-in replacement for asyncio event loop, 2-4× faster',
      'A library for running multiple event loops in parallel',
      'An alternative to asyncio with a different API',
    ],
    correctAnswer: 1,
    explanation:
      "uvloop is a high-performance event loop implementation that's a drop-in replacement for the default asyncio event loop. It\'s written in Cython and wraps libuv (the C library that powers Node.js), achieving 2-4× better performance than standard asyncio. Migration is trivial: import uvloop; asyncio.set_event_loop_policy (uvloop.EventLoopPolicy()); asyncio.run (main()). Your existing asyncio code works unchanged with better performance. Use in production when: (1) You need maximum I/O throughput (>10K req/sec), (2) Deploying on Linux/Unix (uvloop doesn't support Windows), (3) Performance profiling shows event loop as bottleneck. It's not a debugging tool, doesn't run multiple loops in parallel, and uses the same asyncio API.",
  },
];
