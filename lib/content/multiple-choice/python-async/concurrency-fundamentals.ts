import { MultipleChoiceQuestion } from '@/lib/types';

export const concurrencyFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cf-mc-1',
    question: 'What is the key difference between concurrency and parallelism?',
    options: [
      'Concurrency is faster than parallelism',
      'Concurrency is task switching on single core, parallelism is simultaneous execution on multiple cores',
      'Concurrency requires multiprocessing, parallelism uses threading',
      'Concurrency is for Python, parallelism is for other languages',
    ],
    correctAnswer: 1,
    explanation:
      'Concurrency is about dealing with multiple tasks at once by interleaving their execution (task switching), which can happen on a single CPU core. Parallelism is about doing multiple tasks simultaneously, which requires multiple CPU cores running tasks at exactly the same time. Example: Concurrency is like a chef switching between chopping vegetables and stirring a pot. Parallelism is like two chefs working on different dishes at the same time. In Python: asyncio provides concurrency (single-threaded task switching), multiprocessing provides parallelism (multiple processes on multiple cores).',
  },
  {
    id: 'cf-mc-2',
    question:
      'Why is asyncio particularly effective for I/O-bound operations but not for CPU-bound operations?',
    options: [
      'Asyncio is written in C and only optimizes I/O',
      "Asyncio allows other tasks to run while waiting for I/O, but CPU-bound tasks don't wait",
      'Asyncio requires special I/O hardware',
      'Asyncio only works with network operations',
    ],
    correctAnswer: 1,
    explanation:
      'Asyncio excels at I/O-bound operations because these operations involve waiting (for network responses, disk reads, database queries). During this wait time, asyncio can switch to other tasks, making progress on many tasks concurrently. CPU-bound operations involve continuous computation with no waiting—the CPU is always busy calculating. Since there\'s no "waiting time" to exploit, asyncio provides no benefit. Example: Fetching 100 web pages (I/O-bound): asyncio allows all 100 requests to wait concurrently, completing in ~1 second instead of 100 seconds sequentially. Computing 100 factorial calculations (CPU-bound): asyncio provides no speedup because each calculation requires continuous CPU time with no waiting.',
  },
  {
    id: 'cf-mc-3',
    question:
      "In the context of Python's Global Interpreter Lock (GIL), which concurrency approach is NOT limited by it?",
    options: [
      'Threading for CPU-bound operations',
      'Asyncio for I/O-bound operations',
      'Multiprocessing with separate processes',
      'All approaches are limited by the GIL',
    ],
    correctAnswer: 2,
    explanation:
      "Multiprocessing bypasses the GIL because each process has its own Python interpreter and memory space with its own GIL. This allows true parallelism for CPU-bound tasks across multiple cores. Threading is limited by the GIL for CPU-bound operations (only one thread executes Python bytecode at a time), though it can benefit I/O-bound operations when threads release the GIL during I/O waits. Asyncio runs in a single thread so the GIL doesn't matter for it—it achieves concurrency through cooperative task switching, not parallelism. Example: Resizing 100 images (CPU-bound): Threading with 4 threads takes same time as single thread (GIL blocks), but multiprocessing with 4 processes takes 1/4 the time (no GIL, true parallelism).",
  },
  {
    id: 'cf-mc-4',
    question:
      'What is the primary benefit of using async for a web server handling thousands of concurrent connections?',
    options: [
      'Async makes the CPU faster at processing requests',
      'Async uses less memory per connection and allows single-threaded handling of many concurrent I/O operations',
      'Async automatically scales across multiple servers',
      'Async prevents all security vulnerabilities',
    ],
    correctAnswer: 1,
    explanation:
      "Async web servers can handle thousands of concurrent connections efficiently because: (1) Low memory per connection: Each async task is a lightweight coroutine (~few KB) vs threads (~8MB per thread). 10,000 connections = ~50MB with async vs ~80GB with threading. (2) Single-threaded efficiency: No thread context switching overhead, no race conditions. (3) I/O multiplexing: While one connection waits for database/network, server processes other connections. Example: FastAPI async server can handle 10,000 concurrent connections on a single core with <100MB memory. Traditional threaded server would need 1,000 threads (practical limit) and gigabytes of memory. The server isn't faster at processing each request, but it can juggle many requests efficiently during their I/O wait times.",
  },
  {
    id: 'cf-mc-5',
    question:
      'Which statement about blocking vs non-blocking operations is correct?',
    options: [
      'Blocking operations are always faster than non-blocking',
      'Non-blocking operations allow the program to do other work while waiting for I/O',
      'Blocking operations should always be avoided',
      'Non-blocking operations only work with file I/O',
    ],
    correctAnswer: 1,
    explanation:
      "Non-blocking operations allow a program to continue executing other code while waiting for I/O to complete. Example: await asyncio.sleep(5) is non-blocking—the event loop can run other tasks during the 5 seconds. time.sleep(5) is blocking—the entire program freezes for 5 seconds. Blocking isn't always bad: For simple scripts with sequential operations, blocking code is simpler and more readable. For high-concurrency applications (web servers, scrapers), non-blocking is essential to handle many operations simultaneously. Non-blocking works with all I/O: network requests (aiohttp), database queries (asyncpg), file operations (aiofiles). Key: Use blocking for simplicity in low-concurrency scenarios, non-blocking for high-concurrency I/O-bound applications.",
  },
];
