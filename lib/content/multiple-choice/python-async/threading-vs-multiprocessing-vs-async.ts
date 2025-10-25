import { MultipleChoiceQuestion } from '@/lib/types';

export const threadingVsMultiprocessingVsAsyncMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tvmva-mc-1',
      question:
        'What is the Global Interpreter Lock (GIL) and how does it affect threading?',
      options: [
        'The GIL locks threads permanently',
        'The GIL allows only one thread to execute Python bytecode at a time, preventing CPU-bound parallelism',
        'The GIL only affects async code',
        'The GIL makes Python faster',
      ],
      correctAnswer: 1,
      explanation:
        'The GIL (Global Interpreter Lock) is a mutex that prevents multiple threads from executing Python bytecode simultaneously. Impact: CPU-bound work: No parallelism (only 1 thread executes at a time). Example: 4 threads computing = same speed as 1 thread. I/O-bound work: Threads work well (GIL released during I/O). Example: 10 threads fetching URLs = 10× faster. Why GIL exists: Protects Python object memory management (reference counting). Simplifies C extension development. When GIL releases: I/O operations (network, file, database), time.sleep(), Some C extensions (NumPy). Consequence: Use multiprocessing for CPU parallelism (separate processes = separate GILs). Use threading for I/O concurrency (GIL releases during I/O).',
    },
    {
      id: 'tvmva-mc-2',
      question:
        'For CPU-bound work, why is multiprocessing faster than threading in Python?',
      options: [
        'Multiprocessing uses more memory',
        'Each process has its own Python interpreter and GIL, enabling true parallelism',
        'Processes are always faster than threads',
        'Threading is deprecated',
      ],
      correctAnswer: 1,
      explanation:
        "Multiprocessing achieves true parallelism by creating separate processes, each with its own Python interpreter and GIL. Threading: All threads share 1 GIL → only 1 thread executes at a time. 4 threads on 4-core CPU: No speedup (GIL serializes execution). Multiprocessing: Each process has own GIL → all execute simultaneously. 4 processes on 4-core CPU: 4× speedup (true parallelism). Trade-off: Multiprocessing overhead: Process creation (~100ms), inter-process communication (expensive), more memory (separate interpreters). Use when: CPU-bound work (computation), task duration > 100ms (overhead amortized). Don't use when: I/O-bound (threading/async better), small tasks (overhead dominates), need shared state (IPC expensive).",
    },
    {
      id: 'tvmva-mc-3',
      question:
        'When is async (asyncio) preferred over threading for I/O-bound work?',
      options: [
        'Never, threading is always better',
        'When you need high concurrency (100+) with low memory overhead',
        'When you need CPU parallelism',
        'Async and threading are identical',
      ],
      correctAnswer: 1,
      explanation:
        'Async excels at high-concurrency I/O with minimal overhead. Threading: Each thread = ~8MB stack memory. 1000 threads = 8GB memory (unsustainable). Context switching overhead (OS-level). Async: Each coroutine = ~1KB memory. 10,000 coroutines = 10MB memory (sustainable). Single thread (no context switching). When to use async: High concurrency (>100 concurrent operations), Memory constrained, Long-lived connections (WebSocket), Need 10K+ concurrent I/O. When threading sufficient: Low concurrency (<100), Simple I/O operations, Legacy blocking libraries, Mixed with CPU work. Example: 10,000 concurrent HTTP requests: Threading: Impossible (80GB memory). Async: Easy (10MB memory, 1-2 seconds). Limitation: Async requires async libraries (aiohttp, asyncpg, not requests/psycopg2).',
    },
    {
      id: 'tvmva-mc-4',
      question:
        'What happens if you use threading for CPU-bound work in Python?',
      options: [
        'It works perfectly',
        'Performance is often worse than sequential due to GIL contention and context switching',
        'It automatically uses multiprocessing',
        'The GIL is disabled',
      ],
      correctAnswer: 1,
      explanation:
        'Using threading for CPU-bound work often performs worse than sequential execution due to GIL contention. What happens: Thread 1 acquires GIL, starts computation. Thread 2 waits for GIL. Context switch: Thread 1 releases GIL, Thread 2 acquires. Repeat (frequent context switches). Result: Computation serialized (1 thread at a time). Overhead added (context switching). Benchmark: Sequential: 1.0s. Threading (4 threads): 1.2s (20% slower!). Multiprocessing (4 processes): 0.25s (4× faster). Why slower: Context switching overhead: OS switches threads frequently. GIL contention: Threads compete for GIL. No true parallelism: Only 1 thread executes at a time. Correct approach: Use multiprocessing for CPU work. Reserve threading for I/O work.',
    },
    {
      id: 'tvmva-mc-5',
      question:
        'In a hybrid pipeline (I/O → CPU → I/O), which combination is optimal?',
      options: [
        'All sequential',
        'Async for I/O stages, multiprocessing for CPU stage',
        'All threading',
        'All async',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid pipelines should use the optimal concurrency model for each stage. Pipeline: Read data (I/O) → Transform (CPU) → Write data (I/O). Optimal approach: Stage 1 (Read): Async (aiofiles). High-throughput I/O, minimal overhead. Stage 2 (Transform): Multiprocessing (ProcessPoolExecutor). True CPU parallelism, bypass GIL. Stage 3 (Write): Async (asyncpg). High-throughput database writes. Architecture: async def pipeline(): async for chunk in read_async(): result = await process_in_executor (transform, chunk); await write_async (result). Benefits: Each stage uses optimal model. Async I/O: Low overhead, high concurrency. Multiprocessing CPU: True parallelism. Queues decouple stages (backpressure). Why not all async: CPU work blocks event loop (no parallelism). Why not all multiprocessing: High overhead for I/O. Why not all threading: No CPU parallelism, higher memory than async.',
    },
  ];
