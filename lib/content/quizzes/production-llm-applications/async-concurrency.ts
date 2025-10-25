export const asyncConcurrencyQuiz = [
  {
    id: 'pllm-q-3-1',
    question:
      'Explain how async/await and concurrency patterns dramatically improve LLM application performance. Provide a specific example where proper async implementation reduces processing time from 30 seconds to 3 seconds for 10 API calls. What are the potential pitfalls and how do you avoid them?',
    sampleAnswer:
      'Synchronous execution processes API calls sequentially: 10 calls × 3s each = 30s total. Async with asyncio.gather() processes concurrently: max(3s) ≈ 3s total (10x speedup). Implementation: use AsyncOpenAI client, create tasks with asyncio.create_task(), gather with asyncio.gather(*tasks). Pitfalls: 1) Overwhelming API with too many concurrent requests → use Semaphore(10) to limit concurrency, 2) Ignoring rate limits → implement token bucket rate limiter, 3) Not handling errors properly → use return_exceptions=True in gather, 4) Memory issues with large batches → process in chunks, 5) Blocking operations in async code → use run_in_executor() for CPU-bound tasks. Best practices: limit concurrency to 10-50 based on API limits, implement exponential backoff for retries, use connection pooling, set timeouts on all operations, monitor queue depth. Example: processing 1000 documents - synchronous would take 50 minutes, async with 20 concurrent workers takes 2.5 minutes.',
    keyPoints: [
      'Concurrent execution with asyncio.gather provides 10x+ speedup',
      'Use Semaphore to limit concurrency and avoid overwhelming APIs',
      'Proper error handling and rate limiting are critical',
    ],
  },
  {
    id: 'pllm-q-3-2',
    question:
      'Design a production-ready async queue processing system for LLM tasks that handles backpressure, provides progress updates, implements graceful shutdown, and scales based on queue depth. Include specific implementation details.',
    sampleAnswer:
      'Architecture: AsyncIO Queue with maxsize=100 to prevent memory issues, multiple worker coroutines (scale 3-20 based on load), Redis for persistent queue backup. Backpressure: reject new requests when queue full (return 429), provide queue depth in response, implement priority queues for urgent requests. Progress updates: use task.update_state() in Celery or WebSocket notifications, track percentage complete, estimate time remaining. Graceful shutdown: set shutdown_event flag, allow current tasks to complete (await asyncio.wait (tasks)), save incomplete tasks to Redis for restart, timeout after 30s and force shutdown. Auto-scaling: monitor queue depth every 10s, scale up workers when depth >50 for >1min, scale down when <10 for >5min, min 3 workers, max 50 workers. Implementation: async def worker (queue, shutdown_event): while not shutdown_event.is_set(): task = await asyncio.wait_for (queue.get(), timeout=1.0); await process_task (task). Use FastAPI BackgroundTasks for simple cases, Celery for complex workflows, monitor with Prometheus metrics.',
    keyPoints: [
      'Bounded queue with rejection when full for backpressure',
      'Graceful shutdown with task completion and state persistence',
      'Auto-scaling workers based on queue depth metrics',
    ],
  },
  {
    id: 'pllm-q-3-3',
    question:
      'Compare thread pools, process pools, and async/await for different types of LLM workloads. When would you use each approach, and how would you combine them effectively in a production system?',
    sampleAnswer:
      'Async/await: Best for I/O-bound operations (API calls, database queries, file I/O). Use for LLM API calls, cache lookups, database operations. Pros: lightweight, efficient, handles thousands of concurrent operations. Cons: blocked by CPU-intensive operations. Thread pools: Best for CPU-bound preprocessing (tokenization, embedding calculation, text processing). Use ThreadPoolExecutor with loop.run_in_executor(). Pros: parallel CPU work, shared memory. Cons: GIL limits true parallelism for pure Python. Process pools: Best for heavy CPU work requiring true parallelism (batch embedding generation, model fine-tuning). Use ProcessPoolExecutor. Pros: bypasses GIL, true parallelism. Cons: higher overhead, no shared memory. Combined approach: main async event loop for API/IO, thread pool for light CPU work (parsing), process pool for heavy computation (embeddings). Example: async API call → thread pool tokenization → async DB storage. Implementation: executor = ThreadPoolExecutor (max_workers=4); await loop.run_in_executor (executor, cpu_function, data). Monitor with asyncio debug mode, profile with cProfile.',
    keyPoints: [
      'Async for I/O-bound (API calls), threads for light CPU work, processes for heavy CPU',
      'Combine approaches: async main loop with executors for CPU work',
      'Choose based on workload characteristics and profiling data',
    ],
  },
];
