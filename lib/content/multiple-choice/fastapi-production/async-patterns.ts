import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncPatternsMultipleChoice = [
  {
    id: 1,
    question:
      'When should you use async endpoints vs sync endpoints in FastAPI?',
    options: [
      'Use async for I/O-bound operations (database, HTTP requests, file I/O) and sync for CPU-bound operations (calculations, data processing)',
      "Always use async because it's faster",
      'Use sync for everything to keep code simple',
      'Async is only for WebSockets',
    ],
    correctAnswer: 0,
    explanation:
      "Async shines for I/O-bound operations where you're waiting for external resources. While waiting for database or HTTP response, async yields control so other requests can execute - this is concurrency, not parallelism. For CPU-bound operations (calculations, data processing), sync is appropriate because FastAPI runs sync endpoints in a thread pool. Pattern: async for await operations (database queries, HTTP calls), sync for compute (pandas processing, image manipulation). Using async everywhere (option 2) wastes resources on CPU tasks. Using only sync (option 3) misses concurrency benefits.",
  },
  {
    id: 2,
    question:
      'What happens when you use time.sleep(5) in an async FastAPI endpoint?',
    options: [
      'It blocks the entire event loop, preventing ALL other requests from being processed for 5 seconds',
      'Only that specific request is delayed',
      'FastAPI automatically converts it to await asyncio.sleep(5)',
      'It has no effect in async functions',
    ],
    correctAnswer: 0,
    explanation:
      "time.sleep() is blocking - it doesn't yield control back to the event loop. Scenario: User A calls endpoint with time.sleep(5), while sleeping the event loop is blocked, User B's request arrives but can't be processed, User C, D, E also blocked. After 5 seconds, all requests flood through. This defeats the entire purpose of async! Fix: Use await asyncio.sleep(5) which yields control. FastAPI doesn't auto-convert (option 3). The effect is catastrophic (option 4) - blocks everything. This is the #1 async mistake developers make.",
  },
  {
    id: 3,
    question: 'What does asyncio.gather() do and when should you use it?',
    options: [
      'Executes multiple async operations concurrently and waits for all to complete, ideal for independent I/O operations that can run in parallel',
      'Executes operations sequentially in order',
      'Gathers data from multiple functions',
      'Only works with database queries',
    ],
    correctAnswer: 0,
    explanation:
      "asyncio.gather() runs multiple coroutines concurrently. Example: results = await asyncio.gather(fetch_user(), fetch_posts(), fetch_comments()) executes all three simultaneously, total time = slowest operation (not sum). Benefits: 3x speedup if each takes 1 second. Use when: operations are independent (don't depend on each other), all are I/O-bound, want to wait for all results. Options: return_exceptions=True to handle individual failures. It doesn't run sequentially (option 2), works with any async operations (option 4), and does more than collect data (option 3) - it enables concurrency.",
  },
  {
    id: 4,
    question: 'How does async improve database performance in FastAPI?',
    options: [
      'Async allows the event loop to handle other requests while waiting for database I/O, enabling thousands of concurrent connections with fewer resources',
      'Async makes database queries execute faster',
      'Async caches all database results automatically',
      'Async creates more database connections',
    ],
    correctAnswer: 0,
    explanation:
      "Async doesn't make individual queries faster (option 2), but enables better concurrency. Sync: 1 request → query DB (blocks thread) → wait 10ms → return. 1000 requests = 1000 threads (expensive!). Async: 1 request → query DB → await (yields) → event loop handles request 2, 3, 4... → DB returns → resume request 1. 1000 requests = 1 event loop thread + efficient I/O. Result: Same hardware serves 10x more concurrent users. Async doesn't cache (option 3) or create more connections (option 4) - it uses connections efficiently. With async, connection pool of 20 can serve thousands of concurrent requests.",
  },
  {
    id: 5,
    question:
      'What is a common pitfall when mixing sync and async code in FastAPI?',
    options: [
      'Calling synchronous blocking operations in async functions blocks the event loop and defeats the purpose of async',
      'Mixing sync and async code causes syntax errors',
      'You cannot mix sync and async in the same application',
      'Async code automatically converts sync operations',
    ],
    correctAnswer: 0,
    explanation:
      "Common mistake: async def endpoint(): result = sync_db_query() (blocks event loop!). Sync operations in async functions don't automatically become async (option 4). Mixing is allowed (option 3) but requires care. Pattern for fixing: Use asyncio.to_thread() to run blocking code: result = await asyncio.to_thread(sync_function). Or use async alternatives: async with httpx.AsyncClient(), async SQLAlchemy. The danger: One blocking call in an async endpoint blocks all other requests. Not a syntax error (option 2) but a performance killer. Always use async versions of libraries in async endpoints.",
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
