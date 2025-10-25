import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncContextManagersGeneratorsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'acmg-mc-1',
      question:
        'What is the purpose of the __aenter__ and __aexit__ methods in an async context manager?',
      options: [
        '__aenter__ validates input, __aexit__ returns output',
        '__aenter__ sets up resources when entering async with, __aexit__ cleans up when exiting',
        '__aenter__ starts a task, __aexit__ cancels it',
        '__aenter__ and __aexit__ are for debugging only',
      ],
      correctAnswer: 1,
      explanation:
        '__aenter__ is called when entering an async with block and is responsible for setting up resources (opening files, creating connections, acquiring locks). It can perform async operations and returns the resource to use. __aexit__ is called when exiting the block and handles cleanup (closing files, releasing locks, committing/rolling back transactions). It receives exception information (exc_type, exc_val, exc_tb) and can suppress exceptions by returning True. Example: class AsyncFile: async def __aenter__(self): self.file = await aiofiles.open("data.txt"); return self.file; async def __aexit__(self, exc_type, exc_val, exc_tb): await self.file.close(); return False. Key: __aexit__ is ALWAYS called (even on exceptions), guaranteeing cleanup. This exception-safety is why context managers are essential for resource management.',
    },
    {
      id: 'acmg-mc-2',
      question:
        'What is the difference between a regular generator and an async generator?',
      options: [
        'Async generators are faster than regular generators',
        'Async generators use yield in async functions and allow await between yields',
        'Async generators can only yield one value',
        'There is no difference, they are interchangeable',
      ],
      correctAnswer: 1,
      explanation:
        'Async generators combine async def with yield, allowing async operations between yields. Regular generator: def gen(): yield 1; yield 2 (synchronous). Async generator: async def agen(): await asyncio.sleep(1); yield 1; await asyncio.sleep(1); yield 2 (asynchronous). Key differences: (1) Iteration: Regular uses for x in gen(), async uses async for x in agen(). (2) Can await: Async generators can await between yields (I/O operations). (3) Returns: Regular returns generator object, async returns async generator object. Use case: Async generators are perfect for streaming I/O-bound data (fetching pages, reading files) where you want to yield results as they arrive, not load everything into memory first.',
    },
    {
      id: 'acmg-mc-3',
      question:
        'Why should you use async context managers for database connections instead of manual connection management?',
      options: [
        'Context managers make queries run faster',
        'Context managers guarantee connections are closed even if exceptions occur',
        'Context managers are required by database drivers',
        'Context managers automatically optimize queries',
      ],
      correctAnswer: 1,
      explanation:
        'Async context managers guarantee cleanup even when exceptions occur. Manual management risks: conn = await db.connect(); await conn.query(...); await conn.close() will leak the connection if query() raises an exception (close() never reached). Context manager fixes this: async with db.connect() as conn: await conn.query(...) calls __aexit__ (which closes the connection) automatically, even if an exception occurs. This prevents connection leaks that cause "too many connections" errors and service crashes. Additional benefits for transactions: async with conn.transaction() automatically commits on success or rolls back on exception, preventing partial updates. Context managers don\'t make queries faster or optimize them, but they make your code exception-safe and prevent resource leaks—critical for production stability.',
    },
    {
      id: 'acmg-mc-4',
      question: 'What is the @asynccontextmanager decorator used for?',
      options: [
        'To make regular functions async',
        'To simplify creating async context managers using yield instead of __aenter__/__aexit__',
        'To automatically parallelize context managers',
        'To add logging to context managers',
      ],
      correctAnswer: 1,
      explanation:
        '@asynccontextmanager is a decorator from contextlib that simplifies creating async context managers. Instead of defining a class with __aenter__ and __aexit__, you write a generator function: @asynccontextmanager; async def connection(): conn = await db.connect(); try: yield conn; finally: await conn.close(). Code before yield is like __aenter__ (setup), code in finally after yield is like __aexit__ (cleanup). This is much simpler than: class Connection: async def __aenter__(self): self.conn = await db.connect(); return self.conn; async def __aexit__(self, *args): await self.conn.close(). Use @asynccontextmanager for simple cases, class-based for complex context managers with multiple methods or state.',
    },
    {
      id: 'acmg-mc-5',
      question:
        'What is the memory benefit of using async generators for processing large datasets?',
      options: [
        'Async generators use less CPU than loading all data',
        "Async generators process only what's needed, loading one item at a time instead of all data",
        'Async generators compress data automatically',
        'Async generators use a special memory allocation algorithm',
      ],
      correctAnswer: 1,
      explanation:
        "Async generators enable streaming—processing one item at a time instead of loading everything into memory. Example: Loading all: records = await fetch_all_records() loads 1 million records × 1KB = 1GB memory. Streaming: async for record in fetch_records_streaming(): process(record) keeps only 1 record in memory (~1KB). Memory savings: 1GB → 1KB (1,000,000× reduction!). How it works: Generator yields one value, suspends until consumed, yields next value. Data flows through pipeline without accumulating. This allows processing arbitrarily large datasets (billions of records) with constant memory usage. Also enables: (1) Starting processing immediately (don't wait for all data), (2) Cancellation mid-stream, (3) Backpressure (slow consumer automatically slows producer). Essential pattern for big data processing in production.",
    },
  ];
