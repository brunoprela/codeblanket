import { MultipleChoiceQuestion } from '@/lib/types';

export const coroutinesAsyncAwaitMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'caa-mc-1',
    question:
      'What is the difference between a coroutine function and a coroutine object?',
    options: [
      'There is no difference, they are the same thing',
      'A coroutine function is defined with async def, a coroutine object is returned when you call it',
      'A coroutine function is for Python 3.5+, a coroutine object is for older versions',
      'A coroutine function is faster than a coroutine object',
    ],
    correctAnswer: 1,
    explanation:
      'A coroutine function is defined with async def—it\'s the definition/template. When you call a coroutine function, it returns a coroutine object—an instance that can be executed. Example: async def greet(name): return f"Hello {name}" defines a coroutine function. coro = greet("Alice") creates a coroutine object (hasn\'t executed yet). await coro or asyncio.run(coro) actually executes it. Analogy: Class (coroutine function) vs instance (coroutine object). A common mistake is calling a coroutine function without await, which just creates the object without executing it—Python 3.11+ warns "coroutine was never awaited".',
  },
  {
    id: 'caa-mc-2',
    question:
      'What happens if you forget to use await before calling an async function?',
    options: [
      'The function executes synchronously',
      'You get a coroutine object instead of the result, and Python warns it was never awaited',
      'Python automatically adds await for you',
      "The function doesn't execute at all and raises an exception",
    ],
    correctAnswer: 1,
    explanation:
      'Forgetting await is the most common async mistake. When you call an async function without await, you get a coroutine object (not the result), and the function body never executes. Example: async def get_data(): return "data"; result = get_data() sets result to <coroutine object> (not "data"). Python 3.11+ shows: "RuntimeWarning: coroutine \'get_data\' was never awaited". Correct: result = await get_data(). The coroutine object sits unused until you await it or pass it to asyncio.run(). It doesn\'t execute synchronously, Python doesn\'t auto-add await, and it doesn\'t raise an exception (just a warning).',
  },
  {
    id: 'caa-mc-3',
    question:
      'Which approach correctly runs three async operations concurrently?',
    options: [
      'results = [await op1(), await op2(), await op3()]',
      'results = await asyncio.gather(op1(), op2(), op3())',
      'results = [op1(), op2(), op3()]; await results',
      'results = async [op1(), op2(), op3()]',
    ],
    correctAnswer: 1,
    explanation:
      'asyncio.gather() is the standard way to run multiple coroutines concurrently. It starts all operations immediately and waits for all to complete, returning results in order. Example: results = await asyncio.gather(fetch(url1), fetch(url2), fetch(url3)) starts all three fetches at once (concurrent), completing in max(times) not sum(times). Option A [await op1(), await op2(), await op3()] is sequential—waits for op1 before starting op2. Option C creates coroutine objects but never executes them. Option D has invalid syntax. Gather provides concurrency: if each op takes 1s, gather takes ~1s total vs 3s sequential.',
  },
  {
    id: 'caa-mc-4',
    question:
      'What is the purpose of async generators (functions with async def and yield)?',
    options: [
      'To make regular generators run faster',
      'To generate values asynchronously, allowing I/O operations between yields',
      'To convert sync generators to async automatically',
      'To yield multiple values at once',
    ],
    correctAnswer: 1,
    explanation:
      "Async generators (async def + yield) allow you to yield values with async I/O between yields. Use async for to iterate them. Example: async def fetch_pages(urls): for url in urls: data = await fetch(url); yield data. This yields each page as it's fetched (streaming), not waiting for all. Compare to list: pages = await fetch_all(urls) must wait for all before returning. Async generators enable: (1) Streaming large datasets without loading all in memory, (2) Processing results as they arrive (lower latency), (3) Backpressure (consumer controls rate). They don't make generators faster, don't convert sync generators, and yield one value at a time (like regular generators).",
  },
  {
    id: 'caa-mc-5',
    question:
      'Why is using time.sleep() in an async function considered bad practice?',
    options: [
      'time.sleep() is slower than asyncio.sleep()',
      'time.sleep() blocks the entire event loop, preventing other tasks from running',
      'time.sleep() only works in synchronous code',
      'time.sleep() causes memory leaks in async code',
    ],
    correctAnswer: 1,
    explanation:
      "time.sleep() is a blocking call that freezes the entire event loop thread. While sleeping, NO other async tasks can run—defeating the purpose of async. Example: async def bad(): time.sleep(5) blocks loop for 5 seconds (nothing else runs). async def good(): await asyncio.sleep(5) yields control—other tasks run during the 5 seconds. Demo: await asyncio.gather(bad(), other_task()) makes other_task wait 5 seconds unnecessarily. With asyncio.sleep(), other_task runs concurrently. Always use: await asyncio.sleep() for delays, aiohttp for HTTP (not requests), aiofiles for file I/O (not open), asyncpg for database (not psycopg2). time.sleep() isn't slower (same duration), works in sync code, and doesn't cause memory leaks—the issue is blocking.",
  },
];
