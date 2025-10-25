import { MultipleChoiceQuestion } from '@/lib/types';

export const debuggingAsyncApplicationsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'daa-mc-1',
      question: 'What does asyncio debug mode (debug=True) help detect?',
      options: [
        'Only syntax errors',
        'Forgot await, slow callbacks (>100ms), unawaited coroutines',
        'Network errors',
        'Memory leaks',
      ],
      correctAnswer: 1,
      explanation:
        'asyncio debug mode detects common async programming mistakes: Forgot await: Warns "coroutine X was never awaited". Example: result = async_func() # Missing await. Slow callbacks: Warns "Executing took X seconds" if callback blocks >100ms. Example: time.sleep(1) in async code (should be await asyncio.sleep(1)). Unawaited exceptions: Warns "Task exception was never retrieved". Example: task = create_task (failing_func()); # Never awaited. Enable: asyncio.run (main(), debug=True) or PYTHONASYNCIODEBUG=1. Performance: Adds overhead (use in development only). Debug mode is essential for catching async bugs early!',
    },
    {
      id: 'daa-mc-2',
      question: 'Why do background task exceptions disappear silently?',
      options: [
        "They don't, they always print",
        'create_task returns immediately; exception only raised when task.result() called',
        'Python ignores async exceptions',
        'Exceptions are logged automatically',
      ],
      correctAnswer: 1,
      explanation:
        'Background tasks (create_task without await) can fail silently: What happens: task = create_task (failing_func()); # Task starts running. Task raises exception. Exception stored in task object. If never awaited: Exception never retrieved (silent failure). Solution: Add done callback: def handle (t): try: t.result(); except Exception as e: logger.exception("Task failed"); task.add_done_callback (handle). Or await all tasks: tasks = [create_task (f()) for f in funcs]; await gather(*tasks, return_exceptions=True). Debug mode helps: Warns "Task exception was never retrieved". Always handle background task errors explicitly!',
    },
    {
      id: 'daa-mc-3',
      question: 'What does asyncio.all_tasks() return?',
      options: [
        'Completed tasks only',
        'A set of all currently running tasks in the event loop',
        'Only failed tasks',
        'Tasks waiting to start',
      ],
      correctAnswer: 1,
      explanation:
        'asyncio.all_tasks() returns set of all currently running tasks: Includes: All tasks created with create_task, Currently executing tasks, Suspended tasks (waiting at await). Excludes: Completed tasks (already done), Not-yet-created tasks. Use cases: Monitor active tasks: tasks = all_tasks(); print(f"Active: {len (tasks)}"). Find stuck tasks: for t in all_tasks(): if not t.done(): print(t.get_stack()). Debug memory leaks: Track task count over time. Example: async def monitor(): while True: tasks = all_tasks(); logger.info (f"Tasks: {len (tasks)}"); await sleep(10). Useful for debugging hung tasks, memory leaks, and monitoring.',
    },
    {
      id: 'daa-mc-4',
      question: 'What is the purpose of ContextVar in async debugging?',
      options: [
        'To store variables globally',
        'To maintain request-specific context across async calls (like request_id for tracing)',
        'To speed up async code',
        'To catch exceptions',
      ],
      correctAnswer: 1,
      explanation:
        'ContextVar maintains context across async operations without passing explicitly: Problem: Need request_id in every function for tracing. Passing explicitly is tedious. Global variables shared across requests (wrong). Solution: request_id = ContextVar("request_id"); async def handler (req): req_id = str (uuid.uuid4()); request_id.set (req_id); await process(). async def process(): req_id = request_id.get(); logger.info (f"[{req_id}] Processing"). Each async task has its own context (not shared). Benefits: Automatic propagation (no explicit passing), Request tracing (track across services), User context (auth, permissions). Use cases: Request IDs for logging, User authentication, Transaction IDs. Critical for distributed tracing in production!',
    },
    {
      id: 'daa-mc-5',
      question: 'Why should time.sleep() never be used in async code?',
      options: [
        'It causes syntax errors',
        'It blocks the entire event loop, preventing all other tasks from running',
        'It is slower than asyncio.sleep()',
        'It only works in sync code',
      ],
      correctAnswer: 1,
      explanation:
        'time.sleep() blocks the event loop, freezing all async tasks: What happens: Task calls time.sleep(5). Entire event loop blocks for 5 seconds. All other tasks frozen (can\'t run). Example: async def bad(): time.sleep(5) # ❌ Blocks everything. async def good(): await asyncio.sleep(5) # ✅ Non-blocking. Impact: 1000 concurrent requests: time.sleep(1) = All wait (1000s total!). await asyncio.sleep(1) = All run concurrently (1s total). Debug detection: asyncio debug mode warns: "Executing took 5.001 seconds". Rule: Never use blocking calls in async code. Use: await asyncio.sleep() not time.sleep(). await aiohttp.get() not requests.get(). await asyncpg.fetch() not psycopg2.execute().',
    },
  ];
