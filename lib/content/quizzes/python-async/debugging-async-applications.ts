export const debuggingAsyncApplicationsQuiz = [
  {
    id: 'daa-q-1',
    question:
      'Production async service experiencing intermittent failures. Implement comprehensive debugging: (1) Enable asyncio debug mode, (2) Add structured logging with request IDs, (3) Monitor all active tasks, (4) Detect slow callbacks (>100ms), (5) Track and log background task exceptions. Show how each catches specific bug types.',
    sampleAnswer:
      'Comprehensive async debugging: (1) Debug mode: asyncio.run (main(), debug=True) catches: forgot await, slow callbacks, unawaited coroutines. (2) Structured logging: request_id = ContextVar("request_id"); async def handler (req): req_id = str (uuid.uuid4()); request_id.set (req_id); logger.info (f"[{req_id}] Request started"); result = await process (req); logger.info (f"[{req_id}] Completed"). Traces request across all async operations. (3) Task monitoring: async def monitor(): while True: tasks = asyncio.all_tasks(); logger.info (f"Active: {len (tasks)}"); for t in tasks: if not t.done() and time_since_start (t) > 60: logger.warning (f"Stuck task: {t.get_name()}"); await asyncio.sleep(10). Detects hung tasks. (4) Slow callback detection: loop.slow_callback_duration = 0.1; # Warns if callback >100ms. Debug mode logs: "Executing took 0.250 seconds". (5) Background task errors: def create_task_safe (coro): task = asyncio.create_task (coro); def handle (t): try: t.result(); except Exception as e: logger.exception (f"Background task failed"); task.add_done_callback (handle); return task. Prevents silent failures. Each technique catches: Debug mode: developer mistakes. Logging: traces requests. Monitoring: detects hangs. Slow callback: finds blocking code. Error handling: catches silent failures.',
    keyPoints: [
      'Debug mode: asyncio.run (debug=True), catches forgot await, slow callbacks, unawaited coroutines',
      'Structured logging: ContextVar for request ID, traces across all operations',
      'Task monitoring: asyncio.all_tasks(), detect stuck tasks (>60s), log warnings',
      'Slow callbacks: loop.slow_callback_duration=0.1, warns if callback blocks >100ms',
      'Background errors: add_done_callback with exception logging, prevents silent failures',
    ],
  },
  {
    id: 'daa-q-2',
    question:
      'Async application has memory leak—tasks never complete. Debug: (1) List all active tasks, (2) Get stack traces for long-running tasks, (3) Find where tasks created, (4) Detect tasks without await. Implement task lifecycle tracker showing: created time, current state, stack trace.',
    sampleAnswer:
      'Task lifecycle tracker: class TaskTracker: def __init__(self): self.tasks = {}. def track_task (self, coro, name): task = asyncio.create_task (coro, name=name); self.tasks[id (task)] = {"task": task, "created": time.time(), "name": name, "stack": traceback.extract_stack()}; return task. async def report (self): print("=== Task Report ==="); for task_id, info in self.tasks.items(): task = info["task"]; age = time.time() - info["created"]; print(f"Task: {info[\'name\']}"); print(f"  Age: {age:.1f}s"); print(f"  State: {\'done\' if task.done() else \'running\'}"); if not task.done(): print(f"  Current stack:"); stack = task.get_stack(); for frame in stack: print(f"    {frame.f_code.co_filename}:{frame.f_lineno}"); print(f"  Created at:"); for frame in info["stack"]: print(f"    {frame.filename}:{frame.lineno}"). Usage: tracker = TaskTracker(); tracker.track_task (worker(), "Worker-1"); await tracker.report(). Shows: Long-running tasks (age >60s indicates leak). Current execution point (stack trace). Where task created (creation stack). Debug process: Run report periodically. Identify tasks with age >expected. Check current stack (where stuck). Check creation stack (where created). Cancel or fix root cause. Common causes: Infinite loop without await. Blocking I/O (requests instead of aiohttp). Deadlock (waiting for event that never comes).',
    keyPoints: [
      'Track tasks: Store creation time, name, creation stack trace at create_task',
      'Report: List age, state (done/running), current stack trace for running tasks',
      'Stack traces: task.get_stack() shows where currently executing, creation stack shows where created',
      'Identify leaks: Tasks with age >expected duration, infinite loops, blocking I/O',
      'Debug: Find stuck point (current stack), fix cause (blocking call, missing await)',
    ],
  },
  {
    id: 'daa-q-3',
    question:
      'Compare asyncio debug mode vs production logging vs distributed tracing. What does each catch? Implement: (1) Dev setup with debug mode and detailed logs, (2) Prod setup with structured logging and tracing, (3) Health check endpoint showing task status.',
    sampleAnswer:
      'Debug mode (development): asyncio.run (main(), debug=True); logging.basicConfig (level=DEBUG). Catches: Forgot await, slow callbacks (>100ms), unawaited exceptions. Use: Development only (performance overhead). Structured logging (production): logger.info("Event", extra={"request_id": req_id, "user_id": uid, "duration": elapsed}). Catches: Request flow, errors, performance metrics. Use: Production (parseable for analysis). Distributed tracing: request_id = ContextVar("request_id"); def trace (func): async def wrapper(*args): req_id = request_id.get() or str (uuid.uuid4()); request_id.set (req_id); logger.info (f"[{req_id}] {func.__name__} started"); result = await func(*args); logger.info (f"[{req_id}] {func.__name__} done"); return result; return wrapper. Catches: Cross-service request flow. Use: Microservices. Health check: @app.get("/health"); async def health(): tasks = asyncio.all_tasks(); return {"status": "healthy", "tasks": len (tasks), "details": [{"name": t.get_name(), "done": t.done()} for t in tasks]}. Comparison: Debug mode: Catches dev mistakes (use in dev). Structured logging: Tracks events (use in prod). Tracing: Follows requests (use for debugging prod issues). Health check: Monitors liveness (use for ops).',
    keyPoints: [
      'Debug mode: Catches forgot await, slow callbacks, dev only (overhead)',
      'Structured logging: Tracks events with context (request_id, user_id), production-ready',
      'Distributed tracing: ContextVar for request_id, traces across services',
      'Health check: asyncio.all_tasks(), shows active task count and state',
      'Use cases: Debug (dev), logging (prod), tracing (debug prod), health (ops monitoring)',
    ],
  },
];
