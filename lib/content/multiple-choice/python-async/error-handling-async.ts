import { MultipleChoiceQuestion } from '@/lib/types';

export const errorHandlingAsyncMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'eha-mc-1',
    question:
      'Why is return_exceptions=True important when using asyncio.gather()?',
    options: [
      'It makes the code faster',
      'Without it, first exception stops gather() and remaining results are lost',
      'It automatically retries failed tasks',
      'It is not important, just optional',
    ],
    correctAnswer: 1,
    explanation:
      'Without return_exceptions=True, gather() stops at first exception and raises it, losing all remaining results. Example: await gather (task1(), task2(), task3()) where task2 fails: Without return_exceptions: task1 completes, task2 raises exception (stops gather), task3 result lost. With return_exceptions=True: All 3 tasks complete, results = [result1, Exception, result3]. Critical for batch operations where you want partial results. Pattern: results = await gather(..., return_exceptions=True); for r in results: if isinstance (r, Exception): handle_error (r); else: process (r). Always use return_exceptions=True for multiple independent tasks.',
  },
  {
    id: 'eha-mc-2',
    question:
      "What happens if you catch asyncio.CancelledError and don't re-raise it?",
    options: [
      'Nothing, it works fine',
      'Task appears to complete successfully but was actually cancelled, breaking cancellation propagation',
      'The program crashes',
      'The task automatically restarts',
    ],
    correctAnswer: 1,
    explanation:
      'CancelledError signals "task must stop". Not re-raising breaks cancellation propagation. Example: async def task(): try: await asyncio.sleep(10); except CancelledError: cleanup(); # No re-raise! return "done". Result: Task appears successful (returns "done"), but was cancelled. Caller doesn\'t know it was cancelled. Correct: except CancelledError: cleanup(); raise. This: Propagates cancellation up the stack, Marks task as cancelled (not successful), Enables caller to handle cancellation. Always re-raise CancelledError after cleanup.',
  },
  {
    id: 'eha-mc-3',
    question:
      'What does asyncio.wait_for (coro, timeout) do when timeout expires?',
    options: [
      'Waits longer',
      'Cancels the coroutine and raises asyncio.TimeoutError',
      'Returns None',
      'Retries the operation',
    ],
    correctAnswer: 1,
    explanation:
      'wait_for() enforces timeout by cancelling the coroutine when timeout expires and raising TimeoutError. Example: try: result = await asyncio.wait_for (slow_op(), timeout=5.0); except asyncio.TimeoutError: # slow_op was cancelled. Behavior: If completes within timeout: returns result. If exceeds timeout: cancels coroutine, raises TimeoutError. Important: Cancelled coroutine should handle CancelledError for cleanup. Pattern: try: result = await wait_for (operation(), timeout=T); except TimeoutError: log("Operation timed out"); # fallback. Critical for preventing operations from running forever.',
  },
  {
    id: 'eha-mc-4',
    question:
      'Why do background tasks need explicit error handling with add_done_callback()?',
    options: [
      "They don't need it",
      'Background task exceptions are not raised automatically, causing silent failures',
      'It makes them faster',
      'It is only for debugging',
    ],
    correctAnswer: 1,
    explanation:
      "Background tasks (create_task without await) don't raise exceptions automatically. Without handling, exceptions disappear (silent failures). Example: task = create_task (worker()); # Exception in worker() is lost! Correct: def handle_result (task): try: task.result(); except Exception as e: log_error (e). task = create_task (worker()); task.add_done_callback (handle_result). Now exceptions are logged. Alternative: Store tasks and await: tasks = []; tasks.append (create_task (worker())); results = await gather(*tasks, return_exceptions=True). Always handle background task exceptions explicitly.",
  },
  {
    id: 'eha-mc-5',
    question: 'What is the purpose of a circuit breaker pattern in async code?',
    options: [
      'To break circuits',
      'To fail fast and prevent cascading failures by rejecting requests to failing services',
      'To retry failed requests',
      'To improve performance',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breaker prevents cascading failures by failing fast when service is down. States: CLOSED (normal), OPEN (failing, reject requests), HALF_OPEN (testing recovery). Example: Database down, 1000 requests/sec: Without circuit breaker: All 1000 try to connect, wait for timeout (30s), waste resources. With circuit breaker: After 5 failures → OPEN. Remaining 995 requests fail immediately (fast fail). After 30s → HALF_OPEN (test one request). If success → CLOSED (resume). Benefits: Prevents overloading failing service, Reduces wasted resources (time, connections), Gives service time to recover. Use for external dependencies (database, API, cache).',
  },
];
