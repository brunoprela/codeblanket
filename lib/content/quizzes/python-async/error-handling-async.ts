export const errorHandlingAsyncQuiz = [
  {
    id: 'eha-q-1',
    question:
      'Build a robust API client that: (1) Retries failed requests with exponential backoff (3 attempts), (2) Uses circuit breaker to stop calling failing service, (3) Handles timeouts (5s per request), (4) Cancels all pending requests on shutdown, (5) Logs all errors with context. Explain why return_exceptions=True is critical when calling 100 endpoints concurrently.',
    sampleAnswer:
      'Robust API client: class APIClient: def __init__(self): self.circuit_breaker = CircuitBreaker (threshold=5, timeout=30); self.session = None. async def request (self, url, retries=3): for attempt in range (retries): try: async with self.circuit_breaker: result = await asyncio.wait_for (self.session.get (url), timeout=5.0); return result; except asyncio.TimeoutError: if attempt == retries - 1: raise; delay = 2 ** attempt; await asyncio.sleep (delay); except Exception as e: logger.error (f"Request to {url} failed: {e}"); raise. async def fetch_all (self, urls): results = await asyncio.gather(*[self.request (url) for url in urls], return_exceptions=True); successes = [r for r in results if not isinstance (r, Exception)]; failures = [r for r in results if isinstance (r, Exception)]; return successes, failures. Why return_exceptions=True critical: Without: First exception stops gather(), remaining 99 results lost. With: All 100 requests complete, exceptions returned as results. Example: 100 URLs, 5 fail: Without: 5 successes + exception (95 lost). With: 95 successes + 5 exceptions (all results). Critical for batch operations: want partial results, not all-or-nothing.',
    keyPoints: [
      'Retry: exponential backoff (2^attempt seconds), max 3 attempts, re-raise on final attempt',
      'Circuit breaker: track failures, open after threshold (5), reject requests for timeout (30s)',
      'Timeout: wait_for (request, 5.0), raises asyncio.TimeoutError, cancel request',
      'Shutdown: cancel all tasks, await gather(*tasks, return_exceptions=True)',
      "return_exceptions: critical for concurrent ops, first exception doesn't stop others, returns all results",
    ],
  },
  {
    id: 'eha-q-2',
    question:
      'Design graceful shutdown for long-running service with 50 worker tasks: (1) Catch SIGINT/SIGTERM signals, (2) Stop accepting new work, (3) Let current work finish (30s grace period), (4) Cancel remaining tasks after timeout, (5) Save state before exit. Explain why you must re-raise asyncio.CancelledError.',
    sampleAnswer:
      'Graceful shutdown: class Service: def __init__(self): self.shutdown_event = asyncio.Event(); self.workers = []; self.queue = asyncio.Queue(). async def worker (self, name): try: while not self.shutdown_event.is_set(): try: task = await asyncio.wait_for (self.queue.get(), timeout=1.0); await self.process (task); except asyncio.TimeoutError: continue; except asyncio.CancelledError: logger.info (f"{name} saving state..."); await self.save_state(); raise. async def shutdown (self): self.shutdown_event.set(); self.queue.put(None); # Stop accepting work; try: await asyncio.wait_for (asyncio.gather(*self.workers, return_exceptions=True), timeout=30); except asyncio.TimeoutError: for w in self.workers: w.cancel(); await asyncio.gather(*self.workers, return_exceptions=True). Why re-raise CancelledError: CancelledError signals "task must stop". Not re-raising: Task appears to complete successfully, caller doesn\'t know it was cancelled. Re-raising: Propagates cancellation up the stack, ensures cleanup happens. Example: Without re-raise: task cancelled but returns None (looks like success). With re-raise: task.exception() = CancelledError (correct state).',
    keyPoints: [
      'Signal handling: loop.add_signal_handler(SIGINT, shutdown), set shutdown_event',
      'Grace period: wait_for (gather, 30s), let workers finish current task',
      'Force cancel: after timeout, cancel() all workers, await completion',
      'State saving: in CancelledError handler (finally block), before re-raising',
      'Re-raise CancelledError: propagates cancellation, maintains correct task state, enables cleanup',
    ],
  },
  {
    id: 'eha-q-3',
    question:
      'Implement circuit breaker pattern for database queries: States are CLOSED (normal), OPEN (failing, reject), HALF_OPEN (testing recovery). (1) Track failures (5 in 60s → OPEN), (2) OPEN state rejects for 30s, (3) HALF_OPEN allows 1 test query, (4) Success → CLOSED, failure → OPEN. Why is this better than simple retry?',
    sampleAnswer:
      "Circuit breaker: class CircuitBreaker: def __init__(self): self.state = CLOSED; self.failures = []; self.opened_at = None. async def call (self, func): if self.state == OPEN: if time.time() - self.opened_at > 30: self.state = HALF_OPEN; else: raise CircuitOpenError(). try: result = await func(); if self.state == HALF_OPEN: self.state = CLOSED; self.failures = []; return result; except Exception as e: self.failures.append (time.time()); recent = [f for f in self.failures if time.time() - f < 60]; if len (recent) >= 5: self.state = OPEN; self.opened_at = time.time(); raise. Why better than retry: Retry: Each request tries N times (amplifies load on failing service). Circuit breaker: Fails fast when OPEN (doesn't waste time/resources). Example: DB down, 1000 requests: Retry: 1000 × 3 attempts = 3000 DB hits (overload!). Circuit breaker: 5 hits (to detect failure), rest rejected immediately (protects DB). Benefits: Prevents cascading failures, gives service time to recover, reduces load during outage.",
    keyPoints: [
      'States: CLOSED (normal), OPEN (failing, fast-fail), HALF_OPEN (testing recovery)',
      'Threshold: 5 failures in 60s window triggers OPEN, tracks timestamps',
      'OPEN behavior: rejects requests for 30s (fail fast), no DB load',
      'Recovery: HALF_OPEN allows one test query, success → CLOSED (resume), failure → OPEN (stay closed)',
      'Better than retry: prevents cascading failures, fails fast, protects failing service from overload',
    ],
  },
];
