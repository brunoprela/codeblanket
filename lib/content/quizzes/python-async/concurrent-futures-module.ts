export const concurrentFuturesModuleQuiz = [
  {
    id: 'cfm-q-1',
    question:
      'Design parallel data processing system using concurrent.futures: Process 10,000 CSV files (read → parse → validate → write to DB). Reading/writing are I/O-bound, validation is CPU-bound. Use ThreadPoolExecutor for I/O, ProcessPoolExecutor for CPU. Implement progress tracking, error handling, and rate limiting (max 100 DB writes/sec).',
    sampleAnswer:
      'Hybrid processing system: class DataProcessor: def __init__(self): self.thread_pool = ThreadPoolExecutor (max_workers=20); self.process_pool = ProcessPoolExecutor (max_workers=8); self.rate_limiter = Semaphore(100); self.stats = {"processed": 0, "errors": 0}. def read_file (self, filename): return pd.read_csv (filename). def validate_data (self, data): # CPU-intensive validation; return validate_rules (data). def write_to_db (self, data): with self.rate_limiter: conn.execute("INSERT INTO..."); time.sleep(1.0/100). def process_file (self, filename): try: data = self.thread_pool.submit (self.read_file, filename).result(); validated = self.process_pool.submit (self.validate_data, data).result(); self.thread_pool.submit (self.write_to_db, validated); self.stats["processed"] += 1; except Exception as e: self.stats["errors"] += 1; logger.error (f"Failed {filename}: {e}"). def process_all (self, files): with ThreadPoolExecutor(20) as tpe, ProcessPoolExecutor(8) as ppe: futures = [tpe.submit (self.process_file, f) for f in files]; for future in as_completed (futures): progress = self.stats["processed"] / len (files) * 100; print(f"Progress: {progress:.1f}%"). Key design: Read/write use ThreadPoolExecutor (I/O efficient). Validation uses ProcessPoolExecutor (CPU parallelism). Rate limiter: Semaphore(100) limits DB writes. Progress: Track completed count in as_completed loop.',
    keyPoints: [
      'I/O operations: ThreadPoolExecutor (20 workers), read/write files and database',
      'CPU validation: ProcessPoolExecutor (8 workers), true parallelism for compute-heavy validation',
      'Rate limiting: Semaphore(100) + sleep(1/100) limits DB writes to 100/sec',
      'Progress tracking: Count completed in as_completed(), calculate percentage',
      'Error handling: Try/except per file, log errors, continue processing others',
    ],
  },
  {
    id: 'cfm-q-2',
    question:
      'Compare executor.map() vs executor.submit() with as_completed(). When to use each? Implement batch API calls: map() for simple case, submit() with as_completed() for complex case with retries and timeouts.',
    sampleAnswer:
      'map() vs submit(): map() characteristics: Returns results in order (submission order). Blocks until all complete. Simple interface: executor.map (func, items). Use when: Need results in order. All tasks identical (same function). Don\'t need per-task error handling. submit() characteristics: Returns Future immediately. Can process as completed (any order). Fine control: timeout, cancel, callbacks. Use when: Need results ASAP (as_completed). Different functions per task. Per-task error handling/retries. Simple map() example: with ThreadPoolExecutor(10) as ex: results = list (ex.map (api_call, endpoints)). Complex submit() example: with ThreadPoolExecutor(10) as ex: futures = {ex.submit (api_call, ep): ep for ep in endpoints}; for future in as_completed (futures, timeout=30): ep = futures[future]; try: result = future.result (timeout=5); process (result); except TimeoutError: retry_queue.append (ep); except Exception as e: logger.error (f"{ep} failed: {e}"). Key differences: map(): ordered, blocks, simple. submit(): flexible, as_completed, complex control. Recommendation: Use map() for straightforward batch processing. Use submit() when need timeouts, retries, or progressive results.',
    keyPoints: [
      'map(): Returns results in submission order, blocks until all complete, simple interface',
      'submit(): Returns Future, use as_completed() for any order, fine control (timeout/cancel)',
      'map() use case: Batch operations, need ordered results, simple error handling',
      'submit() use case: Progressive results, per-task timeouts, retries, different functions',
      'Trade-off: map() simpler code, submit() more flexible control',
    ],
  },
  {
    id: 'cfm-q-3',
    question:
      "Explain Future object lifecycle and methods. What\'s the difference between result(), result (timeout), done(), running(), cancel()? Design task manager that tracks running futures, cancels slow tasks (>10s), and retries failed tasks (max 3 attempts).",
    sampleAnswer:
      'Future lifecycle and methods: done(): Returns True if completed (success or error). False if running/pending. running(): True if currently executing. False if pending/done. result(): Blocks until complete, returns result or raises exception. result (timeout): Waits max timeout seconds, raises TimeoutError if not done. cancel(): Attempts to cancel. Returns True if cancelled, False if already running/done. Lifecycle: Created (pending) → Running → Done (success/error/cancelled). Task manager: class TaskManager: def __init__(self): self.futures = {}; self.attempts = defaultdict (int). def submit (self, func, *args): future = executor.submit (func, *args); self.futures[future] = (func, args); self.attempts[(func, args)] = 0; return future. def monitor (self): while self.futures: for future in list (self.futures.keys()): if not future.done(): # Check timeout; if time.time() - future.start_time > 10: future.cancel(); self.retry (future); continue; try: result = future.result (timeout=0); del self.futures[future]; except Exception as e: self.retry (future, e). def retry (self, future, error=None): func, args = self.futures.pop (future); if self.attempts[(func, args)] < 3: self.attempts[(func, args)] += 1; self.submit (func, *args); else: logger.error (f"Max retries: {error}").',
    keyPoints: [
      'done(): True if completed (any state), False if pending/running',
      'result (timeout): Blocks max timeout, returns value or raises exception/TimeoutError',
      'cancel(): Attempts cancel, returns True if succeeded (only works if pending)',
      'Lifecycle: Pending → Running → Done (success/error/cancelled)',
      'Task manager: Track futures dict, monitor done(), cancel if >10s, retry failed (max 3)',
    ],
  },
];
