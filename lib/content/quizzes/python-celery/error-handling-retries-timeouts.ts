/**
 * Quiz questions for Error Handling, Retries & Timeouts section
 */

export const errorHandlingRetriesTimeoutsQuiz = [
  {
    id: 'q1',
    question:
      'Design a comprehensive retry strategy for an API call that experiences: rate limiting (429), temporary outages (503), and permanent errors (404). Include exponential backoff, max retries, and appropriate error handling.',
    sampleAnswer:
      'COMPREHENSIVE API RETRY STRATEGY: ```python @app.task (bind=True, max_retries=5) def call_api_with_retry (self, url, data): try: response = requests.post (url, json=data, timeout=10) if response.status_code == 200: return response.json() elif response.status_code == 429: # Rate limited retry_after = int (response.headers.get("Retry-After", 60)) raise self.retry (countdown=retry_after) elif response.status_code == 503: # Temporary outage - exponential backoff countdown = 60 * (2 ** self.request.retries) raise self.retry (countdown=min (countdown, 600)) elif response.status_code == 404: # Permanent error - don\'t retry raise PermanentError (f"Resource not found: {url}") else: response.raise_for_status() except requests.Timeout as exc: raise self.retry (exc=exc, countdown=30) except requests.ConnectionError as exc: countdown = 60 * (2 ** self.request.retries) raise self.retry (exc=exc, countdown=countdown) except PermanentError: raise # Don\'t retry except Exception as exc: if self.request.retries >= self.max_retries: send_to_dlq (url, data, str (exc)) raise ``` KEY FEATURES: Rate limiting (429): Respect Retry-After header. Temporary errors (503): Exponential backoff (60s, 120s, 240s). Permanent errors (404): Fail immediately, no retry. Timeout: Short retry (30s). Max retries: 5 attempts. Dead letter queue: Store after max retries.',
    keyPoints: [
      'Rate limiting: Respect Retry-After header',
      'Temporary errors: Exponential backoff',
      'Permanent errors: No retry',
      'Timeout handling',
      'Dead letter queue after max retries',
    ],
  },
  {
    id: 'q2',
    question:
      'Your task is timing out after 10 minutes but needs to process 1 million records (takes 2 hours). Design a solution using task chunking and checkpoints.',
    sampleAnswer:
      'CHUNKING + CHECKPOINT SOLUTION: ```python @app.task (bind=True) def process_million_records (self): total = 1_000_000 chunk_size = 10_000 chunks = total // chunk_size for i in range (chunks): chunk_start = i * chunk_size chunk_end = (i + 1) * chunk_size process_chunk.delay (chunk_start, chunk_end) @app.task (soft_time_limit=300, time_limit=330) def process_chunk (start, end): checkpoint_key = f"checkpoint:{start}" if redis.exists (checkpoint_key): return # Already processed for i in range (start, end): process_record (i) redis.setex (checkpoint_key, 86400, "done") ``` BENEFITS: Each chunk < 5 minutes (within time limit). 100 chunks process in parallel. Checkpoints prevent reprocessing. Failure: Only failed chunk retries.',
    keyPoints: [
      'Chunk large tasks',
      'Each chunk < time limit',
      'Parallel processing',
      'Checkpoints',
      'Idempotent chunks',
    ],
  },
  {
    id: 'q3',
    question:
      'Implement a task with soft time limit that saves progress before hard limit kills it.',
    sampleAnswer:
      'GRACEFUL TIMEOUT HANDLING: ```python @app.task (bind=True, soft_time_limit=270, time_limit=300) def long_task (self, data): from celery.exceptions import SoftTimeLimitExceeded processed = [] try: for item in data: processed.append (process_item (item)) except SoftTimeLimitExceeded: save_checkpoint (self.request.id, processed) logger.warning (f"Task hit soft limit, processed {len (processed)} items") raise return {"processed": len (processed)} ``` SOFT LIMIT: Raises exception (270s). HARD LIMIT: SIGKILL (300s). GRACE PERIOD: 30s to save progress. RESUME: Load checkpoint and continue.',
    keyPoints: [
      'Soft limit raises exception',
      'Hard limit kills process',
      'Grace period for cleanup',
      'Save checkpoint',
      'Resume from checkpoint',
    ],
  },
];
