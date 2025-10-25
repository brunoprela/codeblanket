export const asyncDesignPatternsQuiz = [
  {
    id: 'adp-q-1',
    question:
      'Design async web scraper: Scrape 10,000 URLs with (1) Worker pool (50 workers), (2) Rate limiter (100 req/sec), (3) Circuit breaker (5 failures → open), (4) Retry (3 attempts with backoff). Show how patterns compose. What happens when circuit opens?',
    sampleAnswer:
      'Async web scraper combining patterns: class Scraper: def __init__(self): self.pool = WorkerPool(50); self.limiter = RateLimiter(100, per=1.0); self.breaker = CircuitBreaker(threshold=5); self.stats = {"success": 0, "failed": 0, "retried": 0}. async def scrape_url(self, url): await self.limiter.acquire(); # Rate limit first; result = await retry_with_backoff(lambda: self.breaker.call(self._fetch, url), max_attempts=3). async def _fetch(self, url): async with aiohttp.ClientSession() as session: async with session.get(url, timeout=10) as resp: return await resp.text(). async def run(self, urls): await self.pool.start(); for url in urls: await self.pool.submit(self.scrape_url, url); await self.pool.queue.join(); await self.pool.shutdown(). Pattern composition: Rate limiter runs first (prevent overwhelming server). Circuit breaker wraps fetch (detect failing service). Retry wraps circuit breaker (handle transient failures). Worker pool manages concurrency (50 parallel). When circuit opens: Immediate failure (no retry). Saves resources (doesn\'t hammer failing service). After timeout (60s): Transitions to HALF_OPEN (test one request). Success → CLOSED (resume normal). Failure → OPEN (stay closed). Benefits: Rate limiter: Respects server limits (100 req/s). Circuit breaker: Fast fail during outage. Retry: Handles transient errors. Worker pool: Controls concurrency.',
    keyPoints: [
      'Worker pool (50): Controls concurrency, task queue, shutdown coordination',
      'Rate limiter: Token bucket, 100 req/sec, prevents overwhelming server',
      'Circuit breaker: 5 failures → OPEN, fail fast, HALF_OPEN test after timeout',
      'Retry: 3 attempts, exponential backoff, wraps circuit breaker call',
      'Composition: Rate limit → Circuit breaker → Retry → Worker pool, each layer adds protection',
    ],
  },
  {
    id: 'adp-q-2',
    question:
      'Compare Producer-Consumer vs Pipeline patterns for ETL: Read CSV → Parse → Validate → Transform → Write DB. Which pattern when? Implement both. Explain trade-offs: throughput, memory, complexity.',
    sampleAnswer:
      'Producer-Consumer approach: Single queue, all stages share queue. queue = asyncio.Queue(1000); async def read_produce(): async for line in read_csv(): await queue.put(line). async def process_consume(): while True: line = await queue.get(); parsed = parse(line); validated = validate(parsed); transformed = transform(validated); await write_db(transformed). Trade-offs: Throughput: Limited by slowest stage (bottleneck). Memory: Queue size × record size (1000 records). Complexity: Simple (one queue). Pipeline approach: Queue between each stage. async def pipeline(): q1, q2, q3, q4 = [asyncio.Queue(100) for _ in range(4)]; await gather(read_to_queue(q1), parse_stage(q1, q2), validate_stage(q2, q3), transform_stage(q3, q4), write_stage(q4)). Trade-offs: Throughput: Each stage runs in parallel (higher). Memory: 4 queues × 100 records (more memory). Complexity: More complex (multiple queues). When to use: Producer-Consumer: Simple ETL, stages similar speed, low memory. Pipeline: High throughput needed, stages different speeds, can afford memory. Benchmark: Producer-Consumer: 1000 records/sec (sequential bottleneck). Pipeline: 5000 records/sec (parallel stages). Recommendation: Start with Producer-Consumer (simpler). Use Pipeline if throughput insufficient.',
    keyPoints: [
      'Producer-Consumer: Single queue, simple, bottlenecked by slowest stage',
      'Pipeline: Queue per stage, parallel stages, higher throughput but more complex',
      'Throughput: Producer-Consumer ~1K rec/s, Pipeline ~5K rec/s (parallel)',
      'Memory: Producer-Consumer lower (1 queue), Pipeline higher (N queues)',
      'Choice: Producer-Consumer for simple ETL, Pipeline for high throughput needs',
    ],
  },
  {
    id: 'adp-q-3',
    question:
      'Implement Token Bucket rate limiter vs Sliding Window rate limiter. Compare: (1) Smoothness of rate control, (2) Bursty traffic handling, (3) Memory usage, (4) Implementation complexity. When to use each?',
    sampleAnswer:
      'Token Bucket: class TokenBucket: def __init__(self, rate, per): self.tokens = rate; self.rate = rate; self.per = per; self.last_update = time.time(). async def acquire(self): now = time.time(); elapsed = now - self.last_update; self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per)); if self.tokens < 1: await asyncio.sleep((1 - self.tokens) * (self.per / self.rate)); self.tokens = 0; else: self.tokens -= 1; self.last_update = now. Characteristics: Allows bursts (up to rate tokens). Smooth refill (continuous). Memory: O(1) (just counters). Sliding Window: class SlidingWindow: def __init__(self, rate, window): self.rate = rate; self.window = window; self.requests = deque(). async def acquire(self): now = time.time(); # Remove old requests; while self.requests and self.requests[0] < now - self.window: self.requests.popleft(); if len(self.requests) >= self.rate: sleep_time = self.requests[0] + self.window - now; await asyncio.sleep(sleep_time); self.requests.append(now). Characteristics: Strict rate (no bursts). Exact window (more precise). Memory: O(rate) (store timestamps). Comparison: Smoothness: Token bucket smoother (continuous refill). Sliding window stricter (exact rate). Bursts: Token bucket allows bursts (good for caching). Sliding window prevents bursts (good for APIs). Memory: Token bucket: O(1). Sliding window: O(rate). When to use: Token bucket: Smooth traffic, allow bursts, low memory. Sliding window: Strict rate, prevent bursts, can afford memory.',
    keyPoints: [
      'Token Bucket: Continuous refill, allows bursts, O(1) memory, smooth rate control',
      'Sliding Window: Strict rate, prevents bursts, O(rate) memory, exact window enforcement',
      'Smoothness: Token bucket smoother, Sliding window stricter',
      'Bursts: Token bucket allows (up to rate), Sliding window prevents',
      'Use Token Bucket for smooth traffic with bursts; Sliding Window for strict API limits',
    ],
  },
];
