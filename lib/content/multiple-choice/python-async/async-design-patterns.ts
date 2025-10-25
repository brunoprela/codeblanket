import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncDesignPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'adp-mc-1',
    question: 'What is the main benefit of the Producer-Consumer pattern?',
    options: [
      'Makes code faster',
      'Decouples production and consumption, allowing different rates',
      'Uses less memory',
      'Simplifies code',
    ],
    correctAnswer: 1,
    explanation:
      'Producer-Consumer pattern decouples production and consumption using a queue: Decoupling: Producer and consumer run independently. Different rates: Producer fast, consumer slow (or vice versa). Buffer: Queue absorbs rate differences. Example: Fast producer (1000 items/sec), slow consumer (100 items/sec). Without queue: Producer must wait (waste). With queue: Producer fills queue, consumer processes at own pace. Benefits: Flexibility: Change producer/consumer independently. Scalability: Add more consumers if needed. Backpressure: Bounded queue prevents memory overflow. Implementation: queue = asyncio.Queue (maxsize=1000); await queue.put (item) # Producer; item = await queue.get() # Consumer. Use when: Different processing rates, Need buffering, Want to decouple components.',
  },
  {
    id: 'adp-mc-2',
    question: 'What is the purpose of a Circuit Breaker pattern?',
    options: [
      'To break circuits',
      'To fail fast and prevent cascading failures when a service is down',
      'To improve performance',
      'To handle retries',
    ],
    correctAnswer: 1,
    explanation:
      "Circuit Breaker prevents cascading failures by failing fast when service is down: States: CLOSED (normal), OPEN (failing, reject), HALF_OPEN (testing). Behavior: Track failures. After threshold (e.g. 5): Open circuit (reject requests). After timeout (e.g. 60s): Half-open (test one request). If success: Close. If fail: Re-open. Benefits: Fail fast (don't wait for timeout). Prevent overload (don't hammer failing service). Quick recovery (detect when service recovers). Example: Database down. Without: 1000 requests all wait 30s timeout = waste resources. With: After 5 failures, remaining 995 fail immediately. Use when: External dependencies (API, database, cache), Want fast failure detection, Need to prevent cascading failures.",
  },
  {
    id: 'adp-mc-3',
    question: 'Why use exponential backoff in retry logic?',
    options: [
      'It is faster',
      'To avoid overwhelming the service and give it time to recover',
      'It uses less memory',
      'It is simpler',
    ],
    correctAnswer: 1,
    explanation:
      'Exponential backoff increases delay between retries to avoid overwhelming failing service: Linear retry: 1s, 1s, 1s (keeps hammering). Exponential: 1s, 2s, 4s, 8s (gives time to recover). Why it works: Service might be overloaded (retrying immediately makes it worse). Exponential gives increasing recovery time. Prevents thundering herd (many clients retrying simultaneously). Implementation: delay = initial_delay; for attempt in range (max_attempts): try: return await func(); except: await asyncio.sleep (delay); delay *= backoff_factor. Add jitter: delay * (0.5 + random.random()) prevents synchronized retries. Benefits: Better recovery (service has time). Reduced load (fewer retries). Prevents cascading failures. Use for: Network operations, API calls, transient failures.',
  },
  {
    id: 'adp-mc-4',
    question:
      'What is the difference between Worker Pool and asyncio.gather()?',
    options: [
      'They are identical',
      'Worker Pool limits concurrency with fixed workers and queue; gather() runs all tasks immediately',
      'gather() is faster',
      'Worker Pool is only for threads',
    ],
    correctAnswer: 1,
    explanation:
      'Worker Pool vs asyncio.gather(): Worker Pool: Fixed number of workers (e.g. 50). Tasks queued if all workers busy. Controlled concurrency. Example: WorkerPool(50); for task in 10000_tasks: await pool.submit (task). Only 50 concurrent at a time. asyncio.gather(): All tasks start immediately. Unlimited concurrency. Example: await gather(*[task() for _ in range(10000)]). All 10000 run concurrently! When to use: Worker Pool: Need concurrency limit (API rate limit, resource limit). Many tasks (1000+). Want backpressure (queue). gather(): Few tasks (<100). No concurrency limit. Want simplicity. Resource impact: gather(10000 tasks): 10000 concurrent connections (might exhaust resources). WorkerPool(50): Max 50 connections (controlled). Recommendation: Use Worker Pool for production (controlled), gather() for simple cases.',
  },
  {
    id: 'adp-mc-5',
    question: 'In a Pipeline pattern, why have separate queues between stages?',
    options: [
      'It uses less memory',
      'To allow parallel execution of different stages and prevent bottlenecks',
      'It is required by asyncio',
      'To simplify code',
    ],
    correctAnswer: 1,
    explanation:
      "Separate queues enable parallel stage execution: Single queue (Producer-Consumer): Stage 1 → Queue → Stage 2 → Stage 3. Bottleneck: All stages wait for slowest. Multiple queues (Pipeline): Stage 1 → Q1 → Stage 2 → Q2 → Stage 3 → Q3. Parallel: All stages run simultaneously. Benefits: Higher throughput: Stage 1 processes item N+1 while Stage 2 processes item N. No bottleneck: Fast stages don't wait for slow stages. Buffering: Each queue buffers between stages. Example: Read (100ms), Parse (50ms), Write (200ms). Single queue: 350ms per item (2.8 items/sec). Pipeline: 200ms per item (5 items/sec) - 2× faster! Trade-off: More memory (N queues × queue size). More complex (manage multiple queues). Use when: Multi-stage processing, Stages have different speeds, Throughput is critical.",
  },
];
