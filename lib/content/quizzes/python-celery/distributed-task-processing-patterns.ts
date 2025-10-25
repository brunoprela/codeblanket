/**
 * Quiz questions for Distributed Task Processing Patterns section
 */

export const distributedTaskProcessingPatternsQuiz = [
  {
    id: 'q1',
    question:
      'Implement a map-reduce pattern to process 1 million records in parallel and aggregate results.',
    sampleAnswer:
      'MAP-REDUCE IMPLEMENTATION: ```python from celery import group @app.task def map_task(records): results = [] for record in records: results.append(process_record(record)) return results @app.task def reduce_task(all_results): return sum(all_results) # Map phase: Split into chunks records = range(1_000_000) chunk_size = 10_000 chunks = [records[i:i+chunk_size] for i in range(0, len(records), chunk_size)] job = group(map_task.s(chunk) for chunk in chunks) results = job.apply_async().get() # Reduce phase final = reduce_task.delay(results).get() ``` BENEFITS: 100 parallel workers, Process 1M records in minutes, Scalable pattern.',
    keyPoints: [
      'Map: Process chunks in parallel',
      'Reduce: Aggregate results',
      'Chunking: Split into manageable sizes',
      'Parallel execution',
      'Scalable',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a distributed locking pattern to ensure only one worker processes a task at a time.',
    sampleAnswer:
      'DISTRIBUTED LOCKING: ```python import redis redis_client = redis.Redis() @app.task def exclusive_task(resource_id): lock_key = f"lock:{resource_id}" lock = redis_client.lock(lock_key, timeout=300) if lock.acquire(blocking=False): try: process_resource(resource_id) finally: lock.release() else: raise Ignore() # Another worker has lock ``` BENEFITS: Only one worker processes resource, Prevents race conditions, Timeout prevents deadlocks.',
    keyPoints: [
      'Redis lock',
      'Acquire(blocking=False)',
      'Timeout prevents deadlock',
      'Release in finally',
      'Prevents race conditions',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain idempotency and implement an idempotent payment processing task.',
    sampleAnswer:
      'IDEMPOTENT PAYMENT: ```python @app.task def process_payment(order_id): # Check if already processed payment = Payment.query.filter_by(order_id=order_id).first() if payment and payment.status == "complete": return {"status": "already_processed"} # Process payment charge = stripe.charge(order.amount) # Store with unique constraint Payment.create(order_id=order_id, status="complete") return {"status": "processed"} ``` IDEMPOTENCY: Safe to call multiple times, Same result whether called 1× or 100×, Database unique constraint prevents duplicates.',
    keyPoints: [
      'Check if already processed',
      'Unique database constraint',
      'Same result if repeated',
      'Prevents duplicate charges',
      'Safe for retries',
    ],
  },
];
