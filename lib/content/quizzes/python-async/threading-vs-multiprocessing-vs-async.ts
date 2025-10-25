export const threadingVsMultiprocessingVsAsyncQuiz = [
  {
    id: 'tvmva-q-1',
    question:
      'You need to process 10,000 images (resize, filter, compress). Each image takes 200ms CPU time. Compare: (1) Sequential, (2) Threading (10 threads), (3) Multiprocessing (8 processes), (4) Async. Calculate time for each. Which is best and why? Explain role of GIL.',
    sampleAnswer:
      "Image processing comparison: Sequential: 10,000 × 200ms = 2,000 seconds (33 minutes). Threading (10 threads): Due to GIL, only 1 thread executes Python bytecode at a time. CPU-bound work doesn't release GIL. Result: ~2,000 seconds (no speedup, possibly slower due to context switching). Multiprocessing (8 processes): True parallelism, each process has own GIL. Speedup = 8× (number of processes). Result: 2,000 / 8 = 250 seconds (4 minutes). Async: Single-threaded, no parallelism. CPU work blocks event loop. Result: ~2,000 seconds (no benefit). Winner: Multiprocessing (8 processes) = 250 seconds. Why: Image processing is CPU-bound. GIL prevents threading parallelism: Only 1 thread executes at a time. Multiprocessing bypasses GIL: Each process has independent interpreter and GIL. True parallelism achieved. Speedup proportional to CPU cores. Threading only helps when GIL releases (I/O, sleep, some C extensions). CPU-bound work never releases GIL → no speedup.",
    keyPoints: [
      'Sequential: 2,000s (33 min), baseline no concurrency',
      'Threading: ~2,000s (no speedup), GIL prevents parallel CPU execution',
      'Multiprocessing: 250s (4 min), 8× speedup, bypasses GIL with separate processes',
      'Async: ~2,000s (no benefit), single-threaded, CPU blocks event loop',
      'GIL impact: Prevents threading parallelism for CPU work, only releases during I/O/sleep',
    ],
  },
  {
    id: 'tvmva-q-2',
    question:
      'You need to fetch 1,000 URLs (each takes 500ms network time, 10ms processing). Compare threading (50 workers) vs async (1000 concurrent). Calculate: (1) Total time, (2) Memory usage, (3) Thread/coroutine overhead. When does async become necessary vs threading sufficient?',
    sampleAnswer:
      'Fetching 1,000 URLs: Threading (50 workers): Batches: 1,000 / 50 = 20 batches. Time per batch: 500ms (network, GIL released). Total time: 20 × 500ms = 10 seconds. Memory: 50 threads × ~8MB stack = 400MB. Thread overhead: Context switching, OS scheduling. Async (1000 concurrent): All requests concurrent (single thread). Time: max(500ms) = 500ms (all parallel). Memory: 1000 coroutines × ~1KB = 1MB. Overhead: Minimal (event loop in user space). Comparison: Time: Async 20× faster (500ms vs 10s). Memory: Async 400× less (1MB vs 400MB). Threading sufficient when: Concurrency < 100 (low memory impact). Simple I/O operations. Legacy blocking libraries. Async necessary when: Concurrency > 1000 (threading exhausts resources). Need 10K+ concurrent connections. Memory constrained. WebSocket/long-lived connections. Rule of thumb: Threading: 10-100 concurrent. Async: 100+ concurrent.',
    keyPoints: [
      'Threading (50): 10s total time, 400MB memory, 20 batches of 50 concurrent',
      'Async (1000): 500ms total time, 1MB memory, all concurrent in single thread',
      'Performance: Async 20× faster (500ms vs 10s), 400× less memory (1MB vs 400MB)',
      'Threading sufficient: <100 concurrent, simple I/O, legacy blocking libraries',
      'Async necessary: >1000 concurrent, memory constrained, WebSocket, high throughput',
    ],
  },
  {
    id: 'tvmva-q-3',
    question:
      'Design data processing pipeline: Read 100GB CSV → Parse → Transform (CPU-heavy) → Write to database. Input is I/O-bound, transform is CPU-bound, output is I/O-bound. How do you combine threading, multiprocessing, and async? Explain pipeline architecture.',
    sampleAnswer:
      'Hybrid pipeline architecture: Stage 1 - Read (I/O-bound): Use async file I/O (aiofiles). async def read_chunks(): async with aiofiles.open("data.csv") as f: async for chunk in read_in_batches (f, 10000): await input_queue.put (chunk). Single thread, handles 1000s of chunks efficiently. Stage 2 - Transform (CPU-bound): Use multiprocessing pool. with ProcessPoolExecutor (max_workers=8) as executor: while True: chunk = await input_queue.get(); future = executor.submit (transform_chunk, chunk); await output_queue.put (future). 8 processes for true parallelism. Each process transforms chunk independently. Stage 3 - Write (I/O-bound): Use async database (asyncpg). async def write_chunks(): async with db_pool.acquire() as conn: while True: result = await output_queue.get(); await conn.copy_records_to_table("output", records=result). Async handles high-throughput writes. Pipeline flow: Async read → Queue → Multiprocessing transform → Queue → Async write. Queues decouple stages (backpressure handling). Input/output async (I/O efficiency). Transform multiprocessing (CPU parallelism). Benefits: Each stage uses optimal concurrency model. Async for I/O (low overhead). Multiprocessing for CPU (true parallelism). Queues enable streaming (don\'t load 100GB in memory).',
    keyPoints: [
      'Read (I/O): Async file I/O (aiofiles), streams chunks, single thread efficient',
      'Transform (CPU): Multiprocessing (8 workers), true parallelism, bypasses GIL',
      'Write (I/O): Async database (asyncpg), high-throughput inserts, connection pooling',
      'Pipeline: Async read → Queue → Process transform → Queue → Async write',
      'Hybrid approach: Each stage uses optimal model (async I/O, multiprocessing CPU)',
    ],
  },
];
