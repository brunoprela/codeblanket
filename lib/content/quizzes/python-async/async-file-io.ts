export const asyncFileIOQuiz = [
  {
    id: 'afio-q-1',
    question:
      'Build a log aggregator that: (1) Monitors 100 log files concurrently for new entries, (2) Processes each line as it appears (parse, filter, enrich), (3) Writes aggregated results to output file in batches, (4) Handles log file rotation (file deleted/recreated), (5) Limits memory usage despite processing GB/day. Explain why async for line in f is memory-efficient vs await f.read().',
    sampleAnswer:
      'Log aggregator: class LogAggregator: def __init__(self): self.positions = {}; self.output_queue = asyncio.Queue(maxsize=1000). async def monitor_file(self, filename): position = self.positions.get(filename, 0); while True: try: async with aiofiles.open(filename) as f: await f.seek(position); async for line in f: await self.process_line(line); position = await f.tell(); self.positions[filename] = position; except FileNotFoundError: await asyncio.sleep(1); continue; await asyncio.sleep(0.1). async def process_line(self, line): parsed = parse_log(line); if matches_filter(parsed): enriched = await enrich(parsed); await self.output_queue.put(enriched). async def write_batch(self): batch = []; while True: try: entry = await asyncio.wait_for(self.output_queue.get(), timeout=1.0); batch.append(entry); if len(batch) >= 100: await self.flush_batch(batch); batch = []; except asyncio.TimeoutError: if batch: await self.flush_batch(batch); batch = []. Why async for efficient: async for line in f: reads one line at a time, memory = one line. await f.read(): loads entire file, memory = full file size. For 1GB file: async for = ~1KB memory. read() = 1GB memory (1M× more!).',
    keyPoints: [
      'Monitor: tail each file (seek to last position, read new lines), handle FileNotFoundError for rotation',
      'Process: async for line in f (memory-efficient), parse/filter/enrich per line, queue for batching',
      'Write: batch 100 entries, flush on timeout, prevents too many small writes',
      'Memory: async for reads one line at a time (1KB), await read() loads full file (GB)',
      'File rotation: catch FileNotFoundError, reset position, reopen after delay',
    ],
  },
  {
    id: 'afio-q-2',
    question:
      'Compare: (1) Sequential file processing (open/read/close each file), (2) Concurrent with unlimited asyncio.gather(), (3) Concurrent with Semaphore(10). For 1000 files (10MB each): estimate time, memory, file handle usage. Why can unlimited concurrency cause "too many open files" error?',
    sampleAnswer:
      'File processing comparison: Sequential: for file in files: async with aiofiles.open(file) as f: data = await f.read(); process(data). Time: 1000 files × (open 10ms + read 100ms + process 50ms) = 160 seconds. Memory: 10MB (one file at a time). File handles: 1 (reused). Unlimited concurrent: tasks = [process_file(f) for f in files]; await asyncio.gather(*tasks). Time: max(file times) = ~200ms (all concurrent). Memory: 1000 × 10MB = 10GB (all files in memory!). File handles: 1000 (all open simultaneously). Semaphore(10): sem = Semaphore(10); tasks = [process_with_limit(sem, f) for f in files]; await asyncio.gather(*tasks). Time: 1000 / 10 × 200ms = 20 seconds. Memory: 10 × 10MB = 100MB (max 10 files). File handles: 10 (max 10 open). Why "too many open files": OS limits file descriptors (typically 1024). Unlimited: 1000 files × 1 fd each = 1000 fds (hits limit!). Error: OSError: [Errno 24] Too many open files. Semaphore limits concurrent opens (stays under limit). Recommendation: Use Semaphore(10-50) for file operations. Balance: Higher concurrency (faster) vs resource usage (memory, fds).',
    keyPoints: [
      'Sequential: 160s, 10MB memory, 1 file handle, safe but slow',
      'Unlimited: 0.2s, 10GB memory, 1000 handles, fast but hits OS limit (1024 fds)',
      'Semaphore(10): 20s, 100MB, 10 handles, balanced (10× faster than sequential)',
      'OS limit: 1024 file descriptors, unlimited opens cause "too many open files" error',
      'Best: Semaphore(10-50) balances speed and resource usage',
    ],
  },
  {
    id: 'afio-q-3',
    question:
      "Design ETL pipeline: Read 100 CSV files → Transform data → Write to 10 output files (by category). Use async file I/O for: (1) Concurrent reading, (2) Streaming (line-by-line processing), (3) Batch writing (buffer 1000 rows per output), (4) Memory-efficient (process 100GB data with 1GB RAM). Explain when aiofiles helps vs when it doesn't matter.",
    sampleAnswer:
      'ETL pipeline: class ETLPipeline: def __init__(self): self.output_buffers = defaultdict(list); self.semaphore = asyncio.Semaphore(10). async def process_file(self, filename): async with self.semaphore: async with aiofiles.open(filename) as f: async for line in f: row = parse_csv(line); category = extract_category(row); transformed = transform(row); self.output_buffers[category].append(transformed); if len(self.output_buffers[category]) >= 1000: await self.flush_buffer(category). async def flush_buffer(self, category): buffer = self.output_buffers[category]; if not buffer: return; async with aiofiles.open(f"output_{category}.csv", "a") as f: await f.writelines([format_csv(row) for row in buffer]); self.output_buffers[category] = []. async def run(self, input_files): await asyncio.gather(*[self.process_file(f) for f in input_files]); for category in self.output_buffers: await self.flush_buffer(category). Memory efficiency: async for line: reads one line (1KB). Buffer: max 1000 rows × 10 categories = 10,000 rows (~10MB). Total: ~20MB vs 100GB (5000× less!). When aiofiles helps: Many files: Concurrent reading (100 files in parallel). Mixed I/O: File I/O + network (both non-blocking). When it doesn\'t matter: Single file: No concurrency benefit. Pure CPU: Processing time dominates I/O time. Small files: I/O time negligible.',
    keyPoints: [
      'Read: Semaphore(10) limits concurrent opens, async for streams lines (memory-efficient)',
      'Transform: per-line processing, extract category, transform row, buffer by category',
      'Write: batch 1000 rows per category, flush when full, append mode',
      'Memory: async for + batching = 20MB for 100GB data (5000× reduction)',
      "aiofiles helps: concurrent files, mixed I/O; doesn't help: single file, CPU-bound",
    ],
  },
];
