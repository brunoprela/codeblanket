import { MultipleChoiceQuestion } from '@/lib/types';

export const asyncFileIOMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'afio-mc-1',
    question:
      'Why is async for line in file more memory-efficient than await file.read()?',
    options: [
      'async for is faster',
      'async for reads one line at a time, read() loads entire file in memory',
      'async for compresses the data',
      'async for caches lines',
    ],
    correctAnswer: 1,
    explanation:
      'async for line in file is a streaming operation that reads one line at a time. Memory usage: Size of current line (typically <1KB). await file.read() loads entire file into memory at once. Memory usage: Entire file size. Example: 1GB file: async for = ~1KB memory, read() = 1GB memory (1,000,000× more!). Critical for large files (logs, datasets). Enables processing files larger than available RAM. Use async for for large files, read() only for small files (<10MB) where you need full contents.',
  },
  {
    id: 'afio-mc-2',
    question:
      'What is the primary benefit of using aiofiles over regular open()?',
    options: [
      'aiofiles is faster for all operations',
      'aiofiles allows concurrent file operations without blocking the event loop',
      'aiofiles automatically compresses files',
      'aiofiles only works with text files',
    ],
    correctAnswer: 1,
    explanation:
      "aiofiles provides async file operations that don't block the event loop, allowing concurrent file I/O with other async operations. Regular open() is blocking: While reading file, event loop frozen (no other tasks run). aiofiles async: While waiting for file I/O, event loop processes other tasks. Benefits: Multiple files concurrently (await gather(*[read_file (f) for f in files])). Mix file I/O with network I/O. Don't freeze event loop during I/O. When it helps: Multiple files, mixed I/O (file + network). When it doesn't: Single file sequentially, pure CPU-bound processing.",
  },
  {
    id: 'afio-mc-3',
    question:
      'What happens if you open too many files concurrently without a Semaphore?',
    options: [
      'Files read faster',
      'OSError: [Errno 24] Too many open files',
      'Files automatically queue',
      'Nothing, unlimited files allowed',
    ],
    correctAnswer: 1,
    explanation:
      'Operating systems limit file descriptors per process (typically 1024 on Linux/Mac, 512 on Windows). Each open file uses one file descriptor. Opening 1000 files simultaneously: 1000 fds > 1024 limit → OSError: [Errno 24] Too many open files. Solution: Limit concurrent opens with Semaphore: sem = Semaphore(10); async with sem: async with aiofiles.open (file). This ensures max 10 files open at once (well under limit). Example: 1000 files with Semaphore(10): 1000 / 10 = 100 batches, max 10 open at a time, stays under OS limit.',
  },
  {
    id: 'afio-mc-4',
    question:
      'When should you use async with aiofiles.open() instead of just await aiofiles.open()?',
    options: [
      'Always use async with for automatic cleanup',
      'They are identical',
      'async with is slower',
      'async with only works for text files',
    ],
    correctAnswer: 0,
    explanation:
      'Always use async with aiofiles.open() for automatic file closure (exception-safe). async with: async with aiofiles.open("file.txt") as f: content = await f.read() automatically closes file when exiting block (even on exception). Without: f = await aiofiles.open("file.txt"); content = await f.read(); await f.close() risks not closing file if exception occurs before close(). Resource leak: Unclosed files waste file descriptors, eventually hit OS limit. Best practice: Always use async with for files (like regular Python with statement). Exception-safe, guaranteed cleanup.',
  },
  {
    id: 'afio-mc-5',
    question: 'What is the best way to process a 10GB log file with aiofiles?',
    options: [
      'await file.read() to load it all',
      'async for line in file to stream line-by-line',
      'Read in 1GB chunks',
      'Use regular open() instead',
    ],
    correctAnswer: 1,
    explanation:
      'async for line in file streams the file line-by-line, keeping only current line in memory. For 10GB file: async for line = ~1KB memory (one line), read() = 10GB memory (entire file, likely crashes). Streaming advantages: Constant memory (one line), can process files larger than RAM, start processing immediately. Pattern: async with aiofiles.open("huge.log") as f: async for line in f: process (line). Processes 10GB file with <1MB memory. Alternative for binary: Read in chunks (8KB-64KB), but line-by-line better for text. Never use read() for large files (>100MB).',
  },
];
