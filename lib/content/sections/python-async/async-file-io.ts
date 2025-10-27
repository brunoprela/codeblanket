export const asyncFileIO = {
  title: 'Async File I/O',
  id: 'async-file-io',
  content: `
# Async File I/O

## Introduction

While Python\'s built-in \`open()\` is blocking, **aiofiles** provides async file operations that don't freeze the event loop. This is critical for applications that handle many files concurrently or mix file I/O with network operations.

### Why Async File I/O

\`\`\`python
"""
Blocking vs Async File Operations
"""

import asyncio
import aiofiles
import time

# Blocking: Sequential file reads
def read_files_blocking (files):
    start = time.time()
    contents = []
    for file in files:
        with open (file) as f:
            contents.append (f.read())
    elapsed = time.time() - start
    print(f"Blocking: {elapsed:.2f}s")
    return contents

# Async: Concurrent file reads
async def read_files_async (files):
    start = time.time()
    async def read_file (file):
        async with aiofiles.open (file) as f:
            return await f.read()

    contents = await asyncio.gather(*[read_file (f) for f in files])
    elapsed = time.time() - start
    print(f"Async: {elapsed:.2f}s")
    return contents

files = ['file1.txt', 'file2.txt', 'file3.txt']
# Blocking: ~0.3s (sequential)
# Async: ~0.1s (concurrent) - 3× faster!
\`\`\`

By the end of this section, you'll master:
- aiofiles for async file operations
- Reading and writing files concurrently
- Streaming large files
- Directory operations
- File watching patterns
- Production best practices

---

## Basic Async File Operations

### Reading Files

\`\`\`python
"""
Reading Files with aiofiles
"""

import asyncio
import aiofiles

async def read_examples():
    # Read entire file
    async with aiofiles.open('data.txt', 'r') as f:
        content = await f.read()
        print(f"Content: {len (content)} characters")

    # Read line by line
    async with aiofiles.open('data.txt', 'r') as f:
        async for line in f:
            print(f"Line: {line.strip()}")

    # Read specific number of bytes
    async with aiofiles.open('data.txt', 'r') as f:
        chunk = await f.read(1024)  # Read 1KB
        print(f"Chunk: {len (chunk)} bytes")

    # Read all lines into list
    async with aiofiles.open('data.txt', 'r') as f:
        lines = await f.readlines()
        print(f"Total lines: {len (lines)}")

asyncio.run (read_examples())
\`\`\`

### Writing Files

\`\`\`python
"""
Writing Files with aiofiles
"""

import asyncio
import aiofiles

async def write_examples():
    # Write string to file
    async with aiofiles.open('output.txt', 'w') as f:
        await f.write("Hello, async world!\\n")

    # Append to file
    async with aiofiles.open('output.txt', 'a') as f:
        await f.write("Appended line\\n")

    # Write multiple lines
    lines = ['Line 1\\n', 'Line 2\\n', 'Line 3\\n']
    async with aiofiles.open('output.txt', 'w') as f:
        await f.writelines (lines)

    # Write binary data
    data = b'\\x00\\x01\\x02\\x03'
    async with aiofiles.open('binary.dat', 'wb') as f:
        await f.write (data)

asyncio.run (write_examples())
\`\`\`

---

## Concurrent File Operations

### Reading Multiple Files

\`\`\`python
"""
Read Multiple Files Concurrently
"""

import asyncio
import aiofiles

async def read_file (filename):
    """Read a single file"""
    async with aiofiles.open (filename, 'r') as f:
        content = await f.read()
        return {'file': filename, 'size': len (content), 'content': content}

async def read_many_files (filenames):
    """Read all files concurrently"""
    results = await asyncio.gather(*[read_file (f) for f in filenames])
    return results

async def main():
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    results = await read_many_files (files)

    for result in results:
        print(f"{result['file']}: {result['size']} bytes")

asyncio.run (main())

# All files read concurrently!
\`\`\`

### Processing Files in Parallel

\`\`\`python
"""
Read, Process, Write Pipeline
"""

import asyncio
import aiofiles

async def process_file (input_file, output_file):
    """Read, process, and write file"""
    # Read
    async with aiofiles.open (input_file, 'r') as f:
        content = await f.read()

    # Process (e.g., uppercase)
    processed = content.upper()

    # Write
    async with aiofiles.open (output_file, 'w') as f:
        await f.write (processed)

    print(f"Processed {input_file} -> {output_file}")

async def process_many_files (file_pairs):
    """Process multiple files concurrently"""
    await asyncio.gather(*[
        process_file (input_f, output_f)
        for input_f, output_f in file_pairs
    ])

async def main():
    pairs = [
        ('input1.txt', 'output1.txt'),
        ('input2.txt', 'output2.txt'),
        ('input3.txt', 'output3.txt'),
    ]
    await process_many_files (pairs)

asyncio.run (main())
\`\`\`

---

## Streaming Large Files

### Line-by-Line Processing

\`\`\`python
"""
Stream Large File Line by Line
Memory Efficient for Huge Files
"""

import asyncio
import aiofiles

async def process_large_file (filename):
    """Process file line by line (streaming)"""
    line_count = 0
    total_size = 0

    async with aiofiles.open (filename, 'r') as f:
        async for line in f:
            # Process each line
            line_count += 1
            total_size += len (line)

            # Example: Filter lines
            if 'ERROR' in line:
                print(f"Error found: {line.strip()}")

    print(f"Processed {line_count} lines, {total_size} bytes")

# Memory usage: Only current line in memory
# Can process multi-GB files efficiently!

asyncio.run (process_large_file('huge_log.txt'))
\`\`\`

### Chunked Reading

\`\`\`python
"""
Read Large File in Chunks
"""

import asyncio
import aiofiles

async def read_in_chunks (filename, chunk_size=8192):
    """Read file in fixed-size chunks"""
    async with aiofiles.open (filename, 'rb') as f:
        while True:
            chunk = await f.read (chunk_size)
            if not chunk:
                break

            # Process chunk
            yield chunk

async def process_large_binary_file (filename):
    """Process large binary file"""
    total_bytes = 0

    async for chunk in read_in_chunks (filename):
        total_bytes += len (chunk)
        # Process chunk (e.g., hash, compress, transmit)

    print(f"Processed {total_bytes:,} bytes")

asyncio.run (process_large_binary_file('large_file.bin'))
\`\`\`

---

## Directory Operations

### Async Directory Listing

\`\`\`python
"""
Async Directory Operations with aiofiles.os
"""

import asyncio
import aiofiles.os

async def list_directory (path):
    """List directory contents"""
    entries = await aiofiles.os.listdir (path)
    print(f"Found {len (entries)} entries in {path}")

    for entry in entries:
        full_path = f"{path}/{entry}"

        # Check if file or directory
        is_file = await aiofiles.os.path.isfile (full_path)
        is_dir = await aiofiles.os.path.isdir (full_path)

        # Get file size
        if is_file:
            size = await aiofiles.os.path.getsize (full_path)
            print(f"  File: {entry} ({size:,} bytes)")
        elif is_dir:
            print(f"  Dir:  {entry}/")

asyncio.run (list_directory('.'))
\`\`\`

### File Metadata

\`\`\`python
"""
Get File Metadata Asynchronously
"""

import asyncio
import aiofiles.os

async def get_file_info (filename):
    """Get file metadata"""
    # Check if exists
    exists = await aiofiles.os.path.exists (filename)
    if not exists:
        print(f"{filename} does not exist")
        return

    # Get stats
    stat = await aiofiles.os.stat (filename)

    print(f"File: {filename}")
    print(f"  Size: {stat.st_size:,} bytes")
    print(f"  Modified: {stat.st_mtime}")
    print(f"  Permissions: {oct (stat.st_mode)}")

asyncio.run (get_file_info('data.txt'))
\`\`\`

---

## Production Patterns

### Batch File Processing

\`\`\`python
"""
Production Pattern: Batch File Processing
"""

import asyncio
import aiofiles
from typing import List

class FileProcessor:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore (max_concurrent)

    async def process_file (self, filename: str) -> dict:
        """Process single file"""
        async with self.semaphore:  # Limit concurrency
            try:
                async with aiofiles.open (filename, 'r') as f:
                    content = await f.read()

                # Process content
                lines = content.split('\\n')

                return {
                    'filename': filename,
                    'lines': len (lines),
                    'size': len (content),
                    'status': 'success'
                }

            except Exception as e:
                return {
                    'filename': filename,
                    'error': str (e),
                    'status': 'failed'
                }

    async def process_batch (self, filenames: List[str]):
        """Process batch of files"""
        results = await asyncio.gather(*[
            self.process_file (f) for f in filenames
        ])

        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        failed = len (results) - success
        total_lines = sum (r.get('lines', 0) for r in results)

        print(f"Processed {len (results)} files:")
        print(f"  Success: {success}")
        print(f"  Failed: {failed}")
        print(f"  Total lines: {total_lines:,}")

        return results

async def main():
    processor = FileProcessor (max_concurrent=10)
    files = [f'file{i}.txt' for i in range(100)]
    results = await processor.process_batch (files)

asyncio.run (main())
\`\`\`

### Log File Monitor

\`\`\`python
"""
Production Pattern: Async Log Monitor
"""

import asyncio
import aiofiles

class LogMonitor:
    def __init__(self, logfile: str):
        self.logfile = logfile
        self.position = 0

    async def tail (self, callback):
        """Monitor log file for new lines"""
        # Seek to end initially
        async with aiofiles.open (self.logfile, 'r') as f:
            await f.seek(0, 2)  # Seek to end
            self.position = await f.tell()

        while True:
            async with aiofiles.open (self.logfile, 'r') as f:
                await f.seek (self.position)

                # Read new lines
                async for line in f:
                    await callback (line.strip())

                # Update position
                self.position = await f.tell()

            # Wait before checking again
            await asyncio.sleep(1)

async def process_log_line (line: str):
    """Process new log line"""
    if 'ERROR' in line:
        print(f"🔴 Error: {line}")
    elif 'WARNING' in line:
        print(f"🟡 Warning: {line}")

async def main():
    monitor = LogMonitor('app.log')
    await monitor.tail (process_log_line)

# asyncio.run (main())  # Runs forever, monitoring log
\`\`\`

---

## Best Practices

### Do: Use Async Context Managers

\`\`\`python
# ✅ Good: Automatic cleanup
async with aiofiles.open('file.txt', 'r') as f:
    content = await f.read()
# File automatically closed

# ❌ Bad: Manual cleanup (easy to forget)
f = await aiofiles.open('file.txt', 'r')
content = await f.read()
await f.close()  # Forgotten if exception!
\`\`\`

### Do: Stream Large Files

\`\`\`python
# ✅ Good: Streaming (constant memory)
async with aiofiles.open('huge.txt', 'r') as f:
    async for line in f:
        process (line)
# Memory: One line at a time

# ❌ Bad: Load everything (high memory)
async with aiofiles.open('huge.txt', 'r') as f:
    content = await f.read()  # Loads entire file!
    for line in content.split('\\n'):
        process (line)
# Memory: Entire file
\`\`\`

### Do: Limit Concurrency

\`\`\`python
# ✅ Good: Limited concurrency
semaphore = asyncio.Semaphore(10)
async def process_file (f):
    async with semaphore:
        async with aiofiles.open (f) as file:
            return await file.read()

# ❌ Bad: Unlimited concurrency
# Can open 1000s of file handles simultaneously
await asyncio.gather(*[read_file (f) for f in files])
\`\`\`

---

## Summary

### Key Concepts

1. **aiofiles**: Async file operations (don't block event loop)
2. **Streaming**: Process files line-by-line (memory efficient)
3. **Concurrency**: Read/write multiple files in parallel
4. **Context Managers**: Always use \`async with\` for cleanup
5. **Semaphores**: Limit concurrent file operations

### Performance Tips

- Use streaming for large files (>100MB)
- Limit concurrent operations (semaphore)
- Process files in batches
- Close files properly (context managers)

### Common Patterns

- Batch processing: Process many files concurrently
- Log monitoring: Tail log files for new entries
- ETL pipelines: Read, transform, write in parallel

### Next Steps

Now that you master async file I/O, we'll explore:
- Error handling in async code
- Production async patterns
- Complete application examples

**Remember**: Async file I/O enables concurrent file processing without blocking!
`,
};
