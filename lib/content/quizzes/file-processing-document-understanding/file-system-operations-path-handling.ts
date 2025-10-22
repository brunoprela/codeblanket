/**
 * Quiz questions for File System Operations & Path Handling section
 */

export const filesystemoperationspathhandlingQuiz = [
    {
        id: 'fpdu-fs-ops-q-1',
        question:
            'Explain the importance of atomic file operations when building production LLM applications like Cursor. How would you implement an atomic file write operation, and why is it critical for code editors?',
        hint: 'Think about what happens if a write operation is interrupted mid-way, and how code editors prevent data corruption.',
        sampleAnswer:
            'Atomic file operations are critical for code editors because they prevent partial writes that could corrupt files. If a process crashes or system fails during a write, you could end up with a half-written file. To implement atomic writes: (1) Write content to a temporary file in the same directory, (2) Use fsync to ensure data is written to disk, (3) Rename/move the temp file to the target (rename is atomic on most systems). This ensures you either get the complete new file or the old file remains unchanged - never a partial write. Cursor likely uses this pattern to ensure code files are never corrupted, even if the editor crashes mid-save.',
        keyPoints: [
            'Atomic operations prevent partial writes and file corruption',
            'Pattern: write to temp file, fsync, then atomic rename',
            'Critical for code editors where data loss is unacceptable',
            'Rename is atomic on POSIX systems (single syscall)',
            'Always create backup before modifying existing files',
        ],
    },
    {
        id: 'fpdu-fs-ops-q-2',
        question:
            'When processing large files for LLM applications (e.g., analyzing a 500MB log file), what strategies would you use to avoid memory issues? Compare different approaches with code examples.',
        hint: 'Consider loading the entire file into memory versus streaming approaches, and memory limits.',
        sampleAnswer:
            'For large files, avoid loading the entire content into memory with read_text() or read(). Instead, use streaming approaches: (1) Line-by-line iteration using "for line in file:" which reads one line at a time, (2) Chunk reading with read(chunk_size) in a loop, (3) Memory-mapped files with mmap for random access without loading into RAM. For LLM applications, you might process files in chunks, send each chunk to the LLM separately, and aggregate results. For a 500MB log file, reading line-by-line uses constant memory (~1 line worth) versus 500MB if loaded all at once. Use generators to process data lazily without materializing everything in memory.',
        keyPoints: [
            'Never use read_text() or read() for large files',
            'Stream line-by-line: "for line in file:" uses constant memory',
            'Chunk reading: read(chunk_size) for binary or text chunks',
            'Memory-mapped files (mmap) for random access patterns',
            'Process and aggregate results incrementally',
        ],
    },
    {
        id: 'fpdu-fs-ops-q-3',
        question:
            'How would you implement a file watcher system for a Cursor-like code editor that tracks changes to files in a project? What are the challenges and how would you handle edge cases like rapid successive edits or external file modifications?',
        hint: 'Consider polling versus event-driven approaches, debouncing, and handling concurrent modifications.',
        sampleAnswer:
            'For a production file watcher: (1) Use watchdog library for cross-platform file system events rather than polling, (2) Implement debouncing to handle rapid successive edits - wait for a quiet period (e.g., 500ms) before processing, (3) Track file modification times to detect external changes, (4) Use a queue to serialize file processing events, (5) Handle race conditions where file might be deleted before processing. For external modifications, compare file hashes or modification times. Cursor likely uses a combination of OS-level file system events (FSEvents on macOS, inotify on Linux) with debouncing to avoid re-analyzing files too frequently. Also need to ignore temporary files (.tmp, .swp) and certain directories (.git, node_modules).',
        keyPoints: [
            'Use watchdog library for cross-platform file system events',
            'Implement debouncing to avoid processing rapid successive changes',
            'Queue events to serialize processing and avoid race conditions',
            'Handle external modifications by checking timestamps/hashes',
            'Ignore temporary files and version control directories',
            'Test edge cases: rapid edits, deletes, renames, permission changes',
        ],
    },
];

