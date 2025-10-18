/**
 * Quiz questions for File Handling and I/O section
 */

export const filehandlingQuiz = [
  {
    id: 'pi-filehandling-q-1',
    question:
      'Explain the difference between text mode and binary mode when working with files. When should you use each?',
    hint: 'Think about encoding, data types, and what kinds of files you might read.',
    sampleAnswer:
      'Text mode (default) reads files as strings and handles encoding/decoding automatically (usually UTF-8). Use it for .txt, .py, .csv, .json files. Binary mode (rb/wb) reads files as bytes without encoding - use it for images, videos, executables, or when you need exact byte-level control. Binary mode is crucial when file encoding is unknown or for non-text data. Opening a binary file in text mode can corrupt data or raise encoding errors.',
    keyPoints: [
      'Text mode: strings, automatic encoding/decoding',
      'Binary mode: bytes, no encoding',
      'Use text for human-readable files',
      'Use binary for images, executables, unknown encoding',
    ],
  },
  {
    id: 'pi-filehandling-q-2',
    question:
      'Why is using "with" statement crucial for file operations? What happens if you forget to close a file?',
    hint: 'Consider resource management, exceptions, and OS limitations.',
    sampleAnswer:
      'The "with" statement (context manager) automatically closes files even if exceptions occur, preventing resource leaks. Without it, files might stay open if your code crashes, potentially causing: 1) Resource exhaustion (OS limits on open files), 2) File locking issues preventing other processes from accessing the file, 3) Data not being flushed to disk (buffered writes), 4) Memory leaks. Always use "with" - it\'s the Pythonic way and ensures proper cleanup.',
    keyPoints: [
      'Automatically closes files, even on exceptions',
      'Prevents resource leaks and file locking',
      'Ensures buffered data is flushed to disk',
      'Context manager protocol (__enter__/__exit__)',
    ],
  },
  {
    id: 'pi-filehandling-q-3',
    question:
      'What are the advantages of using pathlib over os.path? Should you still learn os.path?',
    hint: 'Consider API design, readability, and cross-platform compatibility.',
    sampleAnswer:
      'pathlib (Python 3.4+) provides object-oriented path manipulation with cleaner syntax: path / "subdir" / "file.txt" instead of os.path.join(). It has convenient methods like .read_text(), .write_text(), .glob(), and works seamlessly across platforms. However, os.path is still useful for: 1) Legacy code, 2) Some specific operations not in pathlib, 3) When you need string paths for compatibility. Learn both - pathlib for new code, os.path to understand existing code.',
    keyPoints: [
      'pathlib: object-oriented, cleaner syntax',
      'Supports / operator for path joining',
      'Built-in methods for common operations',
      'os.path still needed for legacy compatibility',
    ],
  },
];
