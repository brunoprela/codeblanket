/**
 * Quiz questions for String Operations section
 */

export const stringsQuiz = [
  {
    id: 'pf-strings-q-1',
    question:
      'Why are strings immutable in Python? What are the implications of this design decision?',
    hint: 'Think about dictionary keys, memory management, and thread safety.',
    sampleAnswer:
      'Strings are immutable to allow them to be hashable and used as dictionary keys and set members. Immutability enables string interning (reusing identical strings in memory) for efficiency, makes strings thread-safe without locks, and simplifies Python\'s implementation. The trade-off is that any "modification" creates a new string object. For frequent string modifications, use lists or StringIO/join() instead of concatenation to avoid creating many intermediate string objects.',
    keyPoints: [
      'Enables use as dictionary keys (hashable)',
      'Allows string interning for memory efficiency',
      'Thread-safe by default',
      'All "modifications" create new strings',
    ],
  },
  {
    id: 'pf-strings-q-2',
    question:
      'Explain the difference between str.format(), f-strings, and % formatting. Which should you use and why?',
    hint: 'Consider readability, performance, Python version requirements, and flexibility.',
    sampleAnswer:
      'Old-style % formatting (like "Hello %s" % name) is C-style but limited and less readable. str.format() (like "Hello {}".format (name)) is more powerful and readable but verbose. F-strings (like f"Hello {name}") are the modern preferred way (Python 3.6+) - they\'re most readable, fastest, and allow expressions inside braces. Use f-strings for new code unless you need Python 3.5 compatibility. str.format() is still useful when the format string comes from user input or configuration (security concern with f-strings).',
    keyPoints: [
      '% formatting: old style, less readable',
      'str.format(): more flexible, verbose',
      'f-strings: fastest, most readable (Python 3.6+)',
      'Prefer f-strings for new code',
    ],
  },
  {
    id: 'pf-strings-q-3',
    question:
      'When would you use str.join() versus string concatenation with +? What about performance considerations?',
    hint: 'Think about building strings in loops and memory allocation.',
    sampleAnswer:
      'Use str.join() when building strings from multiple parts, especially in loops. Since strings are immutable, using + creates a new string object each time, which is O(n²) for n concatenations. str.join() is O(n) as it allocates the final size once. Example: "".join (parts) is much faster than result = ""; for p in parts: result += p. However, for a small fixed number of concatenations (2-3), + is fine and more readable. F-strings are also efficient for combining a few known values.',
    keyPoints: [
      'join(): O(n), efficient for multiple strings',
      '+: O(n²) in loops due to immutability',
      'Use join() for building strings in loops',
      '+ is fine for 2-3 static concatenations',
    ],
  },
];
