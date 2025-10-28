export default [
  {
    id: 'cp-m1-s2-q1',
    section: 'Why C++ for Competitive Programming',
    question:
      'What is the primary reason C++ is preferred over Python for competitive programming?',
    options: [
      'C++ has better libraries for algorithms',
      'C++ executes significantly faster (10-100x) than Python',
      'C++ is easier to learn than Python',
      'C++ has more readable syntax than Python',
    ],
    correctAnswer: 1,
    explanation:
      'C++ is 10-100x faster than Python due to being compiled to machine code vs interpreted. This speed difference is crucial in competitive programming where time limits are strict. A solution that runs in 0.5s in C++ might take 5-50s in Python, causing TLE (Time Limit Exceeded). While Python has great libraries and readable syntax, raw execution speed is paramount for CP. Example: sorting 10⁶ elements takes ~0.1s in C++ but 1-2s in Python. For tight time limits (1-2 seconds), this difference is critical.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s2-q2',
    section: 'Why C++ for Competitive Programming',
    question:
      'Which C++ feature is particularly useful for optimizing tight loops in competitive programming?',
    options: [
      'Object-oriented programming with classes',
      'Exception handling with try-catch blocks',
      'Inline functions and compiler optimizations',
      'Virtual functions and polymorphism',
    ],
    correctAnswer: 2,
    explanation:
      'Inline functions and compiler optimizations (-O2 flag) are crucial for tight loops. The compiler can eliminate function call overhead by inlining small functions, unroll loops, and perform other optimizations. In CP, we use compiler flag `-O2` which enables aggressive optimizations. Code like `#define rep(i,n) for(int i=0;i<(n);i++)` benefits from inlining. OOP features like classes and virtual functions actually add overhead and are rarely used in CP. Exception handling also adds overhead and is avoided in performance-critical code.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s2-q3',
    section: 'Why C++ for Competitive Programming',
    question:
      "What is the advantage of C++'s STL (Standard Template Library) in competitive programming?",
    options: [
      'It provides GUI components for building interfaces',
      'It offers optimized, battle-tested data structures and algorithms',
      'It includes built-in machine learning algorithms',
      'It provides automatic memory management like garbage collection',
    ],
    correctAnswer: 1,
    explanation:
      'The STL provides highly optimized, well-tested data structures (vector, set, map, queue, etc.) and algorithms (sort, binary_search, etc.). These implementations are fast, debugged, and reliable - crucial when every second counts in a contest. Example: `std::sort` uses introsort (quicksort + heapsort + insertion sort) optimized over decades. Implementing your own sorting would waste time and likely be slower. STL also has priority_queue, unordered_map with fast hash tables, and more. No GUI components, ML algorithms, or garbage collection (C++ uses manual memory management).',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s2-q4',
    section: 'Why C++ for Competitive Programming',
    question:
      "In competitive programming, why do we avoid using endl and prefer '\\n' instead?",
    options: [
      'endl is deprecated in modern C++',
      "endl flushes the output buffer, making it slower than '\\n'",
      "'\\n' produces more readable output",
      "endl doesn't work with fast I/O optimization",
    ],
    correctAnswer: 1,
    explanation:
      "`endl` flushes the output buffer after printing a newline, which is slow for large outputs. `'\\n'` just adds a newline without flushing, making it much faster. When printing thousands of lines, the difference can be seconds vs milliseconds. Example: printing 10⁶ lines with endl might take 2-3 seconds, while '\\n' takes 0.1s. Combined with fast I/O (`ios_base::sync_with_stdio(false)`), '\\n' is the standard practice. Code: `cout << result << '\\n';` not `cout << result << endl;`. The buffer will flush when the program ends anyway.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s2-q5',
    section: 'Why C++ for Competitive Programming',
    question:
      'What makes C++ particularly suited for implementing complex data structures like segment trees?',
    options: [
      'C++ has built-in segment tree classes',
      'C++ allows low-level memory control and pointer manipulation',
      'C++ automatically optimizes tree structures',
      'C++ prevents all memory-related bugs',
    ],
    correctAnswer: 1,
    explanation:
      "C++ provides low-level control over memory layout and direct pointer manipulation, crucial for efficient custom data structures. For segment trees, we can use arrays with index calculations (node at i has children at 2i and 2i+1) for cache-friendly access. We control exactly how memory is laid out and accessed. Example: `int tree[4*MAXN];` gives us direct control. Languages with automatic memory management add overhead. C++ doesn't have built-in segment trees (we implement them), doesn't auto-optimize trees, and doesn't prevent memory bugs (programmer's responsibility). The tradeoff: more control but more responsibility.",
    difficulty: 'intermediate',
  },
] as const;
