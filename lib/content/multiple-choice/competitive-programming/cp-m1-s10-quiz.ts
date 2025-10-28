export default [
  {
    id: 'cp-m1-s10-q1',
    section: 'Memory Management for CP',
    question:
      'When should you use static allocation vs dynamic allocation in competitive programming?',
    options: [
      'Always use dynamic allocation for flexibility',
      'Use static allocation for fixed-size arrays known at compile time for speed',
      'Always use dynamic allocation to avoid stack overflow',
      'Use whichever is more convenient',
    ],
    correctAnswer: 1,
    explanation:
      'Static allocation (arrays like `int arr[100000]`) is faster and simpler when size is known at compile time or can be bounded by a constant. Memory is allocated on stack (or globally) with no overhead. Dynamic allocation (vector, new/delete) is necessary when size is determined at runtime. For CP: Use static global arrays for fixed max sizes: `int arr[MAXN];`. Use vectors when size varies or when need dynamic resizing. Static is faster (no allocation overhead) but limited by stack size (~8MB typically). Global arrays avoid stack overflow. Example: `int dp[1000][1000];` as global is fine, as local might overflow stack. Vectors are safer and more flexible but slightly slower.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s10-q2',
    section: 'Memory Management for CP',
    question:
      'Why are global arrays preferred over local arrays for large data structures in CP?',
    options: [
      'Global arrays are faster to access',
      'Global arrays avoid stack overflow issues',
      'Global arrays use less memory',
      'Global arrays are automatically initialized to zero',
    ],
    correctAnswer: 1,
    explanation:
      'Global arrays avoid stack overflow. The stack typically has ~8MB limit, while global memory (static storage) can handle much larger allocations (GBs). Large local arrays like `int arr[1000000];` in main() will cause stack overflow. Global arrays `int arr[1000000];` outside functions work fine. Bonus: global arrays ARE automatically zero-initialized (local arrays are not). Access speed is similar. Memory usage is the same. Best practice: declare large arrays globally: `const int MAXN = 1e5; int arr[MAXN];` outside main(). For smaller arrays (<10^6 elements), local is fine. Alternatively, use vectors which allocate on heap, avoiding stack issues.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s10-q3',
    section: 'Memory Management for CP',
    question: 'What is the advantage of std::vector over plain arrays?',
    options: [
      'Vectors are always faster than arrays',
      'Vectors handle dynamic sizing and provide bounds checking (in debug mode)',
      'Vectors use less memory than arrays',
      'Vectors automatically sort their elements',
    ],
    correctAnswer: 1,
    explanation:
      "Vectors provide dynamic sizing (can grow/shrink), bounds checking with at() (catches out-of-bounds errors in debug builds), and work with STL algorithms. Arrays are fixed size. Vectors are NOT always faster (slight overhead), don't use less memory (actually more due to capacity), don't auto-sort. When to use each: Arrays for fixed-size, maximum-performance scenarios. Vectors for variable size, convenience, safety. In CP: both are common. Global arrays for speed-critical with known bounds, vectors for flexibility. Example: `vector<int> adj[MAXN];` for graph adjacency lists (each node has variable number of edges). Modern C++: vectors are preferred for safety unless performance-critical.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s10-q4',
    section: 'Memory Management for CP',
    question: 'What happens if you access an array out of bounds in C++?',
    options: [
      'Compilation error',
      'Automatic array expansion',
      'Undefined behavior - may crash, corrupt data, or appear to work',
      'Exception thrown',
    ],
    correctAnswer: 2,
    explanation:
      "Array out-of-bounds access is undefined behavior in C++. It might: (1) Crash (segmentation fault), (2) Corrupt other data (silent bugs), (3) Appear to work (accessing nearby memory). No compilation error (compiler doesn't check bounds). No automatic expansion. No exception thrown (unless using vector::at()). This is a major source of bugs in CP! Prevention: (1) Use correct loop bounds (`i < n` not `i <= n`), (2) Array size = max + 1 for safety, (3) Use vector::at() in debug for bounds checking: `v.at(i)` throws exception if i out of bounds (but slower than v[i]). Always double-check array indices. One common bug: 1-indexed loops with 0-indexed arrays.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s10-q5',
    section: 'Memory Management for CP',
    question:
      'Why is memory locality important for performance in competitive programming?',
    options: [
      'It reduces memory usage',
      'It improves cache performance, making programs faster',
      'It prevents memory leaks',
      'It makes code easier to read',
    ],
    correctAnswer: 1,
    explanation:
      'Memory locality improves cache performance. Modern CPUs load data in cache lines (64-128 bytes). Accessing nearby memory is much faster (cache hit) than random access (cache miss). In CP: (1) Arrays have better locality than linked structures (vector better than list), (2) Iterate arrays sequentially for speed, (3) Segment trees as arrays (not pointers) for cache-friendly access. Example: iterating `arr[i]` sequentially is faster than jumping around. This is why array-based structures dominate in CP. Vector internally is an array (good locality). List is linked nodes (poor locality). For segment trees: `tree[2*i]` and `tree[2*i+1]` are nearby in memory (good). Pointer-based trees scatter nodes (bad). Practical impact: can be 2-5x faster with good locality.',
    difficulty: 'intermediate',
  },
] as const;
