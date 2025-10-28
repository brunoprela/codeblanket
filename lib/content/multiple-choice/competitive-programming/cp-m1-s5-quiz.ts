export default [
  {
    id: 'cp-m1-s5-q1',
    section: 'Fast Input/Output Techniques',
    question:
      'What does ios_base::sync_with_stdio(false) do and why is it important?',
    options: [
      'It disables input/output entirely',
      'It unsynchronizes C++ streams from C streams, making cin/cout faster',
      'It enables colored output in the terminal',
      'It automatically handles file I/O',
    ],
    correctAnswer: 1,
    explanation:
      "By default, C++ streams (cin/cout) are synchronized with C streams (scanf/printf) for compatibility, adding overhead. `ios_base::sync_with_stdio(false)` disables this synchronization, making cin/cout significantly faster (often 2-5x). This is crucial for problems with large I/O. WARNING: After this, don't mix cin/cout with scanf/printf in the same program! Always pair with `cin.tie(nullptr)` for maximum speed. Code: `ios_base::sync_with_stdio(false); cin.tie(nullptr);`. Use at the start of main(). Speedup example: reading 10⁶ integers takes ~1.5s synchronized, ~0.3s unsynchronized. Critical for avoiding TLE on I/O-heavy problems.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s5-q2',
    section: 'Fast Input/Output Techniques',
    question:
      "Why should you use '\\n' instead of endl in competitive programming?",
    options: [
      'endl is not supported in C++17',
      'endl flushes the buffer after each line, making it slower',
      "'\\n' produces platform-independent newlines",
      'endl requires an extra header file',
    ],
    correctAnswer: 1,
    explanation:
      "`endl` performs two operations: (1) outputs a newline, (2) flushes the output buffer. Flushing is slow when done repeatedly. `'\\n'` only outputs a newline without flushing, making it much faster. For printing 10⁶ lines, endl can take 2-3 seconds vs 0.1s for '\\n'. The buffer flushes automatically when program ends or buffer is full, so manual flushing is unnecessary in CP. Code: `cout << result << '\\n';` NOT `cout << result << endl;`. Exception: interactive problems where you NEED to flush after each output to communicate with judge - then use `flush` explicitly. Standard practice: always use '\\n' unless explicitly need flushing.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s5-q3',
    section: 'Fast Input/Output Techniques',
    question:
      'When should you consider using scanf/printf instead of cin/cout?',
    options: [
      'Always, they are always faster',
      'Never, cin/cout with fast I/O is sufficient',
      'Only for floating-point I/O where precision control is needed',
      'Only when reading single characters',
    ],
    correctAnswer: 2,
    explanation:
      'With `ios_base::sync_with_stdio(false)`, cin/cout are typically as fast as scanf/printf for most cases. However, scanf/printf offer better control for floating-point formatting: `printf("%.6f", x)` for 6 decimal places, `scanf("%lf", &x)` for doubles. Cin/cout with fast I/O are simpler and sufficient for integers and strings. Use scanf/printf when: (1) Need precise floating-point formatting, (2) Old habits from C, (3) Specific judge quirks. Don\'t mix them with cin/cout after calling sync_with_stdio(false)! Modern practice: fast cin/cout for everything except when precise float formatting needed. Example: `printf("%.10f\\n", answer);` vs `cout << fixed << setprecision(10) << answer << \'\\n\';` (both work, printf is terser).',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s5-q4',
    section: 'Fast Input/Output Techniques',
    question: 'What is cin.tie(nullptr) and why is it used?',
    options: [
      'It ties cin to a file for input',
      'It unties cin from cout, avoiding automatic flushing',
      'It enables multi-threaded input',
      'It disables input buffering entirely',
    ],
    correctAnswer: 1,
    explanation:
      "By default, cin is tied to cout, meaning cout is automatically flushed before each cin operation. This ensures output appears before input prompts in interactive programs, but adds overhead. `cin.tie(nullptr)` removes this tie, avoiding unnecessary flushes and making I/O faster. Always pair with `ios_base::sync_with_stdio(false)` for maximum speed. Code: `ios_base::sync_with_stdio(false); cin.tie(nullptr);`. Use at start of main(). WARNING: Don't use for interactive problems where you need output to appear before reading input! For normal CP problems (read all input, compute, output all results), this is safe and fast. Speedup: reduces I/O time by 10-20% in some cases.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s5-q5',
    section: 'Fast Input/Output Techniques',
    question:
      'For reading a large array of integers (n ≤ 10⁶), which is the fastest approach?',
    options: [
      'Read one by one with cin in a loop',
      'Use cin with fast I/O settings (sync_with_stdio, cin.tie)',
      'Use getchar() and custom parsing',
      'Use scanf in a loop',
    ],
    correctAnswer: 1,
    explanation:
      "With proper fast I/O settings (`ios_base::sync_with_stdio(false); cin.tie(nullptr);`), cin in a loop is the fastest and simplest approach for most cases. Code: `for(int i = 0; i < n; i++) cin >> arr[i];`. Custom getchar() parsing can be slightly faster but is complex, error-prone, and rarely worth it. Scanf is comparable but cin with fast I/O is cleaner. Benchmark for 10⁶ integers: fast cin (~0.3s), scanf (~0.3s), slow cin (~1.5s), getchar (~0.2s). The 0.1s difference between fast cin and getchar is usually negligible compared to solving time. Best practice: use fast cin for simplicity and good performance. Only optimize to getchar() if you're hitting TLE specifically on I/O.",
    difficulty: 'intermediate',
  },
] as const;
