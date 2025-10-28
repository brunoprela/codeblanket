export default [
  {
    id: 'cp-m1-s8-q1',
    section: 'Macros & Preprocessor Tricks',
    question: 'What is the purpose of #define in competitive programming?',
    options: [
      'To declare global variables',
      'To create text substitution macros that can save typing',
      'To import libraries',
      'To define class methods',
    ],
    correctAnswer: 1,
    explanation:
      "`#define` creates preprocessor macros for text substitution before compilation. Common CP macros: `#define ll long long`, `#define pb push_back`, `#define all(x) (x).begin(),(x).end()`. The preprocessor literally replaces macro text before compiling. Example: `ll sum;` becomes `long long sum;` before compilation. Useful for: (1) Type aliases, (2) Shortening repetitive code, (3) Loop macros. NOT for variables (use const), NOT for imports (use #include), NOT for class methods (use actual methods). Pitfall: macros don't respect scope or types - purely textual replacement. Use parentheses in macro definitions to avoid operator precedence issues: `#define SQUARE(x) ((x) * (x))` not `#define SQUARE(x) x * x`.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s8-q2',
    section: 'Macros & Preprocessor Tricks',
    question: 'Why should you add parentheses in macro definitions?',
    options: [
      'For better readability',
      'To prevent operator precedence issues when macro is used in expressions',
      "It's required by C++ syntax",
      'To make macros work with all data types',
    ],
    correctAnswer: 1,
    explanation:
      'Parentheses in macros prevent operator precedence bugs. Example: `#define SQUARE(x) x * x` seems OK, but `SQUARE(a+b)` expands to `a+b * a+b` = `a + (b*a) + b` (wrong!). Correct: `#define SQUARE(x) ((x) * (x))` expands to `((a+b) * (a+b))` (correct). Always wrap: (1) Macro parameters in parentheses, (2) Entire macro in parentheses. Example: `#define MAX(a,b) ((a) > (b) ? (a) : (b))`. Not required by syntax (code compiles either way), but required for correctness. This is a classic bug source in CP. Modern alternative: use constexpr functions or templates instead of macros when possible for type safety.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s8-q3',
    section: 'Macros & Preprocessor Tricks',
    question:
      'What does #ifdef LOCAL typically do in competitive programming templates?',
    options: [
      'It checks if the code is running on the local machine',
      'It enables conditional compilation for debug code',
      'It imports local library files',
      'It optimizes code for local execution',
    ],
    correctAnswer: 1,
    explanation:
      '`#ifdef LOCAL` enables conditional compilation for debug-only code. When you compile with `-DLOCAL`, the debug code is included; without it, debug code is excluded. Example: `#ifdef LOCAL\\nfreopen("input.txt", "r", stdin);\\n#define debug(x) cerr << #x << " = " << x << endl\\n#else\\n#define debug(x)\\n#endif`. Local: compile with `g++ -DLOCAL solution.cpp` - file I/O and debug enabled. Submission: compile with `g++ solution.cpp` - debug code automatically removed. This prevents accidentally submitting debug output (would cause WA). Clean workflow: write debug statements freely, submit without changes. All debug code is automatically stripped. Similar pattern for other conditional features.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s8-q4',
    section: 'Macros & Preprocessor Tricks',
    question: 'What is the # operator used for in macro definitions?',
    options: [
      'Commenting out code',
      'String-ifying macro arguments (converting to string literal)',
      'Concatenating macro arguments',
      'Including header files',
    ],
    correctAnswer: 1,
    explanation:
      'The `#` operator (stringification) converts macro arguments to string literals. Example: `#define debug(x) cerr << #x << " = " << x << endl`. When you write `debug(sum);`, it expands to `cerr << "sum" << " = " << sum << endl`, printing variable name and value. Very useful for debugging! The `##` operator (different from `#`) concatenates tokens. Example: `#define VAR(n) var##n` makes `VAR(1)` become `var1`. Stringification is more common in CP for debug macros. Not for comments (use //), not for includes (use #include without #). This makes debug output much more informative without manual string typing.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s8-q5',
    section: 'Macros & Preprocessor Tricks',
    question:
      'What is a potential pitfall of using macros in competitive programming?',
    options: [
      'Macros always make code slower',
      'Macros can have unexpected side effects due to multiple evaluation of arguments',
      'Macros are not supported in modern C++',
      'Macros take up too much memory',
    ],
    correctAnswer: 1,
    explanation:
      "Macros can evaluate arguments multiple times, causing bugs. Example: `#define MAX(a,b) ((a) > (b) ? (a) : (b))` with `MAX(i++, j++)` evaluates increments twice! Expands to: `((i++) > (j++) ? (i++) : (j++))` - i or j incremented twice. Solution: use inline functions or templates instead when side effects possible. Macros don't make code slower (preprocessor replacement before compilation). Macros are fully supported (just have pitfalls). Macros don't affect memory (purely textual). Other pitfalls: no type checking, no scope, debugging harder (error messages show expanded text). Despite pitfalls, macros are ubiquitous in CP for convenience. Just be aware of limitations and use carefully with side-effect-free expressions.",
    difficulty: 'intermediate',
  },
] as const;
