export default [
  {
    id: 'cp-m1-s7-q1',
    section: 'C++11/14/17/20 Features for CP',
    question:
      "What is the primary benefit of C++11's auto keyword in competitive programming?",
    options: [
      'It makes code run faster',
      'It automatically optimizes memory usage',
      'It allows type inference, reducing typing and avoiding type errors',
      'It enables automatic parallelization',
    ],
    correctAnswer: 2,
    explanation:
      "`auto` enables type inference, letting the compiler deduce types automatically. This saves typing and prevents type errors. Example: `auto it = m.begin();` instead of `map<int,int>::iterator it = m.begin();`. Especially useful for: (1) Iterators, (2) Pair/tuple unpacking, (3) Complex types. Code: `for(auto& p : map)` instead of `for(pair<const int, int>& p : map)`. Doesn't affect runtime speed or memory (purely compile-time feature). Pitfall: `auto` copies by default, use `auto&` for references to avoid copies. Common usage: `auto [x, y] = pair;` (structured binding in C++17). Time saved: reduces typing and mental overhead, especially for complex nested types.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s7-q2',
    section: 'C++11/14/17/20 Features for CP',
    question:
      'What are structured bindings (C++17) and how are they useful in CP?',
    options: [
      'A way to bind functions to objects',
      'A way to unpack tuples and pairs directly in declarations',
      'A way to create multiple variables with the same value',
      'A way to bind memory addresses',
    ],
    correctAnswer: 1,
    explanation:
      'Structured bindings (C++17) allow unpacking tuples, pairs, and structs directly in declarations. Syntax: `auto [x, y] = pair;` or `auto [a, b, c] = tuple;`. Very useful in CP for: (1) Iterating maps: `for(auto [key, val] : map)`, (2) Unpacking pairs: `auto [x, y] = coordinates[i];`, (3) Returning multiple values: `auto [min, max] = get_range();`. Before C++17: `for(auto p : map) { int key = p.first; int val = p.second; }`. With C++17: `for(auto [key, val] : map)`. Much cleaner! Works with arrays, pairs, tuples, and custom structs. Makes code more readable and reduces boilerplate.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s7-q3',
    section: 'C++11/14/17/20 Features for CP',
    question:
      'What is the benefit of lambda functions in C++ for competitive programming?',
    options: [
      'They automatically optimize recursive calls',
      'They allow defining small inline functions, especially useful for custom comparators',
      'They enable multi-threaded execution',
      'They provide automatic memoization',
    ],
    correctAnswer: 1,
    explanation:
      'Lambda functions allow defining small, inline anonymous functions. Very useful for: (1) Custom comparators in sort, (2) Custom operations in STL algorithms, (3) Short helper functions. Example: `sort(all(v), [](int a, int b) { return a > b; });` for descending sort. Syntax: `[capture](parameters) { body }`. Captures: `[]` (nothing), `[=]` (by value), `[&]` (by reference). Common use: `sort(all(pairs), [](pii a, pii b) { return a.second < b.second; });` to sort pairs by second element. No automatic optimizations or memoization - purely syntactic convenience. Saves time defining named functions for one-off operations. Also useful for: priority_queue comparators, filter/transform operations.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s7-q4',
    section: 'C++11/14/17/20 Features for CP',
    question: 'What does the range-based for loop (for-each loop) in C++11 do?',
    options: [
      'Iterates through a range of numbers',
      'Iterates through container elements without explicit iterators',
      'Creates parallel for loops',
      'Automatically optimizes loop performance',
    ],
    correctAnswer: 1,
    explanation:
      'Range-based for loops iterate through containers without explicit iterators. Syntax: `for(auto x : container) { /* use x */ }`. Works with vectors, arrays, maps, sets, etc. Example: `for(int x : vector) cout << x << " ";` instead of `for(int i = 0; i < v.size(); i++) cout << v[i] << " ";`. Use `auto&` to modify elements or avoid copies: `for(auto& x : vector) x *= 2;`. For const iteration: `for(const auto& x : vector)`. Very clean and readable. Doesn\'t automatically parallelize or optimize (purely syntactic sugar). Equivalent to iterator loop but cleaner. Can\'t easily get index (use traditional loop if needed). Common pitfall: forgetting & for large objects (causes copies).',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s7-q5',
    section: 'C++11/14/17/20 Features for CP',
    question: 'What is std::tie() useful for in competitive programming?',
    options: [
      'It ties input and output streams together',
      'It creates a tuple of references, useful for unpacking and comparing tuples',
      'It optimizes memory layout',
      'It enables multi-threading',
    ],
    correctAnswer: 1,
    explanation:
      '`std::tie()` creates a tuple of references, useful for: (1) Unpacking tuples/pairs: `tie(x, y) = make_pair(1, 2);`, (2) Comparing multiple values: `tie(a, b, c) < tie(x, y, z)` for lexicographic comparison, (3) Returning multiple values from functions. Example: instead of comparing a, then b, then c separately, use `tie(a,b,c) < tie(x,y,z)`. Very clean for custom comparators: `return tie(a.x, a.y) < tie(b.x, b.y);`. Works with assignment: `int x, y; tie(x, y) = pair;`. Use `std::ignore` for unused values: `tie(x, ignore, z) = tuple;`. Purely compile-time feature, no runtime cost. Makes code cleaner and less error-prone for multi-value comparisons.',
    difficulty: 'intermediate',
  },
] as const;
