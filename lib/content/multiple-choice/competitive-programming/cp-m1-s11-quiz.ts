export default [
  {
    id: 'cp-m1-s11-q1',
    section: 'Template Metaprogramming Basics',
    question:
      'What is the primary advantage of using templates in competitive programming?',
    options: [
      'Templates make code run faster at runtime',
      'Templates allow writing generic code that works with multiple types',
      'Templates automatically optimize algorithms',
      'Templates prevent all compilation errors',
    ],
    correctAnswer: 1,
    explanation:
      "Templates enable generic programming - writing code once that works with multiple types. Example: `template<typename T> T max(T a, T b) { return (a > b) ? a : b; }` works for int, long long, double, etc. Benefits: (1) DRY (Don't Repeat Yourself), (2) Type safety, (3) Zero runtime overhead (template instantiation at compile time). Templates don't automatically optimize algorithms or prevent errors - they just enable type-generic code. Common uses in CP: generic utility functions (gcd, max, min), container operations, custom data structures that work with any type. Tradeoff: longer compilation time, cryptic error messages.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s11-q2',
    section: 'Template Metaprogramming Basics',
    question: 'When would you use a function template vs a macro?',
    options: [
      'Always use templates, macros are deprecated',
      'Templates for type-safe generic functions, macros for simple text substitution',
      'Macros are always faster than templates',
      'Templates and macros are interchangeable',
    ],
    correctAnswer: 1,
    explanation:
      'Templates provide type safety and proper scoping; macros are simple text replacement. Use templates when: (1) Need type checking, (2) Multiple types with same logic, (3) Want debuggable code. Use macros when: (1) Very simple substitution (like `#define ll long long`), (2) Need preprocessor features, (3) Extreme brevity needed. Example: `template<typename T> T gcd(T a, T b)` is safer than `#define GCD(a,b) ((b)?gcd(b,a%b):(a))`. Template catches type errors at compile time, prevents multiple evaluation of arguments. Macros are NOT faster at runtime (both are compile-time). In CP: mix of both - templates for complex logic, macros for brevity.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s11-q3',
    section: 'Template Metaprogramming Basics',
    question: 'What is template specialization?',
    options: [
      'Making templates run faster',
      'Providing a specific implementation for a particular type',
      'Optimizing template compilation time',
      'Converting templates to regular functions',
    ],
    correctAnswer: 1,
    explanation:
      'Template specialization provides custom implementation for specific types. Example: Generic template: `template<typename T> void print(T x) { cout << x; }`. Specialization for vector: `template<> void print(vector<int> v) { for(int x : v) cout << x << " "; }`. Now `print(5)` uses generic version, `print(vector)` uses specialized version. Useful for: (1) Type-specific optimizations, (2) Custom behavior for certain types, (3) Handling special cases. Common in CP for: debug printing (different output for containers vs primitives), custom hash functions for pairs/tuples. Doesn\'t make code run faster automatically - just allows type-specific logic.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s11-q4',
    section: 'Template Metaprogramming Basics',
    question:
      'What is the downside of heavy template usage in competitive programming?',
    options: [
      'Templates are slower at runtime',
      'Templates produce cryptic compiler error messages and longer compilation times',
      'Templates use more memory at runtime',
      'Templates are not supported by most judges',
    ],
    correctAnswer: 1,
    explanation:
      "Template downsides: (1) Cryptic error messages (template instantiation errors can be pages long), (2) Longer compilation time (templates generate code for each type used). Templates have ZERO runtime overhead (same performance as hand-written code for each type). They don't use more runtime memory. All modern judges support templates. Example error: using `set<pair<int,int>>` without operator< defined generates 100+ lines of template errors. Solution: (1) Read only first few lines of errors, (2) Test templates thoroughly before use, (3) Use explicit instantiations for debugging. Despite downsides, templates are widely used in CP for their benefits. Just be prepared for debugging challenges.",
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s11-q5',
    section: 'Template Metaprogramming Basics',
    question:
      'What are variadic templates (C++11) useful for in competitive programming?',
    options: [
      'Making templates run faster',
      'Accepting a variable number of arguments of potentially different types',
      'Automatic code optimization',
      'Preventing template errors',
    ],
    correctAnswer: 1,
    explanation:
      'Variadic templates accept variable number of arguments. Example: `template<typename... Args> void print(Args... args)` can be called with any number of arguments: `print(1, 2, "hello", 3.14)`. Useful for: (1) Generic debug functions: `debug(x, y, z)` prints all variables, (2) Min/max of multiple values: `minimum(a, b, c, d, e)`, (3) Generic utility functions. Implementation uses recursion or fold expressions (C++17). Code: `template<typename T, typename... Args> T minimum(T first, Args... args) { return min(first, minimum(args...)); }`. Not automatically faster or preventing errors - just enables flexible function signatures. Very handy for debug macros and generic utilities in your CP template.',
    difficulty: 'advanced',
  },
] as const;
