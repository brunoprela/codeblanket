/**
 * Quiz questions for Memoization: Optimizing Recursion section
 */

export const memoizationQuiz = [
  {
    id: 'q-memo1',
    question:
      'What is memoization and how does it improve recursive algorithms?',
    sampleAnswer:
      'Memoization is caching the results of expensive function calls and returning the cached result when the same inputs occur again. It improves recursion by eliminating redundant calculations in problems with overlapping subproblems. Example: naive Fibonacci is O(2^n) because fib (n-1) and fib (n-2) recalculate the same values. With memoization, we compute each fib (k) only once, reducing to O(n) time. Space trade-off: O(n) for the cache.',
    keyPoints: [
      'Cache results of function calls',
      'Return cached result for repeated inputs',
      'Eliminates redundant calculations',
      'Transforms exponential to polynomial time',
      'Requires overlapping subproblems',
      'Trade-off: O(n) space for O(exponential → polynomial) time',
    ],
  },
  {
    id: 'q-memo2',
    question: 'How do you implement memoization in Python?',
    sampleAnswer:
      'Three main approaches: 1) **@lru_cache decorator** (easiest): `from functools import lru_cache; @lru_cache (maxsize=None)` before function definition, 2) **Dictionary** (manual): pass memo dict as parameter or use default argument, check if key in memo before computing, store result after computing, 3) **Class with __call__** (advanced): maintain cache as instance variable. The decorator approach is most Pythonic and handles argument hashing automatically.',
    keyPoints: [
      '@lru_cache decorator - most Pythonic',
      'Dictionary with memo parameter',
      'Check memo before computing',
      'Store result in memo after computing',
      'Handle both hashable and unhashable args',
      'maxsize=None for unlimited cache',
    ],
  },
  {
    id: 'q-memo3',
    question: 'When should you NOT use memoization?',
    sampleAnswer:
      "Don't use memoization when: 1) **No overlapping subproblems** - each computation is unique (e.g., simple array sum), wasting memory, 2) **Impure functions** - results depend on external state/side effects, cache gives wrong results, 3) **Limited benefit** - small recursion depth or cheap computations, overhead not worth it, 4) **Unbounded input space** - cache grows indefinitely, memory issues. Memoization only helps with pure functions having repeated computations.",
    keyPoints: [
      'No overlapping subproblems = wasted memory',
      'Impure functions give incorrect cached results',
      'Shallow recursion = overhead not worth it',
      'Unbounded inputs = memory explosion',
      'Only for pure functions with repeated work',
    ],
  },
];
