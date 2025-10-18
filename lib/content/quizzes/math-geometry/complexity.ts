/**
 * Quiz questions for Time and Space Complexity section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question: 'Analyze complexity of math operations. Which are fast vs slow?',
    sampleAnswer:
      'Fast O(1): addition, subtraction, multiplication (constant-size integers), bitwise ops, simple comparisons. Moderate: division (slower than multiply), modulo (similar to division). Slow: square root (iterative methods), trigonometric functions, logarithms. For large numbers: fast power O(log n), factorial O(n) linear, GCD O(log min(a,b)), prime check O(sqrt(n)) trial division or O(1) with Sieve. Matrix operations: O(n³) for n×n. For example, adding two 32-bit ints is single CPU instruction O(1). Computing n! naively is n multiplications O(n). Sieve for primes up to n is O(n log log n). Key: understand what operations are expensive, optimize bottlenecks. In practice: addition fast, division moderate, sqrt slow.',
    keyPoints: [
      'Fast O(1): +, -, ×, bitwise',
      'Moderate: /, % (slower than multiply)',
      'Slow: sqrt, trig, log (iterative)',
      'Large numbers: factorial O(n), GCD O(log n)',
      'Matrix: O(n³) for n×n',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare different algorithms for same math problem (e.g., Fibonacci, factorial). When to use each?',
    sampleAnswer:
      'Fibonacci: recursive O(2^n) - never use, memoization O(n) - simple but space, iteration O(n) - standard, matrix O(log n) - large n. Factorial: loop O(n) - standard, cache with DP - reuse, modular at each step - prevent overflow. Prime check: trial division O(sqrt(n)) - single check, Sieve O(n log log n) - many queries. GCD: Euclidean O(log n) - always use, naive O(min(a,b)) - never. Choose based on: problem size, space constraints, query frequency. For example, Fibonacci up to 50: iteration works. For n=10^18: matrix exponentiation. For checking if 1000 numbers are prime: Sieve precomputation. For single GCD: Euclidean. Key: know complexity and constraints.',
    keyPoints: [
      'Fibonacci: iterate O(n) standard, matrix O(log n) large',
      'Factorial: loop O(n), modular at each step',
      'Primes: trial O(sqrt(n)) single, Sieve O(n log log n) many',
      'GCD: Euclidean O(log n) always',
      'Choose based on: size, space, frequency',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain space-time tradeoffs in math problems (e.g., Sieve vs trial division).',
    sampleAnswer:
      'Sieve of Eratosthenes: precompute all primes up to n, O(n log log n) time, O(n) space. Then answer "is p prime?" in O(1). Trial division: O(sqrt(n)) per query, O(1) space. Tradeoff: Sieve pays upfront cost and space for fast queries. Use Sieve when: many queries (amortized cost low), n is reasonable (< 10^7). Use trial when: few queries, n very large, memory limited. Similar: factorial precomputation O(n) space for O(1) query vs O(n) per computation. Memoization: O(n) space for O(1) lookup vs recomputing. The pattern: precompute (time+space) for fast queries, or compute on-demand (time only) for space efficiency. Choice depends on query frequency and memory constraints.',
    keyPoints: [
      'Sieve: O(n) space, O(1) query vs Trial: O(1) space, O(sqrt(n)) query',
      'Precompute (time+space) vs on-demand (time only)',
      'Sieve when: many queries, reasonable n',
      'Trial when: few queries, large n, memory limited',
      'Pattern: space for speed tradeoff',
    ],
  },
];
