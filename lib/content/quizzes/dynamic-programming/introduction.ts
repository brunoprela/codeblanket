/**
 * Quiz questions for Introduction to Dynamic Programming section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what dynamic programming is and how it differs from recursion. When do you use DP over plain recursion?',
    sampleAnswer:
      'Dynamic programming optimizes recursive solutions by storing results of subproblems to avoid recomputing them. Plain recursion recalculates same subproblems many times. For example, Fibonacci fib(5) calls fib(3) multiple times. DP stores fib(3) result once and reuses it. Use DP when problem has two properties: optimal substructure (solution built from optimal solutions of subproblems) and overlapping subproblems (same subproblems solved repeatedly). Without overlapping subproblems, DP provides no benefit over recursion. DP trades space for time: stores O(n) or O(n^2) results to avoid exponential redundant computation. The key insight: if you find yourself solving the same problem multiple times, memoize it.',
    keyPoints: [
      'DP stores subproblem results to avoid recomputation',
      'Plain recursion: recalculates same subproblems',
      'Needs: optimal substructure + overlapping subproblems',
      'Trades space for time: O(n) space, avoids exponential time',
      'If solving same problem multiple times → memoize',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare top-down (memoization) vs bottom-up (tabulation) DP. When would you choose each?',
    sampleAnswer:
      'Top-down uses recursion with memoization: solve problem recursively, store results in cache, check cache before computing. Bottom-up uses iteration with table: solve smallest subproblems first, build up to final answer using table. Top-down is intuitive (matches recursive thinking), only computes needed subproblems, but has recursion overhead. Bottom-up is faster (no recursion), uses less space (can optimize to O(1) in some cases), but computes all subproblems even if not needed. Choose top-down for: complex dependencies, not all subproblems needed, easier to code. Choose bottom-up for: simple dependencies, need all subproblems anyway, want best performance. In interviews, start with top-down (easier), optimize to bottom-up if asked.',
    keyPoints: [
      'Top-down: recursion + memoization',
      'Bottom-up: iteration + table',
      'Top-down: intuitive, only needed subproblems',
      'Bottom-up: faster, can optimize space',
      'Interview: start top-down, optimize to bottom-up',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through recognizing a DP problem. What keywords or patterns signal that DP might be needed?',
    sampleAnswer:
      'Several signals indicate DP. Keywords: "maximum", "minimum", "longest", "shortest", "count ways", "can you", "all possible". Asking for optimal value or count of solutions. Problem naturally recursive but naive recursion is too slow. You are making choices and want optimal outcome. Pattern examples: Fibonacci-like recurrence (current depends on previous), grid paths (combine paths from top and left), subsequence problems (include or skip element), knapsack-like decisions (take or leave item). If you write recursive solution and notice same parameters recurring many times, that is overlapping subproblems - use DP. Ask: can I break problem into smaller similar problems? Do subproblems overlap?',
    keyPoints: [
      'Keywords: max, min, longest, shortest, count ways',
      'Optimal value or count of solutions',
      'Recursive but too slow',
      'Patterns: Fibonacci, grid paths, subsequence, knapsack',
      'Same subproblems recurring → use DP',
    ],
  },
];
