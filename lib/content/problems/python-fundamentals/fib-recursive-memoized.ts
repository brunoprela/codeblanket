/**
 * Fibonacci with Memoization
 * Problem ID: fundamentals-fib-recursive-memoized
 * Order: 95
 */

import { Problem } from '../../../types';

export const fib_recursive_memoizedProblem: Problem = {
  id: 'fundamentals-fib-recursive-memoized',
  title: 'Fibonacci with Memoization',
  difficulty: 'Easy',
  description: `Implement Fibonacci using recursion with memoization.

Use a dictionary to cache computed values.

**Example:** fib(10) computes each fib(i) only once

This tests:
- Recursion
- Memoization
- Dictionary usage`,
  examples: [
    {
      input: 'n = 10',
      output: '55',
    },
  ],
  constraints: ['0 <= n <= 100'],
  hints: [
    'Use dictionary to store results',
    'Check cache before computing',
    'Base cases: fib(0)=0, fib(1)=1',
  ],
  starterCode: `def fib_memoized(n, memo=None):
    """
    Fibonacci with memoization.
    
    Args:
        n: Position in sequence
        memo: Memoization dict
        
    Returns:
        Nth Fibonacci number
        
    Examples:
        >>> fib_memoized(10)
        55
    """
    pass


# Test
print(fib_memoized(10))
`,
  testCases: [
    {
      input: [10],
      expected: 55,
    },
    {
      input: [20],
      expected: 6765,
    },
  ],
  solution: `def fib_memoized(n, memo=None):
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 95,
  topic: 'Python Fundamentals',
};
