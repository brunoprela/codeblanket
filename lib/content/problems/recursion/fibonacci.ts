/**
 * Fibonacci Number
 * Problem ID: recursion-fibonacci
 * Order: 6
 */

import { Problem } from '../../../types';

export const fibonacciProblem: Problem = {
  id: 'recursion-fibonacci',
  title: 'Fibonacci Number',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Calculate the nth Fibonacci number using recursion.

The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

**Definition:**
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Important:** Naive recursion is very slow (O(2^n)). You'll need to optimize with memoization!`,
  examples: [
    { input: 'n = 5', output: '5 (sequence: 0, 1, 1, 2, 3, 5)' },
    { input: 'n = 10', output: '55' },
    { input: 'n = 0', output: '0' },
  ],
  constraints: ['0 <= n <= 30'],
  hints: [
    'Base cases: F(0) = 0, F(1) = 1',
    'Recursive case: F(n) = F(n-1) + F(n-2)',
    'WARNING: This is exponential without optimization!',
    'Use memoization (caching) to make it O(n)',
    'Python: @lru_cache decorator or manual dictionary',
  ],
  starterCode: `def fibonacci(n):
    """
    Calculate nth Fibonacci number using recursion.
    
    Args:
        n: Index in Fibonacci sequence (0-indexed)
        
    Returns:
        nth Fibonacci number
        
    Examples:
        >>> fibonacci(5)
        5
        >>> fibonacci(10)
        55
    """
    pass


# Test cases
print(fibonacci(5))   # Expected: 5
print(fibonacci(10))  # Expected: 55
`,
  testCases: [
    { input: [0], expected: 0 },
    { input: [1], expected: 1 },
    { input: [2], expected: 1 },
    { input: [5], expected: 5 },
    { input: [10], expected: 55 },
  ],
  solution: `# NAIVE SOLUTION - VERY SLOW (don't use!)
def fibonacci_naive(n):
    """Naive recursion - O(2^n) time!"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


# OPTIMIZED WITH MEMOIZATION
def fibonacci(n, cache=None):
    """Fibonacci with memoization"""
    if cache is None:
        cache = {}
    
    # Check cache
    if n in cache:
        return cache[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and cache
    result = fibonacci(n - 1, cache) + fibonacci(n - 2, cache)
    cache[n] = result
    return result


# USING PYTHON DECORATOR
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    """Fibonacci with @lru_cache"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# Naive: O(2^n) time, O(n) space
# Memoized: O(n) time, O(n) space`,
  timeComplexity: 'O(2^n) naive, O(n) memoized',
  spaceComplexity: 'O(n)',
  followUp: [
    'Why is naive Fibonacci so slow?',
    'How does memoization help?',
    'Can you implement this with DP (bottom-up)?',
    'Can you optimize space to O(1)?',
  ],
};
