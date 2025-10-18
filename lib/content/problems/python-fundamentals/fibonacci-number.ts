/**
 * Fibonacci Number (Nth)
 * Problem ID: fundamentals-fibonacci-number
 * Order: 90
 */

import { Problem } from '../../../types';

export const fibonacci_numberProblem: Problem = {
  id: 'fundamentals-fibonacci-number',
  title: 'Fibonacci Number (Nth)',
  difficulty: 'Easy',
  description: `Calculate the Nth Fibonacci number.

F(0) = 0, F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1

**Example:** F(4) = 3 (0,1,1,2,3)

This tests:
- Dynamic programming
- Iterative vs recursive
- Space optimization`,
  examples: [
    {
      input: 'n = 4',
      output: '3',
    },
    {
      input: 'n = 2',
      output: '1',
    },
  ],
  constraints: ['0 <= n <= 30'],
  hints: [
    'Use iteration for O(n) time, O(1) space',
    'Track only last two values',
    'Recursion with memoization works',
  ],
  starterCode: `def fib(n):
    """
    Calculate Nth Fibonacci number.
    
    Args:
        n: Position in sequence
        
    Returns:
        Nth Fibonacci number
        
    Examples:
        >>> fib(4)
        3
    """
    pass


# Test
print(fib(4))
`,
  testCases: [
    {
      input: [4],
      expected: 3,
    },
    {
      input: [2],
      expected: 1,
    },
    {
      input: [0],
      expected: 0,
    },
  ],
  solution: `def fib(n):
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


# Recursive with memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 90,
  topic: 'Python Fundamentals',
};
