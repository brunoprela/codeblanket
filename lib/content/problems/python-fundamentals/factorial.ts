/**
 * Factorial Calculator
 * Problem ID: fundamentals-factorial
 * Order: 12
 */

import { Problem } from '../../../types';

export const factorialProblem: Problem = {
  id: 'fundamentals-factorial',
  title: 'Factorial Calculator',
  difficulty: 'Easy',
  description: `Calculate the factorial of a non-negative integer n.

**Factorial** (n!) is the product of all positive integers less than or equal to n.
- 5! = 5 × 4 × 3 × 2 × 1 = 120
- 0! = 1 (by definition)

This problem tests:
- Recursion or iteration
- Base case handling
- Mathematical operations`,
  examples: [
    {
      input: 'n = 5',
      output: '120',
      explanation: '5! = 5 × 4 × 3 × 2 × 1 = 120',
    },
    {
      input: 'n = 0',
      output: '1',
      explanation: '0! = 1 by definition',
    },
  ],
  constraints: ['0 <= n <= 20'],
  hints: [
    'Use a loop to multiply numbers from 1 to n',
    'Or use recursion: n! = n × (n-1)!',
    'Handle the base case: 0! = 1',
  ],
  starterCode: `def factorial(n):
    """
    Calculate factorial of n.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
        
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    pass`,
  testCases: [
    {
      input: [5],
      expected: 120,
    },
    {
      input: [0],
      expected: 1,
    },
    {
      input: [10],
      expected: 3628800,
    },
  ],
  solution: `def factorial(n):
    # Iterative approach
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Recursive approach
def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

# Using math module
import math
def factorial_builtin(n):
    return math.factorial(n)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) iterative, O(n) recursive',
  order: 12,
  topic: 'Python Fundamentals',
};
