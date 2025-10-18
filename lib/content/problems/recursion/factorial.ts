/**
 * Factorial
 * Problem ID: recursion-factorial
 * Order: 1
 */

import { Problem } from '../../../types';

export const factorialProblem: Problem = {
  id: 'recursion-factorial',
  title: 'Factorial',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Calculate the factorial of a non-negative integer n.

The factorial of n (written as n!) is the product of all positive integers less than or equal to n.

**Definition:**
- 0! = 1
- n! = n × (n-1)!

**Examples:**
- 5! = 5 × 4 × 3 × 2 × 1 = 120
- 3! = 3 × 2 × 1 = 6

This is the classic introduction to recursion problem.`,
  examples: [
    { input: 'n = 5', output: '120' },
    { input: 'n = 0', output: '1' },
    { input: 'n = 1', output: '1' },
  ],
  constraints: ['0 <= n <= 12', 'Result will fit in a 32-bit integer'],
  hints: [
    'Base case: factorial(0) = 1 and factorial(1) = 1',
    'Recursive case: factorial(n) = n * factorial(n-1)',
    'Make sure n decreases with each recursive call',
  ],
  starterCode: `def factorial(n):
    """
    Calculate factorial of n using recursion.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! (factorial of n)
        
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    pass


# Test cases
print(factorial(5))  # Expected: 120
print(factorial(0))  # Expected: 1
`,
  testCases: [
    { input: [5], expected: 120 },
    { input: [0], expected: 1 },
    { input: [1], expected: 1 },
    { input: [3], expected: 6 },
    { input: [10], expected: 3628800 },
  ],
  solution: `def factorial(n):
    """Calculate factorial using recursion"""
    # Base cases
    if n <= 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)


# Time Complexity: O(n) - makes n recursive calls
# Space Complexity: O(n) - call stack depth is n`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n) - call stack',
  followUp: [
    'Can you implement this iteratively?',
    'How would you handle very large numbers?',
    'What happens if n is negative?',
  ],
};
