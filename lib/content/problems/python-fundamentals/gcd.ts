/**
 * Greatest Common Divisor
 * Problem ID: fundamentals-gcd
 * Order: 33
 */

import { Problem } from '../../../types';

export const gcdProblem: Problem = {
  id: 'fundamentals-gcd',
  title: 'Greatest Common Divisor',
  difficulty: 'Easy',
  description: `Find the greatest common divisor (GCD) of two numbers.

The GCD is the largest positive integer that divides both numbers evenly.

**Example:** gcd(48, 18) = 6

Use the Euclidean algorithm:
- gcd(a, b) = gcd(b, a % b)
- Base case: gcd(a, 0) = a

This tests:
- Recursion or iteration
- Modulo operations
- Mathematical algorithms`,
  examples: [
    {
      input: 'a = 48, b = 18',
      output: '6',
    },
    {
      input: 'a = 100, b = 50',
      output: '50',
    },
  ],
  constraints: ['1 <= a, b <= 10^9', 'Both numbers positive'],
  hints: [
    'Use Euclidean algorithm',
    'gcd(a, b) = gcd(b, a % b)',
    'Recursion or while loop',
  ],
  starterCode: `def gcd(a, b):
    """
    Find greatest common divisor.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        GCD of a and b
        
    Examples:
        >>> gcd(48, 18)
        6
    """
    pass


# Test
print(gcd(48, 18))
`,
  testCases: [
    {
      input: [48, 18],
      expected: 6,
    },
    {
      input: [100, 50],
      expected: 50,
    },
    {
      input: [17, 19],
      expected: 1,
    },
  ],
  solution: `def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Recursive version
def gcd_recursive(a, b):
    if b == 0:
        return a
    return gcd_recursive(b, a % b)`,
  timeComplexity: 'O(log(min(a, b)))',
  spaceComplexity: 'O(1) iterative, O(log n) recursive',
  order: 33,
  topic: 'Python Fundamentals',
};
