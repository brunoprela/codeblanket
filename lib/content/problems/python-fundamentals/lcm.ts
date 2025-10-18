/**
 * Least Common Multiple
 * Problem ID: fundamentals-lcm
 * Order: 34
 */

import { Problem } from '../../../types';

export const lcmProblem: Problem = {
  id: 'fundamentals-lcm',
  title: 'Least Common Multiple',
  difficulty: 'Easy',
  description: `Find the least common multiple (LCM) of two numbers.

The LCM is the smallest positive integer divisible by both numbers.

**Formula:** lcm(a, b) = (a * b) / gcd(a, b)

**Example:** lcm(4, 6) = 12

This tests:
- Using GCD to find LCM
- Mathematical relationships
- Integer division`,
  examples: [
    {
      input: 'a = 4, b = 6',
      output: '12',
    },
    {
      input: 'a = 12, b = 18',
      output: '36',
    },
  ],
  constraints: ['1 <= a, b <= 10^6', 'Both numbers positive'],
  hints: [
    'Use relationship: lcm * gcd = a * b',
    'Find GCD first',
    'Use integer division',
  ],
  starterCode: `def lcm(a, b):
    """
    Find least common multiple.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        LCM of a and b
        
    Examples:
        >>> lcm(4, 6)
        12
    """
    pass


# Test
print(lcm(12, 18))
`,
  testCases: [
    {
      input: [4, 6],
      expected: 12,
    },
    {
      input: [12, 18],
      expected: 36,
    },
    {
      input: [5, 7],
      expected: 35,
    },
  ],
  solution: `def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)


# Alternative using math module
import math

def lcm_math(a, b):
    return (a * b) // math.gcd(a, b)`,
  timeComplexity: 'O(log(min(a, b)))',
  spaceComplexity: 'O(1)',
  order: 34,
  topic: 'Python Fundamentals',
};
