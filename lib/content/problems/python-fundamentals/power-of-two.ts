/**
 * Power of Two
 * Problem ID: fundamentals-power-of-two
 * Order: 38
 */

import { Problem } from '../../../types';

export const power_of_twoProblem: Problem = {
  id: 'fundamentals-power-of-two',
  title: 'Power of Two',
  difficulty: 'Easy',
  description: `Check if a number is a power of two.

A number is a power of two if: n = 2^k for some integer k ≥ 0

**Examples:** 1, 2, 4, 8, 16... are powers of two

**Bit trick:** Powers of two have only one bit set
- n & (n-1) == 0 for powers of two

This tests:
- Bit manipulation
- Mathematical properties
- Edge cases`,
  examples: [
    {
      input: 'n = 16',
      output: 'True',
      explanation: '16 = 2^4',
    },
    {
      input: 'n = 5',
      output: 'False',
    },
  ],
  constraints: ['-2^31 <= n <= 2^31 - 1'],
  hints: [
    'Powers of 2 have only one bit set',
    'Use bit manipulation: n & (n-1)',
    'Handle edge cases: 0 and negative numbers',
  ],
  starterCode: `def is_power_of_two(n):
    """
    Check if number is power of two.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is power of 2
        
    Examples:
        >>> is_power_of_two(16)
        True
        >>> is_power_of_two(5)
        False
    """
    pass


# Test
print(is_power_of_two(16))
`,
  testCases: [
    {
      input: [16],
      expected: true,
    },
    {
      input: [5],
      expected: false,
    },
    {
      input: [1],
      expected: true,
    },
    {
      input: [0],
      expected: false,
    },
  ],
  solution: `def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


# Alternative using division
def is_power_of_two_div(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 38,
  topic: 'Python Fundamentals',
};
