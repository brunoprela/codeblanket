/**
 * Ugly Number
 * Problem ID: fundamentals-ugly-number
 * Order: 59
 */

import { Problem } from '../../../types';

export const ugly_numberProblem: Problem = {
  id: 'fundamentals-ugly-number',
  title: 'Ugly Number',
  difficulty: 'Easy',
  description: `Check if a number is ugly.

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

**Example:** 
- 6 = 2 × 3 → ugly
- 14 = 2 × 7 → not ugly (contains 7)

This tests:
- Prime factorization
- Division operations
- Number theory`,
  examples: [
    {
      input: 'n = 6',
      output: 'True',
      explanation: '6 = 2 × 3',
    },
    {
      input: 'n = 14',
      output: 'False',
      explanation: 'Contains prime factor 7',
    },
  ],
  constraints: ['-2^31 <= n <= 2^31 - 1'],
  hints: [
    'Divide by 2, 3, 5 repeatedly',
    'If result is 1, number is ugly',
    'Negative numbers are not ugly',
  ],
  starterCode: `def is_ugly(n):
    """
    Check if number is ugly.
    
    Args:
        n: Integer to check
        
    Returns:
        True if ugly number
        
    Examples:
        >>> is_ugly(6)
        True
        >>> is_ugly(14)
        False
    """
    pass


# Test
print(is_ugly(6))
`,
  testCases: [
    {
      input: [6],
      expected: true,
    },
    {
      input: [14],
      expected: false,
    },
    {
      input: [1],
      expected: true,
    },
  ],
  solution: `def is_ugly(n):
    if n <= 0:
        return False
    
    for factor in [2, 3, 5]:
        while n % factor == 0:
            n //= factor
    
    return n == 1`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 59,
  topic: 'Python Fundamentals',
};
