/**
 * Perfect Number
 * Problem ID: fundamentals-perfect-number
 * Order: 23
 */

import { Problem } from '../../../types';

export const perfect_numberProblem: Problem = {
  id: 'fundamentals-perfect-number',
  title: 'Perfect Number',
  difficulty: 'Easy',
  description: `Check if a number is a perfect number.

A perfect number is a positive integer equal to the sum of its proper positive divisors (excluding itself).

**Examples:**
- 6 → True (divisors: 1, 2, 3; sum = 6)
- 28 → True (divisors: 1, 2, 4, 7, 14; sum = 28)
- 12 → False

This problem tests:
- Finding divisors
- Mathematical properties
- Loop optimization`,
  examples: [
    {
      input: 'num = 6',
      output: 'True',
      explanation: '1 + 2 + 3 = 6',
    },
    {
      input: 'num = 28',
      output: 'True',
      explanation: '1 + 2 + 4 + 7 + 14 = 28',
    },
    {
      input: 'num = 12',
      output: 'False',
    },
  ],
  constraints: ['1 <= num <= 10^8'],
  hints: [
    'Find all divisors up to sqrt(n)',
    'Add both i and n/i when found',
    'Compare sum with original number',
  ],
  starterCode: `def is_perfect_number(num):
    """
    Check if number is perfect.
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect, False otherwise
        
    Examples:
        >>> is_perfect_number(6)
        True
        >>> is_perfect_number(28)
        True
    """
    pass`,
  testCases: [
    {
      input: [6],
      expected: true,
    },
    {
      input: [28],
      expected: true,
    },
    {
      input: [12],
      expected: false,
    },
    {
      input: [1],
      expected: false,
    },
  ],
  solution: `def is_perfect_number(num):
    if num <= 1:
        return False
    
    # Find sum of divisors
    divisor_sum = 1  # 1 is always a divisor
    
    # Only check up to sqrt(num)
    i = 2
    while i * i <= num:
        if num % i == 0:
            divisor_sum += i
            # Add the paired divisor (but not if it's the square root)
            if i * i != num:
                divisor_sum += num // i
        i += 1
    
    return divisor_sum == num`,
  timeComplexity: 'O(sqrt(n))',
  spaceComplexity: 'O(1)',
  order: 23,
  topic: 'Python Fundamentals',
};
