/**
 * Sum of Digits
 * Problem ID: fundamentals-digit-sum
 * Order: 15
 */

import { Problem } from '../../../types';

export const digit_sumProblem: Problem = {
  id: 'fundamentals-digit-sum',
  title: 'Sum of Digits',
  difficulty: 'Easy',
  description: `Calculate the sum of all digits in a positive integer.

**Example:** 
- 123 → 1 + 2 + 3 = 6
- 9999 → 9 + 9 + 9 + 9 = 36

This problem tests:
- Number manipulation
- String conversion or modulo operations
- Loop iteration`,
  examples: [
    {
      input: 'n = 123',
      output: '6',
      explanation: '1 + 2 + 3 = 6',
    },
    {
      input: 'n = 9999',
      output: '36',
      explanation: '9 + 9 + 9 + 9 = 36',
    },
  ],
  constraints: ['0 <= n <= 10^9'],
  hints: [
    'Convert to string and iterate through characters',
    'Or use modulo (%) and division (//) to extract digits',
    'Use sum() with a generator for concise solution',
  ],
  starterCode: `def sum_of_digits(n):
    """
    Calculate sum of digits in a number.
    
    Args:
        n: Positive integer
        
    Returns:
        Sum of all digits
        
    Examples:
        >>> sum_of_digits(123)
        6
    """
    pass`,
  testCases: [
    {
      input: [123],
      expected: 6,
    },
    {
      input: [9999],
      expected: 36,
    },
    {
      input: [0],
      expected: 0,
    },
  ],
  solution: `def sum_of_digits(n):
    # String approach
    return sum(int(digit) for digit in str(n))

# Mathematical approach
def sum_of_digits_math(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total`,
  timeComplexity: 'O(log n) - number of digits',
  spaceComplexity: 'O(1)',
  order: 15,
  topic: 'Python Fundamentals',
};
