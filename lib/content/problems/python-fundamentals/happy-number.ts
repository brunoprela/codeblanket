/**
 * Happy Number
 * Problem ID: fundamentals-happy-number
 * Order: 39
 */

import { Problem } from '../../../types';

export const happy_numberProblem: Problem = {
  id: 'fundamentals-happy-number',
  title: 'Happy Number',
  difficulty: 'Easy',
  description: `Determine if a number is happy.

A happy number is defined by:
1. Start with any positive integer
2. Replace with sum of squares of its digits
3. Repeat until number equals 1 (happy) or loops endlessly (not happy)

**Example:** 19 is happy:
- 1² + 9² = 82
- 8² + 2² = 68
- 6² + 8² = 100
- 1² + 0² + 0² = 1

This tests:
- Set to detect cycles
- Digit extraction
- Loop detection`,
  examples: [
    {
      input: 'n = 19',
      output: 'True',
    },
    {
      input: 'n = 2',
      output: 'False',
      explanation: 'Enters an infinite loop',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1'],
  hints: [
    'Use set to detect cycles',
    'Extract digits and square them',
    'Stop when n=1 or cycle detected',
  ],
  starterCode: `def is_happy(n):
    """
    Check if number is happy.
    
    Args:
        n: Positive integer
        
    Returns:
        True if happy number
        
    Examples:
        >>> is_happy(19)
        True
    """
    pass


# Test
print(is_happy(19))
`,
  testCases: [
    {
      input: [19],
      expected: true,
    },
    {
      input: [2],
      expected: false,
    },
    {
      input: [1],
      expected: true,
    },
  ],
  solution: `def is_happy(n):
    def sum_of_squares(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit ** 2
            num //= 10
        return total
    
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum_of_squares(n)
    
    return n == 1`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',
  order: 39,
  topic: 'Python Fundamentals',
};
