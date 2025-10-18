/**
 * Add Digits Until Single Digit
 * Problem ID: fundamentals-add-digits
 * Order: 60
 */

import { Problem } from '../../../types';

export const add_digitsProblem: Problem = {
  id: 'fundamentals-add-digits',
  title: 'Add Digits Until Single Digit',
  difficulty: 'Easy',
  description: `Repeatedly add all digits until a single digit remains.

**Example:** 38 → 3+8=11 → 1+1=2

**Follow-up:** Can you do it in O(1) without loops?
**Hint:** Digital root = 1 + ((n-1) % 9)

This tests:
- Digit extraction
- Loop iteration
- Mathematical insight (digital root)`,
  examples: [
    {
      input: 'n = 38',
      output: '2',
      explanation: '3+8=11, 1+1=2',
    },
    {
      input: 'n = 0',
      output: '0',
    },
  ],
  constraints: ['0 <= n <= 2^31 - 1'],
  hints: [
    'Extract and sum digits repeatedly',
    'Stop when result < 10',
    'O(1) solution: digital root formula',
  ],
  starterCode: `def add_digits(n):
    """
    Add digits until single digit remains.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Single digit result
        
    Examples:
        >>> add_digits(38)
        2
    """
    pass


# Test
print(add_digits(38))
`,
  testCases: [
    {
      input: [38],
      expected: 2,
    },
    {
      input: [0],
      expected: 0,
    },
    {
      input: [99],
      expected: 9,
    },
  ],
  solution: `def add_digits(n):
    while n >= 10:
        digit_sum = 0
        while n > 0:
            digit_sum += n % 10
            n //= 10
        n = digit_sum
    
    return n


# O(1) solution using digital root
def add_digits_constant(n):
    if n == 0:
        return 0
    return 1 + ((n - 1) % 9)`,
  timeComplexity: 'O(log n) or O(1) with formula',
  spaceComplexity: 'O(1)',
  order: 60,
  topic: 'Python Fundamentals',
};
