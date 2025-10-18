/**
 * Sum Without + Operator
 * Problem ID: fundamentals-sum-of-two-integers
 * Order: 77
 */

import { Problem } from '../../../types';

export const sum_of_two_integersProblem: Problem = {
  id: 'fundamentals-sum-of-two-integers',
  title: 'Sum Without + Operator',
  difficulty: 'Medium',
  description: `Calculate sum of two integers without using + or - operators.

Use bit manipulation instead.

**Key insight:**
- XOR gives sum without carry
- AND gives carry positions
- Shift carry left and repeat

This tests:
- Bit manipulation
- Carry calculation
- Iterative bit operations`,
  examples: [
    {
      input: 'a = 1, b = 2',
      output: '3',
    },
    {
      input: 'a = 2, b = 3',
      output: '5',
    },
  ],
  constraints: ['-1000 <= a, b <= 1000'],
  hints: [
    'XOR for sum without carry',
    'AND << 1 for carry',
    'Repeat until no carry',
  ],
  starterCode: `def get_sum(a, b):
    """
    Add two integers without + operator.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Sum of a and b
        
    Examples:
        >>> get_sum(1, 2)
        3
    """
    pass


# Test
print(get_sum(1, 2))
`,
  testCases: [
    {
      input: [1, 2],
      expected: 3,
    },
    {
      input: [2, 3],
      expected: 5,
    },
  ],
  solution: `def get_sum(a, b):
    # 32-bit integer limit
    mask = 0xFFFFFFFF
    
    while b != 0:
        # XOR: sum without carry
        # AND << 1: carry
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    
    # Handle negative numbers
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)


# Simpler for Python (no 32-bit limit)
def get_sum_simple(a, b):
    while b != 0:
        carry = a & b
        a = a ^ b
        b = carry << 1
    return a`,
  timeComplexity: 'O(1) - at most 32 iterations',
  spaceComplexity: 'O(1)',
  order: 77,
  topic: 'Python Fundamentals',
};
