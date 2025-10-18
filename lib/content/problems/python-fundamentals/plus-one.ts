/**
 * Plus One
 * Problem ID: fundamentals-plus-one
 * Order: 44
 */

import { Problem } from '../../../types';

export const plus_oneProblem: Problem = {
  id: 'fundamentals-plus-one',
  title: 'Plus One',
  difficulty: 'Easy',
  description: `Given a non-empty array of digits representing a non-negative integer, add one to the integer.

Digits are stored such that most significant digit is at head of list.

**Example:** [1,2,3] represents 123, so +1 = [1,2,4]

Handle carry: [9,9,9] + 1 = [1,0,0,0]

This tests:
- Array manipulation
- Carry propagation
- Edge cases`,
  examples: [
    {
      input: 'digits = [1,2,3]',
      output: '[1,2,4]',
    },
    {
      input: 'digits = [9,9,9]',
      output: '[1,0,0,0]',
    },
  ],
  constraints: ['1 <= len(digits) <= 100', '0 <= digits[i] <= 9'],
  hints: [
    'Start from the end',
    'Handle carry propagation',
    'Add new digit at front if needed',
  ],
  starterCode: `def plus_one(digits):
    """
    Add one to number represented as array.
    
    Args:
        digits: Array of digits
        
    Returns:
        Array representing digits + 1
        
    Examples:
        >>> plus_one([1,2,3])
        [1, 2, 4]
        >>> plus_one([9,9,9])
        [1, 0, 0, 0]
    """
    pass


# Test
print(plus_one([9,9,9]))
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: [1, 2, 4],
    },
    {
      input: [[9, 9, 9]],
      expected: [1, 0, 0, 0],
    },
    {
      input: [[0]],
      expected: [1],
    },
  ],
  solution: `def plus_one(digits):
    n = len(digits)
    
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    
    # If we're here, all digits were 9
    return [1] + digits


# Alternative converting to/from int
def plus_one_simple(digits):
    num = int(''.join(map(str, digits))) + 1
    return [int(d) for d in str(num)]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) or O(n) if all 9s',
  order: 44,
  topic: 'Python Fundamentals',
};
