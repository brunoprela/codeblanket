/**
 * Valid Perfect Square
 * Problem ID: fundamentals-valid-perfect-square
 * Order: 69
 */

import { Problem } from '../../../types';

export const valid_perfect_squareProblem: Problem = {
  id: 'fundamentals-valid-perfect-square',
  title: 'Valid Perfect Square',
  difficulty: 'Easy',
  description: `Check if a number is a perfect square without using sqrt function.

Perfect square: n = k² for some integer k

**Example:** 16 = 4² → true, 14 → false

Use binary search for O(log n) solution.

This tests:
- Binary search
- Integer arithmetic
- Square calculation`,
  examples: [
    {
      input: 'num = 16',
      output: 'True',
    },
    {
      input: 'num = 14',
      output: 'False',
    },
  ],
  constraints: ['1 <= num <= 2^31 - 1'],
  hints: [
    'Use binary search',
    'Check if mid * mid == num',
    'Search range: 1 to num//2 + 1',
  ],
  starterCode: `def is_perfect_square(num):
    """
    Check if number is perfect square.
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect square
        
    Examples:
        >>> is_perfect_square(16)
        True
        >>> is_perfect_square(14)
        False
    """
    pass


# Test
print(is_perfect_square(16))
`,
  testCases: [
    {
      input: [16],
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
  solution: `def is_perfect_square(num):
    if num < 2:
        return True
    
    left, right = 2, num // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1
    
    return False


# Alternative using math property
def is_perfect_square_math(num):
    x = num
    while x * x > num:
        x = (x + num // x) // 2
    return x * x == num`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 69,
  topic: 'Python Fundamentals',
};
