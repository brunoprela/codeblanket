/**
 * Arranging Coins
 * Problem ID: fundamentals-arrange-coins
 * Order: 74
 */

import { Problem } from '../../../types';

export const arrange_coinsProblem: Problem = {
  id: 'fundamentals-arrange-coins',
  title: 'Arranging Coins',
  difficulty: 'Easy',
  description: `You have n coins to form a staircase.

Each row k must have exactly k coins.
Find complete rows you can form.

**Example:** n = 5 coins
Row 1: 1 coin, Row 2: 2 coins, Row 3: needs 3 but only 2 left
→ 2 complete rows

**Formula:** k(k+1)/2 ≤ n, find max k

This tests:
- Mathematical formula
- Binary search or math
- Integer arithmetic`,
  examples: [
    {
      input: 'n = 5',
      output: '2',
      explanation: 'Rows 1 and 2',
    },
    {
      input: 'n = 8',
      output: '3',
      explanation: 'Rows 1, 2, and 3',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1'],
  hints: [
    'Sum of 1 to k is k(k+1)/2',
    'Use binary search for O(log n)',
    'Or use quadratic formula',
  ],
  starterCode: `def arrange_coins(n):
    """
    Find number of complete staircase rows.
    
    Args:
        n: Number of coins
        
    Returns:
        Number of complete rows
        
    Examples:
        >>> arrange_coins(5)
        2
    """
    pass


# Test
print(arrange_coins(5))
`,
  testCases: [
    {
      input: [5],
      expected: 2,
    },
    {
      input: [8],
      expected: 3,
    },
    {
      input: [1],
      expected: 1,
    },
  ],
  solution: `def arrange_coins(n):
    left, right = 0, n
    
    while left <= right:
        mid = (left + right) // 2
        curr = mid * (mid + 1) // 2
        
        if curr == n:
            return mid
        elif curr < n:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


# Using quadratic formula
import math

def arrange_coins_math(n):
    return int((-1 + math.sqrt(1 + 8 * n)) / 2)`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 74,
  topic: 'Python Fundamentals',
};
