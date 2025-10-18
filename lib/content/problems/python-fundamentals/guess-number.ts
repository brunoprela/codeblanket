/**
 * Guess Number Higher or Lower
 * Problem ID: fundamentals-guess-number
 * Order: 70
 */

import { Problem } from '../../../types';

export const guess_numberProblem: Problem = {
  id: 'fundamentals-guess-number',
  title: 'Guess Number Higher or Lower',
  difficulty: 'Easy',
  description: `Guess a number from 1 to n using binary search.

API available: guess(num) returns:
- -1: my number is lower
- 1: my number is higher
- 0: correct!

Minimize number of guesses using binary search.

This tests:
- Binary search
- API interaction
- Search strategy`,
  examples: [
    {
      input: 'n = 10, pick = 6',
      output: '6',
      explanation: 'Binary search finds it',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1', '1 <= pick <= n'],
  hints: [
    'Use binary search',
    'Update left/right based on guess result',
    'Classic binary search pattern',
  ],
  starterCode: `# The guess API is predefined
def guess(num):
    """Mock API - compare num with picked number"""
    pick = 6  # This would be set in real game
    if num > pick:
        return -1
    elif num < pick:
        return 1
    else:
        return 0


def guess_number(n):
    """
    Find the number I picked.
    
    Args:
        n: Upper bound
        
    Returns:
        The picked number
        
    Examples:
        >>> guess_number(10)  # pick = 6
        6
    """
    pass


# Test
print(guess_number(10))
`,
  testCases: [
    {
      input: [10],
      expected: 6,
    },
  ],
  solution: `def guess_number(n):
    left, right = 1, n
    
    while left <= right:
        mid = (left + right) // 2
        result = guess(mid)
        
        if result == 0:
            return mid
        elif result == -1:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1  # Should never reach here`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  order: 70,
  topic: 'Python Fundamentals',
};
