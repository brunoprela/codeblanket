/**
 * Find Max and Min
 * Problem ID: fundamentals-max-min
 * Order: 14
 */

import { Problem } from '../../../types';

export const max_minProblem: Problem = {
  id: 'fundamentals-max-min',
  title: 'Find Max and Min',
  difficulty: 'Easy',
  description: `Find both the maximum and minimum values in a list in a single pass.

Return a tuple (min, max).

This problem tests:
- List traversal
- Comparison operations
- Tuple return values`,
  examples: [
    {
      input: 'nums = [3, 1, 4, 1, 5, 9, 2, 6]',
      output: '(1, 9)',
      explanation: 'Minimum is 1, maximum is 9',
    },
    {
      input: 'nums = [-5, -2, -10, -1]',
      output: '(-10, -1)',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^5', 'Cannot use built-in min() or max()'],
  hints: [
    'Initialize min and max with first element',
    'Iterate through remaining elements',
    'Update min and max as needed',
  ],
  starterCode: `def find_max_min(nums):
    """
    Find max and min in a list.
    
    Args:
        nums: List of numbers
        
    Returns:
        Tuple (min, max)
        
    Examples:
        >>> find_max_min([3, 1, 4, 1, 5])
        (1, 5)
    """
    pass`,
  testCases: [
    {
      input: [[3, 1, 4, 1, 5, 9, 2, 6]],
      expected: [1, 9],
    },
    {
      input: [[-5, -2, -10, -1]],
      expected: [-10, -1],
    },
    {
      input: [[42]],
      expected: [42, 42],
    },
  ],
  solution: `def find_max_min(nums):
    if not nums:
        return None
    
    min_val = nums[0]
    max_val = nums[0]
    
    for num in nums[1:]:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num
    
    return (min_val, max_val)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 14,
  topic: 'Python Fundamentals',
};
