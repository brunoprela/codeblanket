/**
 * Max Consecutive Ones
 * Problem ID: fundamentals-max-consecutive-ones
 * Order: 84
 */

import { Problem } from '../../../types';

export const max_consecutive_onesProblem: Problem = {
  id: 'fundamentals-max-consecutive-ones',
  title: 'Max Consecutive Ones',
  difficulty: 'Easy',
  description: `Find maximum number of consecutive 1s in a binary array.

**Example:** [1,1,0,1,1,1] → 3

This tests:
- Array traversal
- Counting consecutive elements
- Max tracking`,
  examples: [
    {
      input: 'nums = [1,1,0,1,1,1]',
      output: '3',
    },
    {
      input: 'nums = [1,0,1,1,0,1]',
      output: '2',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^5', 'nums[i] is 0 or 1'],
  hints: [
    'Track current consecutive count',
    'Reset count when 0 found',
    'Update max as you go',
  ],
  starterCode: `def find_max_consecutive_ones(nums):
    """
    Find max consecutive 1s.
    
    Args:
        nums: Binary array
        
    Returns:
        Max consecutive ones count
        
    Examples:
        >>> find_max_consecutive_ones([1,1,0,1,1,1])
        3
    """
    pass


# Test
print(find_max_consecutive_ones([1,1,0,1,1,1]))
`,
  testCases: [
    {
      input: [[1, 1, 0, 1, 1, 1]],
      expected: 3,
    },
    {
      input: [[1, 0, 1, 1, 0, 1]],
      expected: 2,
    },
  ],
  solution: `def find_max_consecutive_ones(nums):
    max_count = 0
    current_count = 0
    
    for num in nums:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 84,
  topic: 'Python Fundamentals',
};
