/**
 * Max Consecutive Ones
 * Problem ID: max-consecutive-ones
 * Order: 7
 */

import { Problem } from '../../../types';

export const max_consecutive_onesProblem: Problem = {
  id: 'max-consecutive-ones',
  title: 'Max Consecutive Ones',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  order: 7,
  description: `Given a binary array \`nums\`, return the maximum number of consecutive \`1\`'s in the array.`,
  examples: [
    {
      input: 'nums = [1,1,0,1,1,1]',
      output: '3',
      explanation:
        'The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.',
    },
    {
      input: 'nums = [1,0,1,1,0,1]',
      output: '2',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', 'nums[i] is either 0 or 1'],
  hints: [
    'Track current consecutive count',
    'Reset when you see 0',
    'Track maximum count seen',
  ],
  starterCode: `from typing import List

def find_max_consecutive_ones(nums: List[int]) -> int:
    """
    Find maximum consecutive 1s.
    
    Args:
        nums: Binary array
        
    Returns:
        Maximum number of consecutive 1s
    """
    # Write your code here
    pass
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
    {
      input: [[0, 0]],
      expected: 0,
    },
  ],
  solution: `from typing import List

def find_max_consecutive_ones(nums: List[int]) -> int:
    """
    Single pass to count consecutive 1s.
    Time: O(n), Space: O(1)
    """
    max_count = 0
    current_count = 0
    
    for num in nums:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/max-consecutive-ones/',
  youtubeUrl: 'https://www.youtube.com/watch?v=2hBPrR8vx1I',
};
