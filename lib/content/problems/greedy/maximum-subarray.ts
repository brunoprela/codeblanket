/**
 * Maximum Subarray
 * Problem ID: maximum-subarray
 * Order: 6
 */

import { Problem } from '../../../types';

export const maximum_subarrayProblem: Problem = {
  id: 'maximum-subarray',
  title: 'Maximum Subarray',
  difficulty: 'Easy',
  topic: 'Greedy',
  description: `Given an integer array \`nums\`, find the subarray with the largest sum, and return its sum.`,
  examples: [
    {
      input: 'nums = [-2,1,-3,4,-1,2,1,-5,4]',
      output: '6',
      explanation: 'The subarray [4,-1,2,1] has the largest sum 6.',
    },
    {
      input: 'nums = [1]',
      output: '1',
    },
    {
      input: 'nums = [5,4,-1,7,8]',
      output: '23',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
  hints: [
    'Kadane algorithm',
    'Track current sum and max sum',
    'If current sum < 0, reset to 0',
  ],
  starterCode: `from typing import List

def max_sub_array(nums: List[int]) -> int:
    """
    Find maximum subarray sum (Kadane algorithm).
    
    Args:
        nums: Input array
        
    Returns:
        Maximum subarray sum
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[-2, 1, -3, 4, -1, 2, 1, -5, 4]],
      expected: 6,
    },
    {
      input: [[1]],
      expected: 1,
    },
    {
      input: [[5, 4, -1, 7, 8]],
      expected: 23,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/maximum-subarray/',
  youtubeUrl: 'https://www.youtube.com/watch?v=5WZl3MMT0Eg',
};
