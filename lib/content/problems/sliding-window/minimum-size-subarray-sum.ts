/**
 * Minimum Size Subarray Sum
 * Problem ID: minimum-size-subarray-sum
 * Order: 11
 */

import { Problem } from '../../../types';

export const minimum_size_subarray_sumProblem: Problem = {
  id: 'minimum-size-subarray-sum',
  title: 'Minimum Size Subarray Sum',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  description: `Given an array of positive integers \`nums\` and a positive integer \`target\`, return the minimal length of a subarray whose sum is greater than or equal to \`target\`. If there is no such subarray, return \`0\` instead.`,
  examples: [
    {
      input: 'target = 7, nums = [2,3,1,2,4,3]',
      output: '2',
      explanation:
        'The subarray [4,3] has the minimal length under the problem constraint.',
    },
    {
      input: 'target = 4, nums = [1,4,4]',
      output: '1',
    },
  ],
  constraints: [
    '1 <= target <= 10^9',
    '1 <= nums.length <= 10^5',
    '1 <= nums[i] <= 10^4',
  ],
  hints: [
    'Use variable-size sliding window',
    'Expand window by adding right, contract by removing left',
  ],
  starterCode: `from typing import List

def min_subarray_len(target: int, nums: List[int]) -> int:
    """
    Find minimum length of subarray with sum >= target.
    
    Args:
        target: Target sum
        nums: Array of positive integers
        
    Returns:
        Minimum length, or 0 if not possible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [7, [2, 3, 1, 2, 4, 3]],
      expected: 2,
    },
    {
      input: [4, [1, 4, 4]],
      expected: 1,
    },
    {
      input: [11, [1, 1, 1, 1, 1, 1, 1, 1]],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/minimum-size-subarray-sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=aYqYMIqZx5s',
};
