/**
 * Running Sum of 1d Array
 * Problem ID: running-sum-1d-array
 * Order: 24
 */

import { Problem } from '../../../types';

export const running_sum_1d_arrayProblem: Problem = {
  id: 'running-sum-1d-array',
  title: 'Running Sum of 1d Array',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  description: `Given an array \`nums\`. We define a running sum of an array as \`runningSum[i] = sum(nums[0]…nums[i])\`.

Return the running sum of \`nums\`.`,
  examples: [
    {
      input: 'nums = [1,2,3,4]',
      output: '[1,3,6,10]',
      explanation:
        'Running sum is obtained as follows: [1, 1+2, 1+2+3, 1+2+3+4].',
    },
    {
      input: 'nums = [1,1,1,1,1]',
      output: '[1,2,3,4,5]',
    },
  ],
  constraints: ['1 <= nums.length <= 1000', '-10^6 <= nums[i] <= 10^6'],
  hints: [
    'Think about how each element relates to the previous sum',
    'Can you do this in-place?',
  ],
  starterCode: `from typing import List

def running_sum(nums: List[int]) -> List[int]:
    """
    Calculate running sum of array.
    
    Args:
        nums: Input array
        
    Returns:
        Array of running sums
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4]],
      expected: [1, 3, 6, 10],
    },
    {
      input: [[1, 1, 1, 1, 1]],
      expected: [1, 2, 3, 4, 5],
    },
    {
      input: [[3, 1, 2, 10, 1]],
      expected: [3, 4, 6, 16, 17],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/running-sum-of-1d-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=MruDdQWT-4k',
};
