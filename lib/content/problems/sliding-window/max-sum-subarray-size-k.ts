/**
 * Maximum Sum of Subarray of Size K
 * Problem ID: max-sum-subarray-size-k
 * Order: 10
 */

import { Problem } from '../../../types';

export const max_sum_subarray_size_kProblem: Problem = {
  id: 'max-sum-subarray-size-k',
  title: 'Maximum Sum of Subarray of Size K',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  description: `Given an array of integers \`nums\` and an integer \`k\`, find the maximum sum of a contiguous subarray of size \`k\`.`,
  examples: [
    {
      input: 'nums = [2,1,5,1,3,2], k = 3',
      output: '9',
      explanation: 'Subarray [5,1,3] has the maximum sum 9.',
    },
    {
      input: 'nums = [2,3,4,1,5], k = 2',
      output: '7',
      explanation: 'Subarray [3,4] has the maximum sum 7.',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^5',
    '1 <= k <= nums.length',
    '-10^4 <= nums[i] <= 10^4',
  ],
  hints: [
    'Use sliding window technique',
    'Subtract the left element and add the right element',
  ],
  starterCode: `from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    Find maximum sum of subarray of size k.
    
    Args:
        nums: Integer array
        k: Subarray size
        
    Returns:
        Maximum sum
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 1, 5, 1, 3, 2], 3],
      expected: 9,
    },
    {
      input: [[2, 3, 4, 1, 5], 2],
      expected: 7,
    },
    {
      input: [[1, 4, 2, 10, 23, 3, 1, 0, 20], 4],
      expected: 39,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/maximum-sum-of-subarray-of-size-k/',
  youtubeUrl: 'https://www.youtube.com/watch?v=KtpqeN0Goro',
};
