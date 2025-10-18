/**
 * Longest Increasing Subsequence
 * Problem ID: longest-increasing-subsequence
 * Order: 8
 */

import { Problem } from '../../../types';

export const longest_increasing_subsequenceProblem: Problem = {
  id: 'longest-increasing-subsequence',
  title: 'Longest Increasing Subsequence',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `Given an integer array \`nums\`, return the length of the longest **strictly increasing subsequence**.`,
  examples: [
    {
      input: 'nums = [10,9,2,5,3,7,101,18]',
      output: '4',
      explanation:
        'The longest increasing subsequence is [2,3,7,101], therefore the length is 4.',
    },
    {
      input: 'nums = [0,1,0,3,2,3]',
      output: '4',
    },
    {
      input: 'nums = [7,7,7,7,7,7,7]',
      output: '1',
    },
  ],
  constraints: ['1 <= nums.length <= 2500', '-10^4 <= nums[i] <= 10^4'],
  hints: [
    'dp[i] = length of LIS ending at index i',
    'For each i, check all j < i',
    'If nums[j] < nums[i], dp[i] = max(dp[i], dp[j] + 1)',
  ],
  starterCode: `from typing import List

def length_of_lis(nums: List[int]) -> int:
    """
    Find length of longest increasing subsequence.
    
    Args:
        nums: Input array
        
    Returns:
        Length of LIS
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[10, 9, 2, 5, 3, 7, 101, 18]],
      expected: 4,
    },
    {
      input: [[0, 1, 0, 3, 2, 3]],
      expected: 4,
    },
    {
      input: [[7, 7, 7, 7, 7, 7, 7]],
      expected: 1,
    },
  ],
  timeComplexity: 'O(n^2) or O(n log n) with binary search',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/longest-increasing-subsequence/',
  youtubeUrl: 'https://www.youtube.com/watch?v=cjWnW0hdF1Y',
};
