/**
 * 3Sum
 * Problem ID: three-sum
 * Order: 12
 */

import { Problem } from '../../../types';

export const three_sumProblem: Problem = {
  id: 'three-sum',
  title: '3Sum',
  difficulty: 'Medium',
  topic: 'Two Pointers',
  description: `Given an integer array \`nums\`, return all the triplets \`[nums[i], nums[j], nums[k]]\` such that \`i != j\`, \`i != k\`, and \`j != k\`, and \`nums[i] + nums[j] + nums[k] == 0\`.

Notice that the solution set must not contain duplicate triplets.`,
  examples: [
    {
      input: 'nums = [-1,0,1,2,-1,-4]',
      output: '[[-1,-1,2],[-1,0,1]]',
      explanation:
        'The distinct triplets are [-1,0,1] and [-1,-1,2]. Notice that the order of the output and the order of the triplets does not matter.',
    },
    {
      input: 'nums = [0,1,1]',
      output: '[]',
      explanation: 'The only possible triplet does not sum up to 0.',
    },
  ],
  constraints: ['3 <= nums.length <= 3000', '-10^5 <= nums[i] <= 10^5'],
  hints: [
    'Sort the array first',
    'For each element, use two pointers to find pairs that sum to -element',
    'Skip duplicates to avoid duplicate triplets',
  ],
  starterCode: `from typing import List

def three_sum(nums: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to zero.
    
    Args:
        nums: Integer array
        
    Returns:
        List of triplets that sum to zero
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[-1, 0, 1, 2, -1, -4]],
      expected: [
        [-1, -1, 2],
        [-1, 0, 1],
      ],
    },
    {
      input: [[0, 1, 1]],
      expected: [],
    },
    {
      input: [[0, 0, 0]],
      expected: [[0, 0, 0]],
    },
  ],
  timeComplexity: 'O(n^2)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/3sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=jzZsG8n2R9A',
};
