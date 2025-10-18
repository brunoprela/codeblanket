/**
 * Squares of a Sorted Array
 * Problem ID: squares-sorted-array
 * Order: 10
 */

import { Problem } from '../../../types';

export const squares_sorted_arrayProblem: Problem = {
  id: 'squares-sorted-array',
  title: 'Squares of a Sorted Array',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  description: `Given an integer array \`nums\` sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.`,
  examples: [
    {
      input: 'nums = [-4,-1,0,3,10]',
      output: '[0,1,9,16,100]',
      explanation:
        'After squaring, the array becomes [16,1,0,9,100]. After sorting, it becomes [0,1,9,16,100].',
    },
    {
      input: 'nums = [-7,-3,2,3,11]',
      output: '[4,9,9,49,121]',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^4',
    '-10^4 <= nums[i] <= 10^4',
    'nums is sorted in non-decreasing order',
  ],
  hints: [
    'Use two pointers from both ends',
    'The largest square will always be at one of the two ends',
    'Fill the result array from right to left',
  ],
  starterCode: `from typing import List

def sorted_squares(nums: List[int]) -> List[int]:
    """
    Return squares of sorted array in sorted order.
    
    Args:
        nums: Sorted integer array
        
    Returns:
        Sorted array of squares
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[-4, -1, 0, 3, 10]],
      expected: [0, 1, 9, 16, 100],
    },
    {
      input: [[-7, -3, 2, 3, 11]],
      expected: [4, 9, 9, 49, 121],
    },
    {
      input: [[-5, -3, -2, -1]],
      expected: [1, 4, 9, 25],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/squares-of-a-sorted-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=FPCZsG_AkUg',
};
