/**
 * Single Number II
 * Problem ID: single-number-ii
 * Order: 8
 */

import { Problem } from '../../../types';

export const single_number_iiProblem: Problem = {
  id: 'single-number-ii',
  title: 'Single Number II',
  difficulty: 'Medium',
  topic: 'Bit Manipulation',
  description: `Given an integer array \`nums\` where every element appears **three times** except for one, which appears **exactly once**. Find the single element and return it.

You must implement a solution with a linear runtime complexity and use only constant extra space.`,
  examples: [
    {
      input: 'nums = [2,2,3,2]',
      output: '3',
    },
    {
      input: 'nums = [0,1,0,1,0,1,99]',
      output: '99',
    },
  ],
  constraints: [
    '1 <= nums.length <= 3 * 10^4',
    '-2^31 <= nums[i] <= 2^31 - 1',
    'Each element in nums appears exactly three times except for one element which appears once',
  ],
  hints: [
    'Count bits at each position',
    'If count % 3 != 0, bit belongs to single number',
  ],
  starterCode: `from typing import List

def single_number_ii(nums: List[int]) -> int:
    """
    Find element appearing once (others appear 3x).
    
    Args:
        nums: Array of integers
        
    Returns:
        Single element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 2, 3, 2]],
      expected: 3,
    },
    {
      input: [[0, 1, 0, 1, 0, 1, 99]],
      expected: 99,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/single-number-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=cOFAmaMBVps',
};
