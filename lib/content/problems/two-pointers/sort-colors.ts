/**
 * Sort Colors
 * Problem ID: sort-colors
 * Order: 13
 */

import { Problem } from '../../../types';

export const sort_colorsProblem: Problem = {
  id: 'sort-colors',
  title: 'Sort Colors',
  difficulty: 'Medium',
  topic: 'Two Pointers',
  description: `Given an array \`nums\` with \`n\` objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers \`0\`, \`1\`, and \`2\` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.`,
  examples: [
    {
      input: 'nums = [2,0,2,1,1,0]',
      output: '[0,0,1,1,2,2]',
    },
    {
      input: 'nums = [2,0,1]',
      output: '[0,1,2]',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= n <= 300',
    'nums[i] is either 0, 1, or 2',
  ],
  hints: [
    'Use the Dutch National Flag algorithm',
    'Maintain three pointers: low (for 0s), mid (current), high (for 2s)',
    'Swap elements to their correct regions',
  ],
  starterCode: `from typing import List

def sort_colors(nums: List[int]) -> None:
    """
    Sort array of 0s, 1s, and 2s in-place.
    
    Args:
        nums: Array of integers (0, 1, or 2)
        
    Returns:
        None, modifies nums in-place
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 0, 2, 1, 1, 0]],
      expected: [0, 0, 1, 1, 2, 2],
    },
    {
      input: [[2, 0, 1]],
      expected: [0, 1, 2],
    },
    {
      input: [[0]],
      expected: [0],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/sort-colors/',
  youtubeUrl: 'https://www.youtube.com/watch?v=4xbWSRZHqac',
};
