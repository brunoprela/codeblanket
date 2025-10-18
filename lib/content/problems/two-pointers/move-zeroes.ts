/**
 * Move Zeroes
 * Problem ID: move-zeroes
 * Order: 5
 */

import { Problem } from '../../../types';

export const move_zeroesProblem: Problem = {
  id: 'move-zeroes',
  title: 'Move Zeroes',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  order: 5,
  description: `Given an integer array \`nums\`, move all \`0\`'s to the end of it while maintaining the relative order of the non-zero elements.

**Note** that you must do this in-place without making a copy of the array.`,
  examples: [
    {
      input: 'nums = [0,1,0,3,12]',
      output: '[1,3,12,0,0]',
    },
    {
      input: 'nums = [0]',
      output: '[0]',
    },
  ],
  constraints: ['1 <= nums.length <= 10^4', '-2^31 <= nums[i] <= 2^31 - 1'],
  hints: [
    'Use two pointers: one for reading, one for placing non-zero elements',
    'After placing all non-zero elements, fill remaining with zeros',
    'Can you do it in one pass?',
  ],
  starterCode: `from typing import List

def move_zeroes(nums: List[int]) -> None:
    """
    Move all zeroes to the end in-place.
    
    Args:
        nums: Array with some zeros
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[0, 1, 0, 3, 12]],
      expected: [1, 3, 12, 0, 0],
    },
    {
      input: [[0]],
      expected: [0],
    },
    {
      input: [[1, 2, 3]],
      expected: [1, 2, 3],
    },
  ],
  solution: `from typing import List

def move_zeroes(nums: List[int]) -> None:
    """
    Two pointers: snowball approach.
    Time: O(n), Space: O(1)
    """
    # Pointer for placing non-zero elements
    left = 0
    
    # Move all non-zero elements to the front
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/move-zeroes/',
  youtubeUrl: 'https://www.youtube.com/watch?v=aayNRwUN3Do',
};
