/**
 * Binary Search
 * Problem ID: binary-search
 * Order: 1
 */

import { Problem } from '../../../types';

export const binary_searchProblem: Problem = {
  id: 'binary-search',
  title: 'Binary Search',
  difficulty: 'Easy',
  topic: 'Binary Search',

  leetcodeUrl: 'https://leetcode.com/problems/binary-search/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s4DPM8ct1pI',
  order: 1,
  description: `Given an array of integers \`nums\` which is sorted in ascending order, and an integer \`target\`, write a function to search \`target\` in \`nums\`. If \`target\` exists, then return its index. Otherwise, return \`-1\`.

You must write an algorithm with **O(log n)** runtime complexity.`,
  examples: [
    {
      input: 'nums = [-1, 0, 3, 5, 9, 12], target = 9',
      output: '4',
      explanation: '9 exists in nums and its index is 4',
    },
    {
      input: 'nums = [-1, 0, 3, 5, 9, 12], target = 2',
      output: '-1',
      explanation: '2 does not exist in nums so return -1',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^4',
    '-10^4 < nums[i], target < 10^4',
    'All the integers in nums are unique',
    'nums is sorted in ascending order',
  ],
  hints: [
    'Think about dividing the search space in half with each iteration',
    'What should you do when the middle element is greater than the target?',
    'What are your loop termination conditions?',
  ],
  starterCode: `from typing import List

def binary_search(nums: List[int], target: int) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[-1, 0, 3, 5, 9, 12], 9],
      expected: 4,
    },
    {
      input: [[-1, 0, 3, 5, 9, 12], 2],
      expected: -1,
    },
    {
      input: [[5], 5],
      expected: 0,
    },
    {
      input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1],
      expected: 0,
    },
    {
      input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10],
      expected: 9,
    },
    {
      input: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11],
      expected: -1,
    },
  ],
  solution: `def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
`,
  timeComplexity: 'O(log n) - we halve the search space with each iteration',
  spaceComplexity: 'O(1) - we only use a constant amount of extra space',
};
