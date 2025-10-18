/**
 * Search in Rotated Sorted Array
 * Problem ID: search-rotated-array
 * Order: 2
 */

import { Problem } from '../../../types';

export const search_rotated_arrayProblem: Problem = {
  id: 'search-rotated-array',
  title: 'Search in Rotated Sorted Array',
  difficulty: 'Medium',
  topic: 'Binary Search',
  order: 2,
  description: `There is an integer array \`nums\` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, \`nums\` is **rotated** at an unknown pivot index \`k\` (\`0 <= k < nums.length\`) such that the resulting array is \`[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]\` (**0-indexed**). For example, \`[0,1,2,4,5,6,7]\` might be rotated at pivot index \`3\` and become \`[4,5,6,7,0,1,2]\`.

Given the array \`nums\` **after** the rotation and an integer \`target\`, return the index of \`target\` if it is in \`nums\`, or \`-1\` if it is not in \`nums\`.

You must write an algorithm with **O(log n)** runtime complexity.`,
  examples: [
    {
      input: 'nums = [4, 5, 6, 7, 0, 1, 2], target = 0',
      output: '4',
      explanation: 'The target 0 is at index 4',
    },
    {
      input: 'nums = [4, 5, 6, 7, 0, 1, 2], target = 3',
      output: '-1',
      explanation: '3 does not exist in nums',
    },
    {
      input: 'nums = [1], target = 0',
      output: '-1',
    },
  ],
  constraints: [
    '1 <= nums.length <= 5000',
    '-10^4 <= nums[i] <= 10^4',
    'All values of nums are unique',
    'nums is guaranteed to be rotated at some pivot',
  ],
  hints: [
    'At least one half of the array is always sorted',
    'Which half is sorted? Check if nums[left] <= nums[mid]',
    'If the left half is sorted, check if target is in that range',
    'If target is not in the sorted half, search the other half',
  ],
  starterCode: `from typing import List

def search(nums: List[int], target: int) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 5, 6, 7, 0, 1, 2], 0],
      expected: 4,
    },
    {
      input: [[4, 5, 6, 7, 0, 1, 2], 3],
      expected: -1,
    },
    {
      input: [[1], 0],
      expected: -1,
    },
    {
      input: [[1], 1],
      expected: 0,
    },
    {
      input: [[3, 1], 1],
      expected: 1,
    },
    {
      input: [[5, 1, 3], 5],
      expected: 0,
    },
    {
      input: [[4, 5, 6, 7, 8, 1, 2, 3], 8],
      expected: 4,
    },
  ],
  solution: `def search(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
`,
  timeComplexity: 'O(log n) - binary search on the rotated array',
  spaceComplexity: 'O(1) - constant space usage',
  leetcodeUrl: 'https://leetcode.com/problems/search-in-rotated-sorted-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=U8XENwh8Oy8',
};
