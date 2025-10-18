/**
 * Search in Rotated Sorted Array II
 * Problem ID: search-rotated-sorted-array-ii
 * Order: 12
 */

import { Problem } from '../../../types';

export const search_rotated_sorted_array_iiProblem: Problem = {
  id: 'search-rotated-sorted-array-ii',
  title: 'Search in Rotated Sorted Array II',
  difficulty: 'Medium',
  topic: 'Binary Search',
  description: `There is an integer array \`nums\` sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, \`nums\` is rotated at an unknown pivot index \`k\` (\`0 <= k < nums.length\`) such that the resulting array is \`[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]\` (0-indexed).

Given the array \`nums\` after the rotation and an integer \`target\`, return \`true\` if \`target\` is in \`nums\`, or \`false\` if it is not in \`nums\`.

You must decrease the overall operation steps as much as possible.`,
  examples: [
    {
      input: 'nums = [2,5,6,0,0,1,2], target = 0',
      output: 'true',
    },
    {
      input: 'nums = [2,5,6,0,0,1,2], target = 3',
      output: 'false',
    },
  ],
  constraints: [
    '1 <= nums.length <= 5000',
    '-10^4 <= nums[i] <= 10^4',
    'nums is guaranteed to be rotated at some pivot',
    '-10^4 <= target <= 10^4',
  ],
  hints: [
    'This is the follow-up problem where nums may contain duplicates',
    'When nums[left] == nums[mid] == nums[right], we cannot determine which side is sorted',
    'In worst case, time complexity degrades to O(n)',
  ],
  starterCode: `from typing import List

def search(nums: List[int], target: int) -> bool:
    """
    Search in rotated sorted array with duplicates.
    
    Args:
        nums: Rotated sorted array (may contain duplicates)
        target: Target value to search
        
    Returns:
        True if target is in nums, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 5, 6, 0, 0, 1, 2], 0],
      expected: true,
    },
    {
      input: [[2, 5, 6, 0, 0, 1, 2], 3],
      expected: false,
    },
    {
      input: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1], 2],
      expected: true,
    },
  ],
  timeComplexity: 'O(log n) average, O(n) worst case',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/search-in-rotated-sorted-array-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=w-Aw00H73ak',
};
