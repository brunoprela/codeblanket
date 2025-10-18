/**
 * Find Minimum in Rotated Sorted Array
 * Problem ID: find-minimum-rotated-sorted-array
 * Order: 11
 */

import { Problem } from '../../../types';

export const find_minimum_rotated_sorted_arrayProblem: Problem = {
  id: 'find-minimum-rotated-sorted-array',
  title: 'Find Minimum in Rotated Sorted Array',
  difficulty: 'Medium',
  topic: 'Binary Search',
  description: `Suppose an array of length \`n\` sorted in ascending order is rotated between \`1\` and \`n\` times. For example, the array \`nums = [0,1,2,4,5,6,7]\` might become:
- \`[4,5,6,7,0,1,2]\` if it was rotated \`4\` times.
- \`[0,1,2,4,5,6,7]\` if it was rotated \`7\` times.

Notice that rotating an array \`[a[0], a[1], a[2], ..., a[n-1]]\` 1 time results in the array \`[a[n-1], a[0], a[1], a[2], ..., a[n-2]]\`.

Given the sorted rotated array \`nums\` of unique elements, return the minimum element of this array.

You must write an algorithm that runs in **O(log n)** time.`,
  examples: [
    {
      input: 'nums = [3,4,5,1,2]',
      output: '1',
      explanation: 'The original array was [1,2,3,4,5] rotated 3 times.',
    },
    {
      input: 'nums = [4,5,6,7,0,1,2]',
      output: '0',
      explanation:
        'The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= n <= 5000',
    '-5000 <= nums[i] <= 5000',
    'All the integers of nums are unique',
    'nums is sorted and rotated between 1 and n times',
  ],
  hints: [
    'If the array was not rotated, nums[0] < nums[n-1]',
    'Use binary search. Compare mid with the right boundary',
    'If nums[mid] > nums[right], the minimum is in the right half',
  ],
  starterCode: `from typing import List

def find_min(nums: List[int]) -> int:
    """
    Find minimum in rotated sorted array.
    
    Args:
        nums: Rotated sorted array
        
    Returns:
        Minimum element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 4, 5, 1, 2]],
      expected: 1,
    },
    {
      input: [[4, 5, 6, 7, 0, 1, 2]],
      expected: 0,
    },
    {
      input: [[11, 13, 15, 17]],
      expected: 11,
    },
    {
      input: [[2, 1]],
      expected: 1,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=nIVW4P8b1VA',
};
