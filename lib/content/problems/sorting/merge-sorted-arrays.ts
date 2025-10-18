/**
 * Merge Two Sorted Arrays
 * Problem ID: merge-sorted-arrays
 * Order: 1
 */

import { Problem } from '../../../types';

export const merge_sorted_arraysProblem: Problem = {
  id: 'merge-sorted-arrays',
  title: 'Merge Two Sorted Arrays',
  difficulty: 'Easy',
  topic: 'Sorting Algorithms',
  order: 1,
  description: `You are given two integer arrays \`nums1\` and \`nums2\`, sorted in **non-decreasing order**.

Merge \`nums2\` into \`nums1\` as one sorted array and return the result.

**Note:** You may assume that both arrays are already sorted.`,
  examples: [
    {
      input: 'nums1 = [1,2,3], nums2 = [2,5,6]',
      output: '[1,2,2,3,5,6]',
      explanation: 'The arrays we are merging are [1,2,3] and [2,5,6].',
    },
    {
      input: 'nums1 = [1], nums2 = []',
      output: '[1]',
      explanation: 'Only one array to merge.',
    },
    {
      input: 'nums1 = [], nums2 = [1]',
      output: '[1]',
      explanation: 'Only one array to merge.',
    },
  ],
  constraints: [
    'nums1.length == m',
    'nums2.length == n',
    '0 <= m, n <= 200',
    '-10^9 <= nums1[i], nums2[i] <= 10^9',
  ],
  hints: [
    'Use two pointers, one for each array',
    'Compare elements and take the smaller one',
    'This is the merge step from merge sort!',
  ],
  starterCode: `from typing import List

def merge_sorted_arrays(nums1: List[int], nums2: List[int]) -> List[int]:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 3],
        [2, 5, 6],
      ],
      expected: [1, 2, 2, 3, 5, 6],
    },
    {
      input: [[1], []],
      expected: [1],
    },
    {
      input: [[], [1]],
      expected: [1],
    },
    {
      input: [
        [1, 3, 5],
        [2, 4, 6],
      ],
      expected: [1, 2, 3, 4, 5, 6],
    },
    {
      input: [
        [-5, -2, 0, 3],
        [-3, -1, 1, 4],
      ],
      expected: [-5, -3, -2, -1, 0, 1, 3, 4],
    },
  ],
  solution: `def merge_sorted_arrays(nums1, nums2):
    result = []
    i = j = 0
    
    # Merge while both arrays have elements
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    # Add remaining elements from nums1
    while i < len(nums1):
        result.append(nums1[i])
        i += 1
    
    # Add remaining elements from nums2
    while j < len(nums2):
        result.append(nums2[j])
        j += 1
    
    return result

# Alternative: using Python's extend
def merge_sorted_arrays_v2(nums1, nums2):
    result = []
    i = j = 0
    
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    result.extend(nums1[i:])
    result.extend(nums2[j:])
    
    return result
`,
  timeComplexity:
    'O(n + m) - we visit each element in both arrays exactly once',
  spaceComplexity:
    'O(n + m) - we create a new array to store the merged result',
  leetcodeUrl: 'https://leetcode.com/problems/merge-sorted-array/',
};
