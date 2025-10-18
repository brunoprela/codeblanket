/**
 * Median of Two Sorted Arrays
 * Problem ID: median-two-sorted-arrays
 * Order: 3
 */

import { Problem } from '../../../types';

export const median_two_sorted_arraysProblem: Problem = {
  id: 'median-two-sorted-arrays',
  title: 'Median of Two Sorted Arrays',
  difficulty: 'Hard',
  topic: 'Binary Search',
  order: 3,
  description: `Given two sorted arrays \`nums1\` and \`nums2\` of size \`m\` and \`n\` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be **O(log(m+n))**.`,
  examples: [
    {
      input: 'nums1 = [1, 3], nums2 = [2]',
      output: '2.0',
      explanation: 'merged array = [1, 2, 3] and median is 2',
    },
    {
      input: 'nums1 = [1, 2], nums2 = [3, 4]',
      output: '2.5',
      explanation:
        'merged array = [1, 2, 3, 4] and median is (2 + 3) / 2 = 2.5',
    },
  ],
  constraints: [
    'nums1.length == m',
    'nums2.length == n',
    '0 <= m <= 1000',
    '0 <= n <= 1000',
    '1 <= m + n <= 2000',
    '-10^6 <= nums1[i], nums2[i] <= 10^6',
  ],
  hints: [
    'The median divides the array into two equal halves',
    'Use binary search on the smaller array',
    'Find the correct partition where elements on the left <= elements on the right',
    'Handle edge cases: empty arrays, arrays of different sizes',
  ],
  starterCode: `from typing import List

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 3], [2]],
      expected: 2.0,
    },
    {
      input: [
        [1, 2],
        [3, 4],
      ],
      expected: 2.5,
    },
    {
      input: [
        [0, 0],
        [0, 0],
      ],
      expected: 0.0,
    },
    {
      input: [[], [1]],
      expected: 1.0,
    },
    {
      input: [[2], []],
      expected: 2.0,
    },
    {
      input: [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
      ],
      expected: 5.5,
    },
    {
      input: [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
      ],
      expected: 5.5,
    },
  ],
  solution: `def findMedianSortedArrays(nums1, nums2):
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        # Handle edge cases
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if we found the correct partition
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            # If total length is even
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0
            # If total length is odd
            else:
                return float(max(maxLeft1, maxLeft2))
        elif maxLeft1 > minRight2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    return 0.0
`,
  timeComplexity: 'O(log(min(m, n))) - binary search on the smaller array',
  spaceComplexity: 'O(1) - constant space usage',
  leetcodeUrl: 'https://leetcode.com/problems/median-of-two-sorted-arrays/',
  youtubeUrl: 'https://www.youtube.com/watch?v=q6IEA26hvXc',
};
