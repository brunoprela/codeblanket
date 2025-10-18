/**
 * Find Peak Element
 * Problem ID: find-peak-element
 * Order: 13
 */

import { Problem } from '../../../types';

export const find_peak_elementProblem: Problem = {
  id: 'find-peak-element',
  title: 'Find Peak Element',
  difficulty: 'Medium',
  topic: 'Binary Search',
  description: `A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array \`nums\`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that \`nums[-1] = nums[n] = -∞\`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in **O(log n)** time.`,
  examples: [
    {
      input: 'nums = [1,2,3,1]',
      output: '2',
      explanation:
        '3 is a peak element and your function should return the index number 2.',
    },
    {
      input: 'nums = [1,2,1,3,5,6,4]',
      output: '5',
      explanation:
        'Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.',
    },
  ],
  constraints: [
    '1 <= nums.length <= 1000',
    '-2^31 <= nums[i] <= 2^31 - 1',
    'nums[i] != nums[i + 1] for all valid i',
  ],
  hints: [
    'Use binary search',
    'If nums[mid] < nums[mid + 1], there must be a peak in the right half',
    'Otherwise, there must be a peak in the left half (including mid)',
  ],
  starterCode: `from typing import List

def find_peak_element(nums: List[int]) -> int:
    """
    Find a peak element and return its index.
    
    Args:
        nums: Integer array
        
    Returns:
        Index of a peak element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 1]],
      expected: 2,
    },
    {
      input: [[1, 2, 1, 3, 5, 6, 4]],
      expected: 5,
    },
    {
      input: [[1]],
      expected: 0,
    },
    {
      input: [[1, 2]],
      expected: 1,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/find-peak-element/',
  youtubeUrl: 'https://www.youtube.com/watch?v=kMzJy9es7Hc',
};
