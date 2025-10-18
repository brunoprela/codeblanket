/**
 * Intersection of Two Arrays II
 * Problem ID: intersection-two-arrays-ii
 * Order: 11
 */

import { Problem } from '../../../types';

export const intersection_two_arrays_iiProblem: Problem = {
  id: 'intersection-two-arrays-ii',
  title: 'Intersection of Two Arrays II',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  description: `Given two integer arrays \`nums1\` and \`nums2\`, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.`,
  examples: [
    {
      input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
      output: '[2,2]',
    },
    {
      input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
      output: '[4,9]',
      explanation: '[9,4] is also accepted.',
    },
  ],
  constraints: [
    '1 <= nums1.length, nums2.length <= 1000',
    '0 <= nums1[i], nums2[i] <= 1000',
  ],
  hints: [
    'Sort both arrays first',
    'Use two pointers to traverse both arrays',
    'When values match, add to result and move both pointers',
  ],
  starterCode: `from typing import List

def intersect(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find intersection of two arrays with duplicates.
    
    Args:
        nums1: First integer array
        nums2: Second integer array
        
    Returns:
        Array of intersection elements
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 2, 1],
        [2, 2],
      ],
      expected: [2, 2],
    },
    {
      input: [
        [4, 9, 5],
        [9, 4, 9, 8, 4],
      ],
      expected: [4, 9],
    },
    {
      input: [
        [1, 2, 3],
        [4, 5, 6],
      ],
      expected: [],
    },
  ],
  timeComplexity: 'O(n log n + m log m)',
  spaceComplexity: 'O(min(n, m))',
  leetcodeUrl: 'https://leetcode.com/problems/intersection-of-two-arrays-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=lKuK69-hMcc',
};
