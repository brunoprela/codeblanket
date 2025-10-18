/**
 * Intersection of Two Arrays
 * Problem ID: intersection-two-arrays
 * Order: 6
 */

import { Problem } from '../../../types';

export const intersection_two_arraysProblem: Problem = {
  id: 'intersection-two-arrays',
  title: 'Intersection of Two Arrays',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 6,
  description: `Given two integer arrays \`nums1\` and \`nums2\`, return an array of their intersection. Each element in the result must be **unique** and you may return the result in **any order**.`,
  examples: [
    {
      input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
      output: '[2]',
    },
    {
      input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
      output: '[9,4]',
      explanation: '[4,9] is also accepted.',
    },
  ],
  constraints: [
    '1 <= nums1.length, nums2.length <= 1000',
    '0 <= nums1[i], nums2[i] <= 1000',
  ],
  hints: [
    'Use a set to store unique elements from first array',
    'Check which elements from second array exist in the set',
    'Return the intersection as a list',
  ],
  starterCode: `from typing import List

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find intersection of two arrays.
    
    Args:
        nums1: First array
        nums2: Second array
        
    Returns:
        Array of unique intersection elements
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
      expected: [2],
    },
    {
      input: [
        [4, 9, 5],
        [9, 4, 9, 8, 4],
      ],
      expected: [9, 4],
    },
  ],
  solution: `from typing import List

def intersection(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Set intersection approach.
    Time: O(n + m), Space: O(n)
    """
    set1 = set(nums1)
    result = set()
    
    for num in nums2:
        if num in set1:
            result.add(num)
    
    return list(result)

# Alternative: Built-in set operations
def intersection_builtin(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Using Python set intersection.
    Time: O(n + m), Space: O(n + m)
    """
    return list(set(nums1) & set(nums2))
`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(min(n, m))',
  leetcodeUrl: 'https://leetcode.com/problems/intersection-of-two-arrays/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Yz4V1RdPJx8',
};
