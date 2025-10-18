/**
 * Merge Sorted Array
 * Problem ID: merge-sorted-array
 * Order: 8
 */

import { Problem } from '../../../types';

export const merge_sorted_arrayProblem: Problem = {
  id: 'merge-sorted-array',
  title: 'Merge Sorted Array',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  order: 8,
  description: `You are given two integer arrays \`nums1\` and \`nums2\`, sorted in **non-decreasing order**, and two integers \`m\` and \`n\`, representing the number of elements in \`nums1\` and \`nums2\` respectively.

**Merge** \`nums1\` and \`nums2\` into a single array sorted in **non-decreasing order**.

The final sorted array should not be returned by the function, but instead be **stored inside the array \`nums1\`**. To accommodate this, \`nums1\` has a length of \`m + n\`, where the first \`m\` elements denote the elements that should be merged, and the last \`n\` elements are set to \`0\` and should be ignored. \`nums2\` has a length of \`n\`.`,
  examples: [
    {
      input: 'nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3',
      output: '[1,2,2,3,5,6]',
    },
    {
      input: 'nums1 = [1], m = 1, nums2 = [], n = 0',
      output: '[1]',
    },
  ],
  constraints: [
    'nums1.length == m + n',
    'nums2.length == n',
    '0 <= m, n <= 200',
    '1 <= m + n <= 200',
    '-10^9 <= nums1[i], nums2[j] <= 10^9',
  ],
  hints: [
    'Start from the end of both arrays',
    'Compare elements and place the larger one at the end',
    'This avoids overwriting elements in nums1',
  ],
  starterCode: `from typing import List

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Merge nums2 into nums1 in-place.
    
    Args:
        nums1: First sorted array with extra space
        m: Number of elements in nums1
        nums2: Second sorted array
        n: Number of elements in nums2
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3],
      expected: [1, 2, 2, 3, 5, 6],
    },
    {
      input: [[1], 1, [], 0],
      expected: [1],
    },
    {
      input: [[0], 0, [1], 1],
      expected: [1],
    },
  ],
  solution: `from typing import List

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Three pointers: merge from end to avoid overwriting.
    Time: O(m + n), Space: O(1)
    """
    # Pointers for nums1, nums2, and write position
    p1, p2 = m - 1, n - 1
    write = m + n - 1
    
    # Merge from end to start
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[write] = nums1[p1]
            p1 -= 1
        else:
            nums1[write] = nums2[p2]
            p2 -= 1
        write -= 1
`,
  timeComplexity: 'O(m + n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/merge-sorted-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=P1Ic85RarKY',
};
