/**
 * Next Greater Element I
 * Problem ID: next-greater-element-i
 * Order: 7
 */

import { Problem } from '../../../types';

export const next_greater_element_iProblem: Problem = {
  id: 'next-greater-element-i',
  title: 'Next Greater Element I',
  difficulty: 'Easy',
  topic: 'Stack',
  order: 7,
  description: `The **next greater element** of some element \`x\` in an array is the **first greater** element that is **to the right** of \`x\` in the same array.

You are given two **distinct 0-indexed** integer arrays \`nums1\` and \`nums2\`, where \`nums1\` is a subset of \`nums2\`.

For each \`0 <= i < nums1.length\`, find the index \`j\` such that \`nums1[i] == nums2[j]\` and determine the **next greater element** of \`nums2[j]\` in \`nums2\`. If there is no next greater element, then the answer for this query is \`-1\`.

Return an array \`ans\` of length \`nums1.length\` such that \`ans[i]\` is the **next greater element** as described above.`,
  examples: [
    {
      input: 'nums1 = [4,1,2], nums2 = [1,3,4,2]',
      output: '[-1,3,-1]',
      explanation:
        'Next greater of 4 is -1 (no greater). Next greater of 1 is 3. Next greater of 2 is -1.',
    },
    {
      input: 'nums1 = [2,4], nums2 = [1,2,3,4]',
      output: '[3,-1]',
    },
  ],
  constraints: [
    '1 <= nums1.length <= nums2.length <= 1000',
    '0 <= nums1[i], nums2[i] <= 10^4',
    'All integers in nums1 and nums2 are unique',
    'All the integers of nums1 also appear in nums2',
  ],
  hints: [
    'Use a monotonic stack to find next greater elements',
    'Build a hash map of num -> next greater for all nums2 elements',
    'Then lookup each nums1 element in the map',
  ],
  starterCode: `from typing import List

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find next greater element for each element in nums1.
    
    Args:
        nums1: Query array (subset of nums2)
        nums2: Full array
        
    Returns:
        Array of next greater elements
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [4, 1, 2],
        [1, 3, 4, 2],
      ],
      expected: [-1, 3, -1],
    },
    {
      input: [
        [2, 4],
        [1, 2, 3, 4],
      ],
      expected: [3, -1],
    },
  ],
  solution: `from typing import List

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Monotonic stack + hash map.
    Time: O(n + m), Space: O(n)
    """
    # Build map of num -> next greater for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        # Pop smaller elements and record their next greater
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Remaining elements have no next greater
    for num in stack:
        next_greater[num] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]
`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/next-greater-element-i/',
  youtubeUrl: 'https://www.youtube.com/watch?v=68a1Dc_qVq4',
};
