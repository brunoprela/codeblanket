/**
 * Kth Largest Element in an Array
 * Problem ID: kth-largest-element-array
 * Order: 4
 */

import { Problem } from '../../../types';

export const kth_largest_element_arrayProblem: Problem = {
  id: 'kth-largest-element-array',
  title: 'Kth Largest Element in an Array',
  difficulty: 'Medium',
  topic: 'Heap / Priority Queue',
  description: `Given an integer array \`nums\` and an integer \`k\`, return the \`k-th\` largest element in the array.

Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Can you solve it without sorting?`,
  examples: [
    {
      input: 'nums = [3,2,1,5,6,4], k = 2',
      output: '5',
    },
    {
      input: 'nums = [3,2,3,1,2,4,5,5,6], k = 4',
      output: '4',
    },
  ],
  constraints: ['1 <= k <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
  hints: [
    'Use a min heap of size k',
    'Maintain k largest elements',
    'Top of heap is kth largest',
  ],
  starterCode: `from typing import List
import heapq

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find kth largest element in array.
    
    Args:
        nums: Input array
        k: Position from largest
        
    Returns:
        Kth largest element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 2, 1, 5, 6, 4], 2],
      expected: 5,
    },
    {
      input: [[3, 2, 3, 1, 2, 4, 5, 5, 6], 4],
      expected: 4,
    },
  ],
  timeComplexity: 'O(n log k)',
  spaceComplexity: 'O(k)',
  leetcodeUrl: 'https://leetcode.com/problems/kth-largest-element-in-an-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XEmy13g1Qxc',
};
