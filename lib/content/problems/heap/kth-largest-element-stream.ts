/**
 * Kth Largest Element in a Stream
 * Problem ID: kth-largest-element-stream
 * Order: 2
 */

import { Problem } from '../../../types';

export const kth_largest_element_streamProblem: Problem = {
  id: 'kth-largest-element-stream',
  title: 'Kth Largest Element in a Stream',
  difficulty: 'Easy',
  topic: 'Heap / Priority Queue',
  description: `Design a class to find the \`k-th\` largest element in a stream. Note that it is the \`k-th\` largest element in the sorted order, not the \`k-th\` distinct element.

Implement \`KthLargest\` class:
- \`KthLargest(int k, int[] nums)\` Initializes the object with the integer \`k\` and the stream of integers \`nums\`.
- \`int add(int val)\` Appends the integer \`val\` to the stream and returns the element representing the \`k-th\` largest element in the stream.`,
  examples: [
    {
      input:
        '["KthLargest", "add", "add", "add", "add", "add"], [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]',
      output: '[null, 4, 5, 5, 8, 8]',
    },
  ],
  constraints: [
    '1 <= k <= 10^4',
    '0 <= nums.length <= 10^4',
    '-10^4 <= nums[i] <= 10^4',
    '-10^4 <= val <= 10^4',
    'At most 10^4 calls will be made to add',
  ],
  hints: [
    'Use a min heap of size k',
    'Always maintain k largest elements',
    'Top of heap is kth largest',
  ],
  starterCode: `from typing import List
import heapq

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        """
        Initialize with k and initial stream.
        
        Args:
            k: Position of element to track
            nums: Initial stream
        """
        # Write your code here
        pass
    
    def add(self, val: int) -> int:
        """
        Add value to stream and return kth largest.
        
        Args:
            val: Value to add
            
        Returns:
            Kth largest element
        """
        # Write your code here
        pass
`,
  testCases: [
    {
      input: [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
      expected: [null, 4, 5, 5, 8, 8],
    },
  ],
  timeComplexity: 'O(log k) for add',
  spaceComplexity: 'O(k)',
  leetcodeUrl: 'https://leetcode.com/problems/kth-largest-element-in-a-stream/',
  youtubeUrl: 'https://www.youtube.com/watch?v=hOjcdrqMoQ8',
};
