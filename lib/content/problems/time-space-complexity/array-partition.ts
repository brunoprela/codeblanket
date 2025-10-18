/**
 * Array Partition
 * Problem ID: array-partition
 * Order: 5
 */

import { Problem } from '../../../types';

export const array_partitionProblem: Problem = {
  id: 'array-partition',
  title: 'Array Partition',
  difficulty: 'Medium',
  topic: 'Time & Space Complexity',
  order: 5,
  description: `Given an integer array \`nums\` of \`2n\` integers, group these integers into \`n\` pairs \`(a1, b1), (a2, b2), ..., (an, bn)\` such that the sum of \`min(ai, bi)\` for all \`i\` is **maximized**. Return the maximized sum.

**Key Insight:** To maximize the sum of minimums, pair adjacent elements after sorting!`,
  examples: [
    {
      input: 'nums = [1,4,3,2]',
      output: '4',
      explanation:
        'All possible pairings:\n1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3\n2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3\n3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4 (maximum)',
    },
    {
      input: 'nums = [6,2,6,5,1,2]',
      output: '9',
      explanation:
        'Optimal pairing: (2, 1), (2, 5), (6, 6). Sum of mins = 1 + 2 + 6 = 9.',
    },
  ],
  constraints: [
    '1 <= n <= 10^4',
    'nums.length == 2 * n',
    '-10^4 <= nums[i] <= 10^4',
  ],
  hints: [
    'To maximize sum of minimums, avoid wasting large numbers',
    'Sort the array first',
    'After sorting, pair adjacent elements',
  ],
  starterCode: `from typing import List

def array_pair_sum(nums: List[int]) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 4, 3, 2]],
      expected: 4,
    },
    {
      input: [[6, 2, 6, 5, 1, 2]],
      expected: 9,
    },
    {
      input: [[1, 2]],
      expected: 1,
    },
    {
      input: [[0, 0, 0, 0]],
      expected: 0,
    },
    {
      input: [[-1, -2, -3, -4]],
      expected: -4,
    },
  ],
  solution: `# Optimal: Sort and sum alternating elements - O(n log n) time, O(1) space
def array_pair_sum(nums):
    nums.sort()
    return sum(nums[::2])  # Sum elements at even indices

# Expanded version
def array_pair_sum_verbose(nums):
    nums.sort()
    result = 0
    for i in range(0, len(nums), 2):
        result += nums[i]
    return result
`,
  timeComplexity: 'O(n log n) - dominated by sorting',
  spaceComplexity: 'O(1) or O(log n) depending on sorting algorithm',
  leetcodeUrl: 'https://leetcode.com/problems/array-partition/',
};
