/**
 * Find Pivot Index
 * Problem ID: pivot-index
 * Order: 2
 */

import { Problem } from '../../../types';

export const pivot_indexProblem: Problem = {
  id: 'pivot-index',
  title: 'Find Pivot Index',
  difficulty: 'Easy',
  topic: 'Time & Space Complexity',
  order: 2,
  description: `Given an array of integers \`nums\`, calculate the **pivot index** of this array.

The pivot index is the index where the sum of all the numbers **strictly** to the left of the index is equal to the sum of all the numbers **strictly** to the index's right.

If the index is on the left edge of the array, then the left sum is \`0\` because there are no elements to the left. This also applies to the right edge of the array.

Return the **leftmost pivot index**. If no such index exists, return \`-1\`.`,
  examples: [
    {
      input: 'nums = [1,7,3,6,5,6]',
      output: '3',
      explanation:
        'The pivot index is 3. Left sum = 1 + 7 + 3 = 11. Right sum = 5 + 6 = 11.',
    },
    {
      input: 'nums = [1,2,3]',
      output: '-1',
      explanation: 'There is no index that satisfies the conditions.',
    },
    {
      input: 'nums = [2,1,-1]',
      output: '0',
      explanation:
        'The pivot index is 0. Left sum = 0. Right sum = 1 + (-1) = 0.',
    },
  ],
  constraints: ['1 <= nums.length <= 10^4', '-1000 <= nums[i] <= 1000'],
  hints: [
    'Can you compute left and right sums for each index? What complexity?',
    'Better: compute total sum once, then track left sum as you iterate',
    'At each position: right_sum = total_sum - left_sum - nums[i]',
  ],
  starterCode: `from typing import List

def pivot_index(nums: List[int]) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 7, 3, 6, 5, 6]],
      expected: 3,
    },
    {
      input: [[1, 2, 3]],
      expected: -1,
    },
    {
      input: [[2, 1, -1]],
      expected: 0,
    },
    {
      input: [[1, 1, 1, 1, 1, 1]],
      expected: -1,
    },
    {
      input: [[-1, -1, -1, -1, -1, 0]],
      expected: 2,
    },
  ],
  solution: `# Optimal: O(n) time, O(1) space
def pivot_index(nums):
    total_sum = sum(nums)
    left_sum = 0
    
    for i in range(len(nums)):
        # Right sum = total - left - current
        right_sum = total_sum - left_sum - nums[i]
        
        if left_sum == right_sum:
            return i
        
        left_sum += nums[i]
    
    return -1

# Naive: Compute sums for each index - O(n²) time
def pivot_index_naive(nums):
    for i in range(len(nums)):
        left_sum = sum(nums[:i])
        right_sum = sum(nums[i+1:])
        if left_sum == right_sum:
            return i
    return -1
`,
  timeComplexity:
    'O(n) optimal vs O(n²) naive - demonstrates precomputation optimization',
  spaceComplexity: 'O(1) - only tracking sums',
  leetcodeUrl: 'https://leetcode.com/problems/find-pivot-index/',
  youtubeUrl: 'https://www.youtube.com/watch?v=u89i60lYx8U',
};
