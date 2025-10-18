/**
 * Wiggle Sort
 * Problem ID: wiggle-sort
 * Order: 5
 */

import { Problem } from '../../../types';

export const wiggle_sortProblem: Problem = {
  id: 'wiggle-sort',
  title: 'Wiggle Sort',
  difficulty: 'Medium',
  topic: 'Sorting Algorithms',
  order: 5,
  description: `Given an integer array \`nums\`, reorder it such that \`nums[0] <= nums[1] >= nums[2] <= nums[3]...\`

You may assume the input array always has a valid answer.

**Challenge:** Can you do it in O(n) time without fully sorting?`,
  examples: [
    {
      input: 'nums = [3,5,2,1,6,4]',
      output: '[3,5,1,6,2,4]',
      explanation: '[1,6,2,5,3,4] is also accepted.',
    },
    {
      input: 'nums = [6,6,5,6,3,8]',
      output: '[6,6,5,6,3,8]',
    },
  ],
  constraints: ['1 <= nums.length <= 5 * 10^4', '0 <= nums[i] <= 10^4'],
  hints: [
    'Naive: sort then swap adjacent pairs',
    'Optimal: ensure even indices are <= next, odd indices are >= next',
    'One pass with local swaps is enough!',
  ],
  starterCode: `from typing import List

def wiggle_sort(nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[3, 5, 2, 1, 6, 4]]],
      expected: [3, 5, 1, 6, 2, 4],
    },
    {
      input: [[[6, 6, 5, 6, 3, 8]]],
      expected: [6, 6, 5, 6, 3, 8],
    },
    {
      input: [[[1, 2, 3]]],
      expected: [1, 3, 2],
    },
    {
      input: [[[1, 1, 1]]],
      expected: [1, 1, 1],
    },
    {
      input: [[[5, 3, 1, 2, 6, 7, 8, 5, 5]]],
      expected: [3, 5, 1, 6, 2, 7, 5, 8, 5],
    },
  ],
  solution: `# Optimal: O(n) time, O(1) space
def wiggle_sort(nums):
    for i in range(len(nums) - 1):
        if i % 2 == 0:
            # Even index: should be <= next
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        else:
            # Odd index: should be >= next
            if nums[i] < nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
    
    return nums  # For testing

# Naive approach: O(n log n) time
def wiggle_sort_sort(nums):
    nums.sort()
    
    # Swap pairs: 0-1, 2-3, 4-5, etc.
    for i in range(1, len(nums) - 1, 2):
        nums[i], nums[i + 1] = nums[i + 1], nums[i]
    
    return nums
`,
  timeComplexity:
    'O(n) optimal vs O(n log n) sorting - demonstrates avoiding unnecessary sorting',
  spaceComplexity: 'O(1) - in-place with local swaps',
  leetcodeUrl: 'https://leetcode.com/problems/wiggle-sort/',
};
