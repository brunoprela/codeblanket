/**
 * Sort Array By Parity
 * Problem ID: sort-array-parity
 * Order: 2
 */

import { Problem } from '../../../types';

export const sort_array_parityProblem: Problem = {
  id: 'sort-array-parity',
  title: 'Sort Array By Parity',
  difficulty: 'Easy',
  topic: 'Sorting Algorithms',

  leetcodeUrl: 'https://leetcode.com/problems/sort-array-by-parity/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6YZn-z5jkrg',
  order: 2,
  description: `Given an integer array \`nums\`, move all the even integers at the beginning of the array followed by all the odd integers.

Return **any array** that satisfies this condition.

**Challenge:** Can you do it in-place with O(1) extra space?`,
  examples: [
    {
      input: 'nums = [3,1,2,4]',
      output: '[2,4,3,1]',
      explanation:
        'The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.',
    },
    {
      input: 'nums = [0]',
      output: '[0]',
    },
  ],
  constraints: ['1 <= nums.length <= 5000', '0 <= nums[i] <= 5000'],
  hints: [
    'Two pointers: one at the start, one at the end',
    'Swap when left is odd and right is even',
    'Similar to the partition step in quicksort',
  ],
  starterCode: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 1, 2, 4]],
      expected: [2, 4, 3, 1],
    },
    {
      input: [[0]],
      expected: [0],
    },
    {
      input: [[1, 2, 3, 4]],
      expected: [2, 4, 1, 3],
    },
    {
      input: [[2, 4, 6]],
      expected: [2, 4, 6],
    },
    {
      input: [[1, 3, 5]],
      expected: [1, 3, 5],
    },
  ],
  solution: `# Two-pointer in-place: O(n) time, O(1) space
def sort_array_by_parity(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        # If left is even, move forward
        if nums[left] % 2 == 0:
            left += 1
        # If right is odd, move backward
        elif nums[right] % 2 == 1:
            right -= 1
        # Both in wrong position, swap
        else:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
    
    return nums

# Alternative: create new array (easier but uses O(n) space)
def sort_array_by_parity_v2(nums):
    even = [x for x in nums if x % 2 == 0]
    odd = [x for x in nums if x % 2 == 1]
    return even + odd

# Using Python's sort with custom key
def sort_array_by_parity_v3(nums):
    return sorted(nums, key=lambda x: x % 2)
`,
  timeComplexity: 'O(n) - single pass with two pointers',
  spaceComplexity: 'O(1) for in-place solution, O(n) for creating new arrays',
};
