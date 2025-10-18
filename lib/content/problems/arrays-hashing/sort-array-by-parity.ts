/**
 * Sort Array By Parity
 * Problem ID: sort-array-by-parity
 * Order: 11
 */

import { Problem } from '../../../types';

export const sort_array_by_parityProblem: Problem = {
  id: 'sort-array-by-parity',
  title: 'Sort Array By Parity',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 11,
  description: `Given an integer array \`nums\`, move all the even integers at the beginning of the array followed by all the odd integers.

Return **any array** that satisfies this condition.`,
  examples: [
    {
      input: 'nums = [3,1,2,4]',
      output: '[2,4,3,1]',
      explanation:
        '[4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.',
    },
    {
      input: 'nums = [0]',
      output: '[0]',
    },
  ],
  constraints: ['1 <= nums.length <= 5000', '0 <= nums[i] <= 5000'],
  hints: [
    'Use two pointers approach',
    'One pointer for even position, one for odd',
    'Swap when necessary',
  ],
  starterCode: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    """
    Sort array so evens come before odds.
    
    Args:
        nums: Array of integers
        
    Returns:
        Array with evens first, then odds
    """
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
  ],
  solution: `from typing import List

def sort_array_by_parity(nums: List[int]) -> List[int]:
    """
    Two-pointer in-place sort.
    Time: O(n), Space: O(1)
    """
    left = 0
    right = len(nums) - 1
    
    while left < right:
        # If left is odd and right is even, swap
        if nums[left] % 2 > nums[right] % 2:
            nums[left], nums[right] = nums[right], nums[left]
        
        # Move left pointer if even
        if nums[left] % 2 == 0:
            left += 1
        
        # Move right pointer if odd
        if nums[right] % 2 == 1:
            right -= 1
    
    return nums
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/sort-array-by-parity/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6YZn-z5jkrg',
};
