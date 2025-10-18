/**
 * Two Sum
 * Problem ID: two-sum
 * Order: 2
 */

import { Problem } from '../../../types';

export const two_sumProblem: Problem = {
  id: 'two-sum',
  title: 'Two Sum',
  difficulty: 'Medium',
  description: `Given an array of integers \`nums\` and an integer \`target\`, return **indices** of the two numbers such that they add up to \`target\`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.


**Approach:**
Use a hash map to store each number and its index. For each number, check if its complement (target - num) exists in the map.`,
  examples: [
    {
      input: 'nums = [2,7,11,15], target = 9',
      output: '[0,1]',
      explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].',
    },
    {
      input: 'nums = [3,2,4], target = 6',
      output: '[1,2]',
      explanation: 'nums[1] + nums[2] == 6',
    },
    {
      input: 'nums = [3,3], target = 6',
      output: '[0,1]',
      explanation: 'Both elements sum to the target.',
    },
  ],
  constraints: [
    '2 <= nums.length <= 10^4',
    '-10^9 <= nums[i] <= 10^9',
    '-10^9 <= target <= 10^9',
    'Only one valid answer exists',
  ],
  hints: [
    'For each number, calculate what its complement should be (target - num)',
    'Use a hash map to check if the complement exists in O(1) time',
    'Store both the number and its index in the hash map',
    "Don't use the same element twice",
  ],
  starterCode: `from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
        
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] == target
    """
    # Your code here
    pass
`,
  testCases: [
    {
      input: [[2, 7, 11, 15], 9],
      expected: [0, 1],
    },
    {
      input: [[3, 2, 4], 6],
      expected: [1, 2],
    },
    {
      input: [[3, 3], 6],
      expected: [0, 1],
    },
    {
      input: [[1, 2, 3, 4, 5], 9],
      expected: [3, 4],
    },
  ],
  solution: `def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []  # Should never reach if input is valid`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/two-sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=KLlXCFG5TnA',
  order: 2,
  topic: 'Arrays & Hashing',
};
