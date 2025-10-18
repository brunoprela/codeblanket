/**
 * First Missing Positive
 * Problem ID: first-missing-positive
 * Order: 19
 */

import { Problem } from '../../../types';

export const first_missing_positiveProblem: Problem = {
  id: 'first-missing-positive',
  title: 'First Missing Positive',
  difficulty: 'Hard',
  topic: 'Arrays & Hashing',
  order: 19,
  description: `Given an unsorted integer array \`nums\`, return the smallest missing positive integer.

You must implement an algorithm that runs in **O(n)** time and uses **O(1)** auxiliary space.`,
  examples: [
    {
      input: 'nums = [1,2,0]',
      output: '3',
    },
    {
      input: 'nums = [3,4,-1,1]',
      output: '2',
    },
    {
      input: 'nums = [7,8,9,11,12]',
      output: '1',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', '-2^31 <= nums[i] <= 2^31 - 1'],
  hints: [
    'Use the array itself as a hash table',
    'Place each number at its correct index: nums[i] = i + 1',
    'First index where nums[i] != i + 1 is the answer',
  ],
  starterCode: `from typing import List

def first_missing_positive(nums: List[int]) -> int:
    """
    Find smallest missing positive integer.
    
    Args:
        nums: Array of integers
        
    Returns:
        Smallest missing positive integer
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 0]],
      expected: 3,
    },
    {
      input: [[3, 4, -1, 1]],
      expected: 2,
    },
    {
      input: [[7, 8, 9, 11, 12]],
      expected: 1,
    },
  ],
  solution: `from typing import List

def first_missing_positive(nums: List[int]) -> int:
    """
    In-place cyclic sort.
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    
    # Place each positive number at its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] with nums[nums[i] - 1]
            correct_idx = nums[i] - 1
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    # All positions correct, return n + 1
    return n + 1
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/first-missing-positive/',
  youtubeUrl: 'https://www.youtube.com/watch?v=8g78yfzMlao',
};
