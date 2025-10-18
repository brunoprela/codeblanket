/**
 * Find All Numbers Disappeared in an Array
 * Problem ID: find-disappeared-numbers
 * Order: 8
 */

import { Problem } from '../../../types';

export const find_disappeared_numbersProblem: Problem = {
  id: 'find-disappeared-numbers',
  title: 'Find All Numbers Disappeared in an Array',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 8,
  description: `Given an array \`nums\` of \`n\` integers where \`nums[i]\` is in the range \`[1, n]\`, return an array of all the integers in the range \`[1, n]\` that do not appear in \`nums\`.`,
  examples: [
    {
      input: 'nums = [4,3,2,7,8,2,3,1]',
      output: '[5,6]',
    },
    {
      input: 'nums = [1,1]',
      output: '[2]',
    },
  ],
  constraints: ['n == nums.length', '1 <= n <= 10^5', '1 <= nums[i] <= n'],
  hints: [
    'Use a set to track which numbers appear',
    'Then check which numbers from [1, n] are missing',
    'Can you do it without extra space by marking in-place?',
  ],
  starterCode: `from typing import List

def find_disappeared_numbers(nums: List[int]) -> List[int]:
    """
    Find all numbers that disappeared.
    
    Args:
        nums: Array with some numbers in [1, n]
        
    Returns:
        List of missing numbers
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 3, 2, 7, 8, 2, 3, 1]],
      expected: [5, 6],
    },
    {
      input: [[1, 1]],
      expected: [2],
    },
    {
      input: [[1, 2, 3, 4, 5]],
      expected: [],
    },
  ],
  solution: `from typing import List

def find_disappeared_numbers(nums: List[int]) -> List[int]:
    """
    Set approach to find missing numbers.
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    result = []
    
    for i in range(1, len(nums) + 1):
        if i not in num_set:
            result.append(i)
    
    return result

# Alternative: In-place marking
def find_disappeared_numbers_inplace(nums: List[int]) -> List[int]:
    """
    Mark presence by negating values at indices.
    Time: O(n), Space: O(1)
    """
    # Mark presence by negating value at index
    for num in nums:
        index = abs(num) - 1
        if nums[index] > 0:
            nums[index] = -nums[index]
    
    # Positive indices indicate missing numbers
    result = []
    for i in range(len(nums)):
        if nums[i] > 0:
            result.append(i + 1)
    
    return result
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n) for set, O(1) for in-place',
  leetcodeUrl:
    'https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=8i-f24YFWC4',
};
