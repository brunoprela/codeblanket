/**
 * Concatenation of Array
 * Problem ID: concatenation-of-array
 * Order: 7
 */

import { Problem } from '../../../types';

export const concatenation_of_arrayProblem: Problem = {
  id: 'concatenation-of-array',
  title: 'Concatenation of Array',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 7,
  description: `Given an integer array \`nums\` of length \`n\`, you want to create an array \`ans\` of length \`2n\` where \`ans[i] == nums[i]\` and \`ans[i + n] == nums[i]\` for \`0 <= i < n\` (**0-indexed**).

Specifically, \`ans\` is the **concatenation** of two \`nums\` arrays.

Return the array \`ans\`.`,
  examples: [
    {
      input: 'nums = [1,2,1]',
      output: '[1,2,1,1,2,1]',
      explanation:
        'The array ans is formed as follows: ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]] = [1,2,1,1,2,1]',
    },
    {
      input: 'nums = [1,3,2,1]',
      output: '[1,3,2,1,1,3,2,1]',
    },
  ],
  constraints: ['n == nums.length', '1 <= n <= 1000', '1 <= nums[i] <= 1000'],
  hints: [
    'Create a new array of size 2n',
    'Copy elements from nums twice',
    'Or use array concatenation built-ins',
  ],
  starterCode: `from typing import List

def get_concatenation(nums: List[int]) -> List[int]:
    """
    Concatenate array with itself.
    
    Args:
        nums: Input array
        
    Returns:
        Array concatenated with itself
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 1]],
      expected: [1, 2, 1, 1, 2, 1],
    },
    {
      input: [[1, 3, 2, 1]],
      expected: [1, 3, 2, 1, 1, 3, 2, 1],
    },
    {
      input: [[1]],
      expected: [1, 1],
    },
  ],
  solution: `from typing import List

def get_concatenation(nums: List[int]) -> List[int]:
    """
    Simple concatenation approach.
    Time: O(n), Space: O(n)
    """
    return nums + nums

# Alternative: Manual approach
def get_concatenation_manual(nums: List[int]) -> List[int]:
    """
    Manually build result array.
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    ans = [0] * (2 * n)
    
    for i in range(n):
        ans[i] = nums[i]
        ans[i + n] = nums[i]
    
    return ans

# Alternative: List comprehension
def get_concatenation_comprehension(nums: List[int]) -> List[int]:
    """
    Using list comprehension.
    Time: O(n), Space: O(n)
    """
    return [nums[i % len(nums)] for i in range(2 * len(nums))]
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/concatenation-of-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=68a1Dc_qVq4',
};
