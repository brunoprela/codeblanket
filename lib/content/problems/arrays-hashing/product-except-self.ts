/**
 * Product of Array Except Self
 * Problem ID: product-except-self
 * Order: 14
 */

import { Problem } from '../../../types';

export const product_except_selfProblem: Problem = {
  id: 'product-except-self',
  title: 'Product of Array Except Self',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  order: 14,
  description: `Given an integer array \`nums\`, return an array \`answer\` such that \`answer[i]\` is equal to the product of all the elements of \`nums\` except \`nums[i]\`.

The product of any prefix or suffix of \`nums\` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in **O(n)** time and without using the division operation.`,
  examples: [
    {
      input: 'nums = [1,2,3,4]',
      output: '[24,12,8,6]',
    },
    {
      input: 'nums = [-1,1,0,-3,3]',
      output: '[0,0,9,0,0]',
    },
  ],
  constraints: ['2 <= nums.length <= 10^5', '-30 <= nums[i] <= 30'],
  hints: [
    'Use prefix and suffix products',
    'First pass: calculate prefix products',
    'Second pass: calculate suffix products and combine',
    'Can you do it with O(1) extra space?',
  ],
  starterCode: `from typing import List

def product_except_self(nums: List[int]) -> List[int]:
    """
    Calculate product of array except self.
    
    Args:
        nums: Array of integers
        
    Returns:
        Array where each element is product of all others
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4]],
      expected: [24, 12, 8, 6],
    },
    {
      input: [[-1, 1, 0, -3, 3]],
      expected: [0, 0, 9, 0, 0],
    },
  ],
  solution: `from typing import List

def product_except_self(nums: List[int]) -> List[int]:
    """
    Two-pass with prefix and suffix products.
    Time: O(n), Space: O(1) excluding output
    """
    n = len(nums)
    result = [1] * n
    
    # First pass: prefix products
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # Second pass: suffix products
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) excluding output',
  leetcodeUrl: 'https://leetcode.com/problems/product-of-array-except-self/',
  youtubeUrl: 'https://www.youtube.com/watch?v=bNvIQI2wAjk',
};
