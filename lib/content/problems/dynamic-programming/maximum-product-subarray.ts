/**
 * Maximum Product Subarray
 * Problem ID: maximum-product-subarray
 * Order: 7
 */

import { Problem } from '../../../types';

export const maximum_product_subarrayProblem: Problem = {
  id: 'maximum-product-subarray',
  title: 'Maximum Product Subarray',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `Given an integer array \`nums\`, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a **32-bit** integer.`,
  examples: [
    {
      input: 'nums = [2,3,-2,4]',
      output: '6',
      explanation: 'Subarray [2,3] has the largest product 6.',
    },
    {
      input: 'nums = [-2,0,-1]',
      output: '0',
      explanation: 'The result cannot be 2, because [-2,-1] is not a subarray.',
    },
  ],
  constraints: [
    '1 <= nums.length <= 2 * 10^4',
    '-10 <= nums[i] <= 10',
    'The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer',
  ],
  hints: [
    'Track both max and min products',
    'Negative * negative can become max',
    'Handle zero by resetting',
  ],
  starterCode: `from typing import List

def max_product(nums: List[int]) -> int:
    """
    Find maximum product of contiguous subarray.
    
    Args:
        nums: Input array
        
    Returns:
        Maximum product
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 3, -2, 4]],
      expected: 6,
    },
    {
      input: [[-2, 0, -1]],
      expected: 0,
    },
    {
      input: [[-2, 3, -4]],
      expected: 24,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/maximum-product-subarray/',
  youtubeUrl: 'https://www.youtube.com/watch?v=lXVy6YWFcRM',
};
