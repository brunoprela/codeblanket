/**
 * Product Except Self
 * Problem ID: fundamentals-list-product
 * Order: 28
 */

import { Problem } from '../../../types';

export const list_productProblem: Problem = {
  id: 'fundamentals-list-product',
  title: 'Product Except Self',
  difficulty: 'Medium',
  description: `Return an array where each element is the product of all elements except itself.

**Constraint:** Cannot use division operator.

**Example:** [1, 2, 3, 4] → [24, 12, 8, 6]

This problem tests:
- Array manipulation
- Prefix/suffix products
- Space optimization`,
  examples: [
    {
      input: 'nums = [1, 2, 3, 4]',
      output: '[24, 12, 8, 6]',
      explanation: '[2×3×4, 1×3×4, 1×2×4, 1×2×3]',
    },
    {
      input: 'nums = [-1, 1, 0, -3, 3]',
      output: '[0, 0, 9, 0, 0]',
    },
  ],
  constraints: ['2 <= len(nums) <= 10^5', 'Cannot use division'],
  hints: [
    'Calculate prefix products (left to right)',
    'Calculate suffix products (right to left)',
    'Multiply prefix and suffix for each position',
  ],
  starterCode: `def product_except_self(nums):
    """
    Return array of products except self.
    
    Args:
        nums: List of integers
        
    Returns:
        List where result[i] = product of all except nums[i]
        
    Examples:
        >>> product_except_self([1, 2, 3, 4])
        [24, 12, 8, 6]
    """
    pass`,
  testCases: [
    {
      input: [[1, 2, 3, 4]],
      expected: [24, 12, 8, 6],
    },
    {
      input: [[-1, 1, 0, -3, 3]],
      expected: [0, 0, 9, 0, 0],
    },
    {
      input: [[1, 2]],
      expected: [2, 1],
    },
  ],
  solution: `def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products (left to right)
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    
    # Calculate suffix products and multiply (right to left)
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    
    return result

# Two-pass approach with extra space
def product_except_self_verbose(nums):
    n = len(nums)
    
    # Build prefix products
    prefix = [1] * n
    for i in range(1, n):
        prefix[i] = prefix[i-1] * nums[i-1]
    
    # Build suffix products
    suffix = [1] * n
    for i in range(n-2, -1, -1):
        suffix[i] = suffix[i+1] * nums[i+1]
    
    # Multiply prefix and suffix
    return [prefix[i] * suffix[i] for i in range(n)]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) excluding output array',
  order: 28,
  topic: 'Python Fundamentals',
};
