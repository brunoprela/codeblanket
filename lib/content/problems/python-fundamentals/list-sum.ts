/**
 * Sum of List Elements
 * Problem ID: fundamentals-list-sum
 * Order: 19
 */

import { Problem } from '../../../types';

export const list_sumProblem: Problem = {
  id: 'fundamentals-list-sum',
  title: 'Sum of List Elements',
  difficulty: 'Easy',
  description: `Calculate the sum of all elements in a list without using the built-in sum() function.

This problem tests:
- Loop iteration
- Accumulator pattern
- Basic arithmetic`,
  examples: [
    {
      input: 'nums = [1, 2, 3, 4, 5]',
      output: '15',
      explanation: '1 + 2 + 3 + 4 + 5 = 15',
    },
    {
      input: 'nums = [-1, -2, 3]',
      output: '0',
      explanation: '-1 + -2 + 3 = 0',
    },
  ],
  constraints: ['0 <= len(nums) <= 10^4', 'Cannot use built-in sum()'],
  hints: [
    'Initialize a total variable to 0',
    'Loop through each element',
    'Add each element to the total',
  ],
  starterCode: `def list_sum(nums):
    """
    Calculate sum of list elements.
    
    Args:
        nums: List of numbers
        
    Returns:
        Sum of all elements
        
    Examples:
        >>> list_sum([1, 2, 3, 4, 5])
        15
    """
    pass`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5]],
      expected: 15,
    },
    {
      input: [[-1, -2, 3]],
      expected: 0,
    },
    {
      input: [[]],
      expected: 0,
    },
  ],
  solution: `def list_sum(nums):
    total = 0
    for num in nums:
        total += num
    return total

# Using reduce
from functools import reduce
def list_sum_reduce(nums):
    return reduce(lambda x, y: x + y, nums, 0)

# Recursive approach
def list_sum_recursive(nums):
    if not nums:
        return 0
    return nums[0] + list_sum_recursive(nums[1:])`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) iterative, O(n) recursive',
  order: 19,
  topic: 'Python Fundamentals',
};
