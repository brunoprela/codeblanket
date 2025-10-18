/**
 * Single Number
 * Problem ID: fundamentals-find-single-number
 * Order: 26
 */

import { Problem } from '../../../types';

export const find_single_numberProblem: Problem = {
  id: 'fundamentals-find-single-number',
  title: 'Single Number',
  difficulty: 'Easy',
  description: `Find the element that appears once in an array where every other element appears twice.

**Must use O(1) extra space.**

**Example:** [2, 2, 1] → 1

This problem tests:
- XOR properties
- Bit manipulation basics
- Space optimization`,
  examples: [
    {
      input: 'nums = [2, 2, 1]',
      output: '1',
    },
    {
      input: 'nums = [4, 1, 2, 1, 2]',
      output: '4',
    },
    {
      input: 'nums = [1]',
      output: '1',
    },
  ],
  constraints: ['Each element appears twice except one', 'Must use O(1) space'],
  hints: [
    'XOR has special properties: a ^ a = 0, a ^ 0 = a',
    'XOR all numbers together',
    'Pairs cancel out, leaving the single number',
  ],
  starterCode: `def single_number(nums):
    """
    Find the number that appears once.
    
    Args:
        nums: List where each element appears twice except one
        
    Returns:
        The single number
        
    Examples:
        >>> single_number([2, 2, 1])
        1
        >>> single_number([4, 1, 2, 1, 2])
        4
    """
    pass`,
  testCases: [
    {
      input: [[2, 2, 1]],
      expected: 1,
    },
    {
      input: [[4, 1, 2, 1, 2]],
      expected: 4,
    },
    {
      input: [[1]],
      expected: 1,
    },
  ],
  solution: `def single_number(nums):
    # XOR all numbers together
    # Numbers appearing twice cancel out (a ^ a = 0)
    # Result is the single number (a ^ 0 = a)
    result = 0
    for num in nums:
        result ^= num
    return result

# Using reduce
from functools import reduce
import operator
def single_number_reduce(nums):
    return reduce(operator.xor, nums)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 26,
  topic: 'Python Fundamentals',
};
