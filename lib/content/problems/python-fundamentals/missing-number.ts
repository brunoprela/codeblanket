/**
 * Missing Number
 * Problem ID: fundamentals-missing-number
 * Order: 24
 */

import { Problem } from '../../../types';

export const missing_numberProblem: Problem = {
  id: 'fundamentals-missing-number',
  title: 'Missing Number',
  difficulty: 'Easy',
  description: `Find the missing number in an array containing n distinct numbers from 0 to n.

**Example:** [3, 0, 1] → 2

This problem tests:
- Mathematical formulas
- Set operations
- Array manipulation`,
  examples: [
    {
      input: 'nums = [3, 0, 1]',
      output: '2',
      explanation: 'Range is [0,3], missing number is 2',
    },
    {
      input: 'nums = [0, 1]',
      output: '2',
    },
    {
      input: 'nums = [9,6,4,2,3,5,7,0,1]',
      output: '8',
    },
  ],
  constraints: ['n == len(nums)', '1 <= n <= 10^4', 'All numbers are unique'],
  hints: [
    'Sum of first n numbers: n * (n + 1) / 2',
    'Subtract sum of array from expected sum',
    'Or use XOR properties',
  ],
  starterCode: `def missing_number(nums):
    """
    Find the missing number in range [0, n].
    
    Args:
        nums: List of distinct numbers from 0 to n
        
    Returns:
        The missing number
        
    Examples:
        >>> missing_number([3, 0, 1])
        2
    """
    pass`,
  testCases: [
    {
      input: [[3, 0, 1]],
      expected: 2,
    },
    {
      input: [[0, 1]],
      expected: 2,
    },
    {
      input: [[9, 6, 4, 2, 3, 5, 7, 0, 1]],
      expected: 8,
    },
  ],
  solution: `def missing_number(nums):
    n = len(nums)
    # Expected sum of 0 to n
    expected_sum = n * (n + 1) // 2
    # Actual sum
    actual_sum = sum(nums)
    # Difference is the missing number
    return expected_sum - actual_sum

# Alternative using XOR
def missing_number_xor(nums):
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

# Alternative using set
def missing_number_set(nums):
    return (set(range(len(nums) + 1)) - set(nums)).pop()`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 24,
  topic: 'Python Fundamentals',
};
