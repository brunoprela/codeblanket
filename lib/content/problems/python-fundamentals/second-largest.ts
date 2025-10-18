/**
 * Second Largest Number
 * Problem ID: fundamentals-second-largest
 * Order: 18
 */

import { Problem } from '../../../types';

export const second_largestProblem: Problem = {
  id: 'fundamentals-second-largest',
  title: 'Second Largest Number',
  difficulty: 'Easy',
  description: `Find the second largest number in a list.

If there is no second largest (all elements are the same), return None.

This problem tests:
- List traversal
- Tracking multiple values
- Edge case handling`,
  examples: [
    {
      input: 'nums = [5, 2, 8, 1, 9]',
      output: '8',
      explanation: 'Largest is 9, second largest is 8',
    },
    {
      input: 'nums = [5, 5, 5]',
      output: 'None',
      explanation: 'All elements are the same',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^4', 'Cannot use sorting'],
  hints: [
    'Track both largest and second largest',
    'Update both values as you iterate',
    'Handle duplicates correctly',
  ],
  starterCode: `def second_largest(nums):
    """
    Find second largest number in a list.
    
    Args:
        nums: List of numbers
        
    Returns:
        Second largest number or None
        
    Examples:
        >>> second_largest([5, 2, 8, 1, 9])
        8
    """
    pass`,
  testCases: [
    {
      input: [[5, 2, 8, 1, 9]],
      expected: 8,
    },
    {
      input: [[10, 20, 5, 15]],
      expected: 15,
    },
    {
      input: [[5, 5, 5]],
      expected: null,
    },
  ],
  solution: `def second_largest(nums):
    if not nums:
        return None
    
    # Remove duplicates and sort
    unique_nums = list(set(nums))
    
    if len(unique_nums) < 2:
        return None
    
    unique_nums.sort()
    return unique_nums[-2]

# Single pass approach
def second_largest_optimized(nums):
    if len(nums) < 2:
        return None
    
    largest = second = float('-inf')
    
    for num in nums:
        if num > largest:
            second = largest
            largest = num
        elif num > second and num != largest:
            second = num
    
    return None if second == float('-inf') else second`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n) for set approach, O(1) for optimized',
  order: 18,
  topic: 'Python Fundamentals',
};
