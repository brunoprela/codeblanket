/**
 * Find Peak Element
 * Problem ID: fundamentals-find-peak
 * Order: 71
 */

import { Problem } from '../../../types';

export const find_peakProblem: Problem = {
  id: 'fundamentals-find-peak',
  title: 'Find Peak Element',
  difficulty: 'Medium',
  description: `Find a peak element in an array.

A peak element is greater than its neighbors.
- arr[i] > arr[i-1] and arr[i] > arr[i+1]
- Edge elements only compare with one neighbor

**Example:** [1,2,3,1] → peak at index 2 (value 3)

This tests:
- Array traversal
- Binary search (advanced)
- Neighbor comparison`,
  examples: [
    {
      input: 'nums = [1,2,3,1]',
      output: '2',
      explanation: 'Peak at index 2',
    },
    {
      input: 'nums = [1,2,1,3,5,6,4]',
      output: '5',
      explanation: 'Peak at index 5',
    },
  ],
  constraints: ['1 <= len(nums) <= 1000'],
  hints: [
    'Linear scan is O(n)',
    'Binary search is O(log n)',
    'Compare with next element',
  ],
  starterCode: `def find_peak_element(nums):
    """
    Find a peak element index.
    
    Args:
        nums: Array of integers
        
    Returns:
        Index of any peak element
        
    Examples:
        >>> find_peak_element([1,2,3,1])
        2
    """
    pass


# Test
print(find_peak_element([1,2,3,1]))
`,
  testCases: [
    {
      input: [[1, 2, 3, 1]],
      expected: 2,
    },
    {
      input: [[1, 2, 1, 3, 5, 6, 4]],
      expected: 5,
    },
  ],
  solution: `def find_peak_element(nums):
    for i in range(len(nums) - 1):
        if nums[i] > nums[i + 1]:
            return i
    return len(nums) - 1


# Binary search O(log n) solution
def find_peak_element_binary(nums):
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left`,
  timeComplexity: 'O(n) or O(log n) with binary search',
  spaceComplexity: 'O(1)',
  order: 71,
  topic: 'Python Fundamentals',
};
