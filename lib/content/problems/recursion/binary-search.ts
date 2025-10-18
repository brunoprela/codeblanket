/**
 * Binary Search (Recursive)
 * Problem ID: recursion-binary-search
 * Order: 7
 */

import { Problem } from '../../../types';

export const binary_searchProblem: Problem = {
  id: 'recursion-binary-search',
  title: 'Binary Search (Recursive)',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Implement binary search using recursion.

Given a sorted array and a target value, return the index where target is found, or -1 if not found.

**Binary Search Algorithm:**
1. Compare target with middle element
2. If equal, return middle index
3. If target is less, search left half
4. If target is greater, search right half

This is a classic divide-and-conquer algorithm.`,
  examples: [
    { input: 'arr = [1,2,3,4,5,6,7,8,9], target = 5', output: '4' },
    { input: 'arr = [1,2,3,4,5,6,7,8,9], target = 10', output: '-1' },
    { input: 'arr = [1], target = 1', output: '0' },
  ],
  constraints: [
    '0 <= arr.length <= 10⁴',
    'arr is sorted in ascending order',
    'All elements are unique',
    '-10⁴ <= arr[i], target <= 10⁴',
  ],
  hints: [
    'Use left and right pointers to define search range',
    'Base case: left > right means target not found',
    'Calculate middle: mid = (left + right) // 2',
    'Compare arr[mid] with target and recurse on appropriate half',
  ],
  starterCode: `def binary_search(arr, target):
    """
    Binary search using recursion.
    
    Args:
        arr: Sorted array of integers
        target: Value to search for
        
    Returns:
        Index of target, or -1 if not found
        
    Examples:
        >>> binary_search([1,2,3,4,5], 3)
        2
        >>> binary_search([1,2,3,4,5], 6)
        -1
    """
    pass


# Test cases
print(binary_search([1,2,3,4,5,6,7,8,9], 5))   # Expected: 4
print(binary_search([1,2,3,4,5,6,7,8,9], 10))  # Expected: -1
`,
  testCases: [
    { input: [[1, 2, 3, 4, 5, 6, 7, 8, 9], 5], expected: 4 },
    { input: [[1, 2, 3, 4, 5, 6, 7, 8, 9], 10], expected: -1 },
    { input: [[1], 1], expected: 0 },
    { input: [[1], 2], expected: -1 },
    { input: [[], 1], expected: -1 },
  ],
  solution: `def binary_search(arr, target, left=0, right=None):
    """Binary search using recursion"""
    # Initialize right on first call
    if right is None:
        right = len(arr) - 1
    
    # Base case: search space exhausted
    if left > right:
        return -1
    
    # Calculate middle
    mid = (left + right) // 2
    
    # Found target
    if arr[mid] == target:
        return mid
    
    # Target is in left half
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    
    # Target is in right half
    else:
        return binary_search(arr, target, mid + 1, right)


# Time Complexity: O(log n) - halves search space each time
# Space Complexity: O(log n) - call stack depth`,
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(log n)',
  followUp: [
    'How is this different from iterative binary search?',
    'Which is better - recursive or iterative?',
    'Can you find the first/last occurrence of a repeated element?',
  ],
};
