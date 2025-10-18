/**
 * Sum of Array
 * Problem ID: recursion-sum-array
 * Order: 3
 */

import { Problem } from '../../../types';

export const sum_arrayProblem: Problem = {
  id: 'recursion-sum-array',
  title: 'Sum of Array',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Calculate the sum of all elements in an array using recursion.

You cannot use loops or built-in sum() function - must use recursion!

**Approach:**
- Process the array element by element
- Base case: empty array returns 0
- Recursive case: first element + sum of rest`,
  examples: [
    { input: 'arr = [1, 2, 3, 4, 5]', output: '15' },
    { input: 'arr = []', output: '0' },
    { input: 'arr = [10]', output: '10' },
  ],
  constraints: ['0 <= arr.length <= 1000', '-1000 <= arr[i] <= 1000'],
  hints: [
    'Base case: if array is empty, return 0',
    'Recursive case: first element + sum(rest of array)',
    'You can use array slicing: arr[1:] for "rest of array"',
    'Alternative: use index parameter to avoid creating new arrays',
  ],
  starterCode: `def sum_array(arr):
    """
    Calculate sum of array using recursion.
    
    Args:
        arr: List of integers
        
    Returns:
        Sum of all elements
        
    Examples:
        >>> sum_array([1, 2, 3, 4, 5])
        15
        >>> sum_array([])
        0
    """
    pass


# Test cases
print(sum_array([1, 2, 3, 4, 5]))  # Expected: 15
print(sum_array([]))  # Expected: 0
`,
  testCases: [
    { input: [[1, 2, 3, 4, 5]], expected: 15 },
    { input: [[]], expected: 0 },
    { input: [[10]], expected: 10 },
    { input: [[-1, -2, -3]], expected: -6 },
    { input: [[100, -50, 25]], expected: 75 },
  ],
  solution: `def sum_array(arr):
    """Sum array using recursion - Method 1: Array slicing"""
    # Base case: empty array
    if len(arr) == 0:
        return 0
    
    # Recursive case: first element + sum of rest
    return arr[0] + sum_array(arr[1:])


# More efficient method using index (avoids array copying):
def sum_array_index(arr, index=0):
    """Sum array using recursion - Method 2: Index tracking"""
    # Base case: reached end of array
    if index >= len(arr):
        return 0
    
    # Recursive case: current element + sum of rest
    return arr[index] + sum_array_index(arr, index + 1)


# Time Complexity: O(n) - processes each element once
# Space Complexity: O(n) - call stack depth
# Note: Method 1 also has O(n) for array slicing`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  followUp: [
    'Which approach is more efficient - slicing or index?',
    'How would you sum a 2D array recursively?',
    'Can you make it tail recursive?',
  ],
};
