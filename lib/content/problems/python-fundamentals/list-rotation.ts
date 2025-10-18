/**
 * Rotate List
 * Problem ID: fundamentals-list-rotation
 * Order: 10
 */

import { Problem } from '../../../types';

export const list_rotationProblem: Problem = {
  id: 'fundamentals-list-rotation',
  title: 'Rotate List',
  difficulty: 'Easy',
  description: `Rotate a list to the right by k positions.

**Requirements:**
- Rotate the list k positions to the right
- If k is negative, rotate to the left
- If k > length, rotate k % length positions
- Modify in-place or return new list

**Example:**
- [1, 2, 3, 4, 5] rotated right by 2 → [4, 5, 1, 2, 3]
- [1, 2, 3, 4, 5] rotated left by 2 → [3, 4, 5, 1, 2]

**Visualize:**
\`\`\`
Original:  [1, 2, 3, 4, 5]
Rotate 2:  [4, 5, 1, 2, 3]
           ↑     ↑
           |_____|
\`\`\``,
  examples: [
    {
      input: 'arr = [1, 2, 3, 4, 5], k = 2',
      output: '[4, 5, 1, 2, 3]',
    },
  ],
  constraints: ['1 <= list length <= 10^5', '-10^9 <= k <= 10^9'],
  hints: [
    'Handle k > length using modulo',
    'Use list slicing for simple solution',
    'For in-place: reverse entire array, then reverse two parts',
  ],
  starterCode: `def rotate_list(arr, k):
    """
    Rotate list to the right by k positions.
    
    Args:
        arr: List to rotate
        k: Number of positions (positive = right, negative = left)
        
    Returns:
        Rotated list
        
    Examples:
        >>> rotate_list([1, 2, 3, 4, 5], 2)
        [4, 5, 1, 2, 3]
        >>> rotate_list([1, 2, 3, 4, 5], -2)
        [3, 4, 5, 1, 2]
    """
    pass


# Test
print(rotate_list([1, 2, 3, 4, 5], 2))
print(rotate_list([1, 2, 3, 4, 5], -2))
print(rotate_list([1, 2, 3, 4, 5], 7))  # Same as rotating by 2
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5], 2],
      expected: [4, 5, 1, 2, 3],
    },
    {
      input: [[1, 2, 3, 4, 5], -2],
      expected: [3, 4, 5, 1, 2],
    },
    {
      input: [[1, 2, 3], 5],
      expected: [2, 3, 1],
    },
  ],
  solution: `def rotate_list(arr, k):
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n  # Handle k > n
    
    # Slice and concatenate
    return arr[-k:] + arr[:-k] if k else arr


# In-place rotation using reverse
def rotate_list_inplace(arr, k):
    if not arr:
        return arr
    
    n = len(arr)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    # Reverse entire array
    reverse(0, n - 1)
    # Reverse first k elements
    reverse(0, k - 1)
    # Reverse remaining elements
    reverse(k, n - 1)
    
    return arr


# Using deque.rotate
from collections import deque

def rotate_list_deque(arr, k):
    d = deque(arr)
    d.rotate(k)
    return list(d)`,
  timeComplexity: 'O(n) with slicing, O(n) in-place',
  spaceComplexity: 'O(n) with slicing, O(1) in-place',
  order: 10,
  topic: 'Python Fundamentals',
};
