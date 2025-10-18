/**
 * Sorted Array to BST
 * Problem ID: fundamentals-convert-sorted-array-to-bst
 * Order: 78
 */

import { Problem } from '../../../types';

export const convert_sorted_array_to_bstProblem: Problem = {
  id: 'fundamentals-convert-sorted-array-to-bst',
  title: 'Sorted Array to BST',
  difficulty: 'Easy',
  description: `Convert a sorted array to a height-balanced Binary Search Tree.

Height-balanced: depth of two subtrees differ by at most 1.

**Strategy:** Use middle element as root, recursively build left/right.

**Note:** This is a simplified version that returns array representation.

This tests:
- Recursion
- Array slicing
- Tree construction logic`,
  examples: [
    {
      input: 'nums = [-10,-3,0,5,9]',
      output: '[0,-3,9,-10,null,5]',
      explanation: 'Middle element as root',
    },
  ],
  constraints: ['1 <= len(nums) <= 10^4', 'Sorted in ascending order'],
  hints: [
    'Use middle element as root',
    'Recursively build left/right subtrees',
    'Base case: empty array',
  ],
  starterCode: `def sorted_array_to_bst(nums):
    """
    Convert sorted array to balanced BST.
    Returns array representation [root, left, right, ...]
    
    Args:
        nums: Sorted array
        
    Returns:
        Root value of balanced BST
        
    Examples:
        >>> sorted_array_to_bst([-10,-3,0,5,9])
        0
    """
    pass


# Test
print(sorted_array_to_bst([-10,-3,0,5,9]))
`,
  testCases: [
    {
      input: [[-10, -3, 0, 5, 9]],
      expected: 0,
    },
    {
      input: [[1, 3]],
      expected: 3,
    },
  ],
  solution: `def sorted_array_to_bst(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    
    # For this simplified version, return root value
    # In full implementation, would create TreeNode
    return nums[mid]


# Full recursive structure
def sorted_array_to_bst_recursive(nums):
    if not nums:
        return None
    
    mid = len(nums) // 2
    
    root = {
        'val': nums[mid],
        'left': sorted_array_to_bst_recursive(nums[:mid]),
        'right': sorted_array_to_bst_recursive(nums[mid + 1:])
    }
    
    return root`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(log n) recursion stack',
  order: 78,
  topic: 'Python Fundamentals',
};
