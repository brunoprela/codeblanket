/**
 * Symmetric Tree
 * Problem ID: fundamentals-symmetric-tree
 * Order: 81
 */

import { Problem } from '../../../types';

export const symmetric_treeProblem: Problem = {
  id: 'fundamentals-symmetric-tree',
  title: 'Symmetric Tree',
  difficulty: 'Easy',
  description: `Check if a binary tree is symmetric (mirror image of itself).

A tree is symmetric if left subtree is mirror of right subtree.

**Example:** [1,2,2,3,4,4,3] is symmetric

This tests:
- Tree traversal
- Mirror comparison
- Recursion`,
  examples: [
    {
      input: 'root = [1,2,2,3,4,4,3]',
      output: 'True',
    },
    {
      input: 'root = [1,2,2,null,3,null,3]',
      output: 'False',
    },
  ],
  constraints: ['0 <= number of nodes <= 1000'],
  hints: [
    'Compare left and right subtrees',
    'Check if mirror images',
    'Recursively verify symmetry',
  ],
  starterCode: `def is_symmetric(tree_array):
    """
    Check if tree is symmetric.
    
    Args:
        tree_array: Array representation of tree
        
    Returns:
        True if symmetric
        
    Examples:
        >>> is_symmetric([1,2,2,3,4,4,3])
        True
    """
    pass


# Test
print(is_symmetric([1,2,2,3,4,4,3]))
`,
  testCases: [
    {
      input: [[1, 2, 2, 3, 4, 4, 3]],
      expected: true,
    },
    {
      input: [[1, 2, 2, null, 3, null, 3]],
      expected: false,
    },
  ],
  solution: `def is_symmetric(tree_array):
    if not tree_array or tree_array[0] is None:
        return True
    
    def is_mirror(left_idx, right_idx):
        if left_idx >= len(tree_array) and right_idx >= len(tree_array):
            return True
        
        if left_idx >= len(tree_array) or right_idx >= len(tree_array):
            return False
        
        left_val = tree_array[left_idx] if left_idx < len(tree_array) else None
        right_val = tree_array[right_idx] if right_idx < len(tree_array) else None
        
        if left_val != right_val:
            return False
        
        if left_val is None:
            return True
        
        # Compare outer and inner children
        return (is_mirror(2 * left_idx + 1, 2 * right_idx + 2) and
                is_mirror(2 * left_idx + 2, 2 * right_idx + 1))
    
    return is_mirror(1, 2)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  order: 81,
  topic: 'Python Fundamentals',
};
