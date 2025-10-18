/**
 * Invert Binary Tree
 * Problem ID: fundamentals-invert-binary-tree
 * Order: 82
 */

import { Problem } from '../../../types';

export const invert_binary_treeProblem: Problem = {
  id: 'fundamentals-invert-binary-tree',
  title: 'Invert Binary Tree',
  difficulty: 'Easy',
  description: `Invert a binary tree (swap left and right children).

**Example:** [4,2,7,1,3,6,9] → [4,7,2,9,6,3,1]

This tests:
- Tree traversal
- Node swapping
- Recursion or iteration`,
  examples: [
    {
      input: 'root = [4,2,7,1,3,6,9]',
      output: '[4,7,2,9,6,3,1]',
    },
  ],
  constraints: ['0 <= number of nodes <= 100'],
  hints: [
    'Swap left and right children',
    'Recursively invert subtrees',
    'Or use level-order traversal',
  ],
  starterCode: `def invert_tree(tree_array):
    """
    Invert binary tree.
    
    Args:
        tree_array: Array representation
        
    Returns:
        Inverted tree array
        
    Examples:
        >>> invert_tree([4,2,7,1,3,6,9])
        [4, 7, 2, 9, 6, 3, 1]
    """
    pass


# Test
print(invert_tree([4,2,7,1,3,6,9]))
`,
  testCases: [
    {
      input: [[4, 2, 7, 1, 3, 6, 9]],
      expected: [4, 7, 2, 9, 6, 3, 1],
    },
    {
      input: [[2, 1, 3]],
      expected: [2, 3, 1],
    },
  ],
  solution: `def invert_tree(tree_array):
    if not tree_array:
        return []
    
    result = tree_array.copy()
    
    for i in range(len(result)):
        if result[i] is not None:
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2
            
            # Swap children
            if left_idx < len(result) and right_idx < len(result):
                result[left_idx], result[right_idx] = result[right_idx], result[left_idx]
    
    return result`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 82,
  topic: 'Python Fundamentals',
};
