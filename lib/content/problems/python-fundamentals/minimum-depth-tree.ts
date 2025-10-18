/**
 * Minimum Depth of Binary Tree
 * Problem ID: fundamentals-minimum-depth-tree
 * Order: 79
 */

import { Problem } from '../../../types';

export const minimum_depth_treeProblem: Problem = {
  id: 'fundamentals-minimum-depth-tree',
  title: 'Minimum Depth of Binary Tree',
  difficulty: 'Easy',
  description: `Find minimum depth of a binary tree.

Minimum depth = shortest path from root to a leaf node.

**Input Format:** Array representation [root, left, right, ...]
null represents missing node.

This tests:
- Tree traversal
- BFS or DFS
- Base case handling`,
  examples: [
    {
      input: 'root = [3,9,20,null,null,15,7]',
      output: '2',
      explanation: 'Path: 3 → 9',
    },
  ],
  constraints: ['0 <= number of nodes <= 10^5'],
  hints: [
    'Use BFS for level-order traversal',
    'Return depth when leaf found',
    'Leaf = both children are null',
  ],
  starterCode: `def min_depth(tree_array):
    """
    Find minimum depth of binary tree.
    
    Args:
        tree_array: Array representation of tree
        
    Returns:
        Minimum depth
        
    Examples:
        >>> min_depth([3,9,20,None,None,15,7])
        2
    """
    pass


# Test
print(min_depth([3,9,20,None,None,15,7]))
`,
  testCases: [
    {
      input: [[3, 9, 20, null, null, 15, 7]],
      expected: 2,
    },
    {
      input: [[2, null, 3, null, 4, null, 5, null, 6]],
      expected: 5,
    },
  ],
  solution: `def min_depth(tree_array):
    if not tree_array or tree_array[0] is None:
        return 0
    
    # BFS approach
    from collections import deque
    queue = deque([(0, 1)])  # (index, depth)
    
    while queue:
        idx, depth = queue.popleft()
        
        if idx >= len(tree_array) or tree_array[idx] is None:
            continue
        
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        
        # Check if leaf
        left_null = left_idx >= len(tree_array) or tree_array[left_idx] is None
        right_null = right_idx >= len(tree_array) or tree_array[right_idx] is None
        
        if left_null and right_null:
            return depth
        
        if not left_null:
            queue.append((left_idx, depth + 1))
        if not right_null:
            queue.append((right_idx, depth + 1))
    
    return 0`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 79,
  topic: 'Python Fundamentals',
};
