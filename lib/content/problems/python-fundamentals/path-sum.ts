/**
 * Path Sum
 * Problem ID: fundamentals-path-sum
 * Order: 80
 */

import { Problem } from '../../../types';

export const path_sumProblem: Problem = {
  id: 'fundamentals-path-sum',
  title: 'Path Sum',
  difficulty: 'Easy',
  description: `Check if tree has root-to-leaf path with given sum.

Path sum = sum of all node values along the path.

**Input:** Array representation and target sum

This tests:
- Tree traversal
- Path tracking
- Sum calculation`,
  examples: [
    {
      input: 'root = [5,4,8,11,null,13,4,7,2], targetSum = 22',
      output: 'True',
      explanation: 'Path: 5→4→11→2 = 22',
    },
  ],
  constraints: ['0 <= number of nodes <= 5000', '-1000 <= Node.val <= 1000'],
  hints: [
    'Use DFS with running sum',
    'Check sum at leaf nodes',
    'Subtract current value from target',
  ],
  starterCode: `def has_path_sum(tree_array, target_sum):
    """
    Check if path exists with given sum.
    
    Args:
        tree_array: Array representation
        target_sum: Target sum value
        
    Returns:
        True if path exists
        
    Examples:
        >>> has_path_sum([5,4,8,11,None,13,4,7,2], 22)
        True
    """
    pass


# Test
print(has_path_sum([5,4,8,11,None,13,4,7,2,None,None,None,1], 22))
`,
  testCases: [
    {
      input: [[5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1], 22],
      expected: true,
    },
    {
      input: [[1, 2, 3], 5],
      expected: false,
    },
  ],
  solution: `def has_path_sum(tree_array, target_sum):
    if not tree_array or tree_array[0] is None:
        return False
    
    def dfs(idx, current_sum):
        if idx >= len(tree_array) or tree_array[idx] is None:
            return False
        
        current_sum += tree_array[idx]
        
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2
        
        # Check if leaf
        left_null = left_idx >= len(tree_array) or tree_array[left_idx] is None
        right_null = right_idx >= len(tree_array) or tree_array[right_idx] is None
        
        if left_null and right_null:
            return current_sum == target_sum
        
        return dfs(left_idx, current_sum) or dfs(right_idx, current_sum)
    
    return dfs(0, 0)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h) where h is height',
  order: 80,
  topic: 'Python Fundamentals',
};
