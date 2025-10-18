/**
 * Binary Tree Maximum Path Sum
 * Problem ID: binary-tree-max-path-sum
 * Order: 3
 */

import { Problem } from '../../../types';

export const binary_tree_max_path_sumProblem: Problem = {
  id: 'binary-tree-max-path-sum',
  title: 'Binary Tree Maximum Path Sum',
  difficulty: 'Hard',
  description: `A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the \`root\` of a binary tree, return **the maximum path sum** of any **non-empty** path.


**Approach:**
Use post-order DFS. For each node, calculate:
1. Maximum path going through left child
2. Maximum path going through right child
3. Maximum path using current node as highest point (left + node + right)

Track global maximum while returning the maximum single path (for parent to use).`,
  examples: [
    {
      input: 'root = [1,2,3]',
      output: '6',
      explanation:
        'The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.',
    },
    {
      input: 'root = [-10,9,20,null,null,15,7]',
      output: '42',
      explanation:
        'The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 3 * 10^4]',
    '-1000 <= Node.val <= 1000',
  ],
  hints: [
    'Use post-order DFS to process children before parent',
    'For each node, consider: max path through left, right, or both',
    'Track global maximum separately from what you return to parent',
    'Return to parent: max single path (node + max(left, right, 0))',
    'Update global max: consider path through current node (left + node + right)',
    'Use max(0, child) to ignore negative paths',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    Find the maximum path sum in a binary tree.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        Maximum path sum
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[1, 2, 3]]],
      expected: 6,
    },
    {
      input: [[[-10, 9, 20, null, null, 15, 7]]],
      expected: 42,
    },
    {
      input: [[[-3]]],
      expected: -3,
    },
    {
      input: [[[2, -1]]],
      expected: 2,
    },
  ],
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root: Optional[TreeNode]) -> int:
    """
    Post-order DFS with global max tracking.
    Time: O(N), Space: O(H) for recursion
    """
    max_sum = [float('-inf')]  # Use list to modify in nested function
    
    def dfs(node):
        if not node:
            return 0
        
        # Get max path sum from children (ignore negative paths)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))
        
        # Update global max considering path through current node
        current_max = node.val + left_max + right_max
        max_sum[0] = max(max_sum[0], current_max)
        
        # Return max single path for parent to use
        return node.val + max(left_max, right_max)
    
    dfs(root)
    return max_sum[0]


# Alternative: Using class variable
def max_path_sum_class(root: Optional[TreeNode]) -> int:
    """
    Using class variable for cleaner syntax.
    """
    class Solution:
        def __init__(self):
            self.max_sum = float('-inf')
        
        def dfs(self, node):
            if not node:
                return 0
            
            # Recursively get max path from children
            left = max(0, self.dfs(node.left))
            right = max(0, self.dfs(node.right))
            
            # Update global max
            self.max_sum = max(self.max_sum, node.val + left + right)
            
            # Return max single path
            return node.val + max(left, right)
    
    sol = Solution()
    sol.dfs(root)
    return sol.max_sum


# Alternative: Using nonlocal
def max_path_sum_nonlocal(root: Optional[TreeNode]) -> int:
    """
    Using nonlocal keyword.
    """
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        
        # Update max considering current node as highest point
        max_sum = max(max_sum, node.val + left + right)
        
        # Return single path for parent
        return node.val + max(left, right)
    
    dfs(root)
    return max_sum`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h) where h is the height of the tree',

  leetcodeUrl: 'https://leetcode.com/problems/binary-tree-maximum-path-sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Hr5cWUld4vU',
  order: 3,
  topic: 'Trees',
};
