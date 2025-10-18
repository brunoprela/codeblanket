/**
 * Lowest Common Ancestor of a Binary Tree
 * Problem ID: lowest-common-ancestor-binary-tree
 * Order: 9
 */

import { Problem } from '../../../types';

export const lowest_common_ancestor_binary_treeProblem: Problem = {
  id: 'lowest-common-ancestor-binary-tree',
  title: 'Lowest Common Ancestor of a Binary Tree',
  difficulty: 'Medium',
  topic: 'Trees',
  description: `Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA: "The lowest common ancestor is defined between two nodes \`p\` and \`q\` as the lowest node in T that has both \`p\` and \`q\` as descendants (where we allow **a node to be a descendant of itself**)."

**Note:** This is for a **general binary tree**, not a BST. You cannot use the ordering property.`,
  examples: [
    {
      input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1',
      output: '3',
      explanation: 'The LCA of nodes 5 and 1 is 3.',
    },
    {
      input: 'root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4',
      output: '5',
      explanation:
        'The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself.',
    },
    {
      input: 'root = [1,2], p = 1, q = 2',
      output: '1',
      explanation: 'The LCA of nodes 1 and 2 is 1.',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [2, 10^5]',
    '-10^9 <= Node.val <= 10^9',
    'All Node.val are unique',
    'p != q',
    'p and q will exist in the tree',
  ],
  hints: [
    'Use recursion to search for both nodes',
    'If current node is p or q, return it',
    'If left and right subtrees both return non-null, current node is LCA',
    'Otherwise, return whichever subtree found something',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in a binary tree.
    
    Args:
        root: Root of binary tree
        p: First node
        q: Second node
        
    Returns:
        LCA node
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], 5, 1],
      expected: 3,
    },
    {
      input: [[3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], 5, 4],
      expected: 5,
    },
    {
      input: [[1, 2], 1, 2],
      expected: 1,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in a binary tree using recursive approach.
    
    Time: O(N) - might visit all nodes
    Space: O(H) - recursion depth (height of tree)
    
    Key Insight:
    - The LCA is the first node where p and q diverge into different subtrees
    - Use post-order traversal (process children before parent)
    - Return nodes upward and look for split point
    
    Algorithm:
    1. Base case: if root is None or root is p or q, return root
    2. Recursively search left and right subtrees
    3. If both subtrees return non-null → root is LCA (split point)
    4. If only one returns non-null → both nodes in that subtree
    """
    # Base case: empty tree or found one of the target nodes
    if not root or root == p or root == q:
        return root
    
    # Recursively search left and right subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    # Case 1: Found nodes in different subtrees
    # Current root is the split point (LCA)
    if left and right:
        return root
    
    # Case 2: Both nodes in one subtree
    # Return whichever subtree found something
    return left if left else right


# Example usage and explanation
def example():
    """
    Example tree:
            3
           / \\
          5   1
         / \\ / \\
        6  2 0  8
          / \\
         7   4
    
    Example 1: LCA(5, 1)
    - Searching from 3:
      - Left subtree (5): returns 5
      - Right subtree (1): returns 1
      - Both non-null → 3 is LCA
    
    Example 2: LCA(5, 4)
    - Searching from 3:
      - Left subtree (5):
        - Finds 5 immediately (base case)
        - Returns 5
      - Right subtree (1): returns None
      - Only left non-null → return 5
    - Result: 5 (node can be ancestor of itself)
    
    Example 3: LCA(7, 4)
    - Searching from 3:
      - Left subtree (5):
        - Neither 5 is target, search children
        - Left (6): returns None
        - Right (2):
          - Left (7): returns 7
          - Right (4): returns 4
          - Both non-null → 2 is LCA
        - Right returned 2 → return 2
      - Right subtree (1): returns None
      - Only left non-null → return 2
    - Result: 2
    """
    pass


# Alternative: Iterative with Parent Pointers (if nodes have parent)
def lca_with_parent(p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA using parent pointers (if available).
    
    Time: O(H) - traverse up to root
    Space: O(H) - store ancestors in set
    
    Approach: Similar to finding intersection of two linked lists
    """
    # Store all ancestors of p
    ancestors = set()
    while p:
        ancestors.add(p)
        p = p.parent
    
    # Find first ancestor of q that's also ancestor of p
    while q:
        if q in ancestors:
            return q
        q = q.parent
    
    return None


# Comparison: BST vs Binary Tree LCA
"""
Binary Search Tree LCA:
- Can use BST ordering property
- If both nodes < root → go left
- If both nodes > root → go right
- Otherwise → root is LCA
- Time: O(H), Space: O(1) with iteration

Binary Tree LCA:
- No ordering property
- Must explore both subtrees
- Use recursive post-order traversal
- Time: O(N), Space: O(H) for recursion

Key Difference:
BST property allows us to determine direction without exploring
both subtrees, enabling iterative O(1) space solution.
"""`,
  leetcodeUrl:
    'https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=13m9ZCB8gjw',
};
