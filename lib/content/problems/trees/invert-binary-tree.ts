/**
 * Invert Binary Tree
 * Problem ID: invert-binary-tree
 * Order: 1
 */

import { Problem } from '../../../types';

export const invert_binary_treeProblem: Problem = {
  id: 'invert-binary-tree',
  title: 'Invert Binary Tree',
  difficulty: 'Easy',
  description: `Given the \`root\` of a binary tree, invert the tree, and return its root.

**Inverting** a binary tree means swapping the left and right children of every node in the tree.


**Approach:**
Use recursion to swap left and right children at each node. Recursively invert the left and right subtrees, then swap them.

**Note:** This is famously the problem Max Howell (creator of Homebrew) could not solve in his Google interview!`,
  examples: [
    {
      input: 'root = [4,2,7,1,3,6,9]',
      output: '[4,7,2,9,6,3,1]',
      explanation: 'The tree is inverted, all left-right children are swapped.',
    },
    {
      input: 'root = [2,1,3]',
      output: '[2,3,1]',
    },
    {
      input: 'root = []',
      output: '[]',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 100]',
    '-100 <= Node.val <= 100',
  ],
  hints: [
    'Use recursion to solve this elegantly',
    'Base case: if node is null, return null',
    'Recursively invert left and right subtrees',
    'Swap the left and right children',
    'Can also be solved iteratively using BFS or DFS',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Invert a binary tree.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        Root of the inverted tree
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[4, 2, 7, 1, 3, 6, 9]]],
      expected: [4, 7, 2, 9, 6, 3, 1],
    },
    {
      input: [[[2, 1, 3]]],
      expected: [2, 3, 1],
    },
    {
      input: [[[]]],
      expected: [],
    },
    {
      input: [[[1]]],
      expected: [1],
    },
  ],
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Recursive solution.
    Time: O(N), Space: O(H) for recursion stack
    """
    # Base case
    if not root:
        return None
    
    # Recursively invert subtrees
    left = invert_tree(root.left)
    right = invert_tree(root.right)
    
    # Swap children
    root.left = right
    root.right = left
    
    return root


# Alternative: More concise
def invert_tree_concise(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    
    # Swap and recurse in one step
    root.left, root.right = (
        invert_tree_concise(root.right),
        invert_tree_concise(root.left)
    )
    
    return root


# Alternative: Iterative BFS
def invert_tree_iterative(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Iterative solution using BFS.
    Time: O(N), Space: O(W) where W is max width
    """
    if not root:
        return None
    
    from collections import deque
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Swap children
        node.left, node.right = node.right, node.left
        
        # Add children to queue
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h) where h is the height of the tree',

  leetcodeUrl: 'https://leetcode.com/problems/invert-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=OnSn2XEQ4MY',
  order: 1,
  topic: 'Trees',
};
