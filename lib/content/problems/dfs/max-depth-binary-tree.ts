/**
 * Maximum Depth of Binary Tree
 * Problem ID: max-depth-binary-tree
 * Order: 1
 */

import { Problem } from '../../../types';

export const max_depth_binary_treeProblem: Problem = {
  id: 'max-depth-binary-tree',
  title: 'Maximum Depth of Binary Tree',
  difficulty: 'Easy',
  description: `Given the \`root\` of a binary tree, return its **maximum depth**.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.


**Approach:**
Use **DFS (Depth-First Search)** to recursively calculate the depth of left and right subtrees. The maximum depth is 1 (current node) plus the maximum of left and right subtree depths.

**Base Case:** If the node is null, return 0.

**Recursive Case:** Return 1 + max(left_depth, right_depth)

**Key Insight:**
This is a classic bottom-up DFS where we return information from children up to the parent.`,
  examples: [
    {
      input: 'root = [3,9,20,null,null,15,7]',
      output: '3',
      explanation: 'The maximum depth is 3 (3 → 20 → 7 or 3 → 20 → 15)',
    },
    {
      input: 'root = [1,null,2]',
      output: '2',
      explanation: 'The maximum depth is 2 (1 → 2)',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 10^4]',
    '-100 <= Node.val <= 100',
  ],
  hints: [
    'Think recursively: what information do children return?',
    'Base case: what is the depth of an empty tree?',
    "Recursive case: how does parent use children's depths?",
    'This is a postorder traversal (process children first)',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root: Optional[TreeNode]) -> int:
    """
    Return the maximum depth of the binary tree.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        Maximum depth as an integer
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 9, 20, null, null, 15, 7]],
      expected: 3,
    },
    {
      input: [[1, null, 2]],
      expected: 2,
    },
    {
      input: [[]],
      expected: 0,
    },
  ],
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Recursive DFS solution.
    Time: O(N), Space: O(H) where H is height
    """
    # Base case: empty tree has depth 0
    if not root:
        return 0
    
    # Recursively get depth of left and right subtrees
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    # Current depth is 1 + max of children
    return 1 + max(left_depth, right_depth)


# Alternative: Iterative DFS with stack
def max_depth_iterative(root: Optional[TreeNode]) -> int:
    """
    Iterative DFS using stack of (node, depth) pairs.
    Time: O(N), Space: O(H)
    """
    if not root:
        return 0
    
    stack = [(root, 1)]
    max_depth = 0
    
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        
        if node.left:
            stack.append((node.left, depth + 1))
        if node.right:
            stack.append((node.right, depth + 1))
    
    return max_depth`,
  timeComplexity: 'O(N)',
  spaceComplexity: 'O(H) where H is height',

  leetcodeUrl: 'https://leetcode.com/problems/maximum-depth-of-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=hTM3phVI6YQ',
  order: 1,
  topic: 'Depth-First Search (DFS)',
};
