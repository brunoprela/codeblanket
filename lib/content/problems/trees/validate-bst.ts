/**
 * Validate Binary Search Tree
 * Problem ID: validate-bst
 * Order: 2
 */

import { Problem } from '../../../types';

export const validate_bstProblem: Problem = {
  id: 'validate-bst',
  title: 'Validate Binary Search Tree',
  difficulty: 'Medium',
  description: `Given the \`root\` of a binary tree, determine if it is a **valid binary search tree (BST)**.

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.


**Approach:**
Use recursion with valid range tracking. For each node, ensure its value is within a valid range (min, max). When going left, update max. When going right, update min.

**Common Pitfall:** Don't just compare with immediate children - must validate entire subtrees!`,
  examples: [
    {
      input: 'root = [2,1,3]',
      output: 'true',
      explanation: 'This is a valid BST.',
    },
    {
      input: 'root = [5,1,4,null,null,3,6]',
      output: 'false',
      explanation: "The root's value is 5 but its right child's value is 4.",
    },
    {
      input: 'root = [5,4,6,null,null,3,7]',
      output: 'false',
      explanation:
        'The node with value 3 is in the right subtree of 5, but 3 < 5.',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 10^4]',
    '-2^31 <= Node.val <= 2^31 - 1',
  ],
  hints: [
    'Each node must be within a valid range (min, max)',
    'For left child: max becomes parent value',
    'For right child: min becomes parent value',
    'Use negative/positive infinity for initial bounds',
    'Alternative: Inorder traversal of BST should be strictly increasing',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Validate if a binary tree is a valid BST.
    
    Args:
        root: Root of the binary tree
        
    Returns:
        True if valid BST, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[[2, 1, 3]]],
      expected: true,
    },
    {
      input: [[[5, 1, 4, null, null, 3, 6]]],
      expected: false,
    },
    {
      input: [[[5, 4, 6, null, null, 3, 7]]],
      expected: false,
    },
    {
      input: [[[1]]],
      expected: true,
    },
  ],
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """
    Range validation approach.
    Time: O(N), Space: O(H) for recursion
    """
    def validate(node, min_val, max_val):
        # Empty node is valid
        if not node:
            return True
        
        # Check if current value is within valid range
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # Validate left subtree: must be < node.val
        # Validate right subtree: must be > node.val
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


# Alternative: Inorder traversal (should be strictly increasing)
def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """
    Inorder traversal approach.
    Time: O(N), Space: O(H)
    """
    def inorder(node):
        if not node:
            return True
        
        # Check left subtree
        if not inorder(node.left):
            return False
        
        # Check current node (must be greater than previous)
        if node.val <= self.prev:
            return False
        self.prev = node.val
        
        # Check right subtree
        return inorder(node.right)
    
    self = type('obj', (), {'prev': float('-inf')})()
    return inorder(root)


# Alternative: Iterative inorder
def is_valid_bst_iterative(root: Optional[TreeNode]) -> bool:
    """
    Iterative inorder traversal.
    Time: O(N), Space: O(H)
    """
    stack = []
    prev = float('-inf')
    curr = root
    
    while stack or curr:
        # Go to leftmost
        while curr:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        
        # Check if strictly increasing
        if curr.val <= prev:
            return False
        prev = curr.val
        
        # Move to right
        curr = curr.right
    
    return True`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h) where h is the height of the tree',

  leetcodeUrl: 'https://leetcode.com/problems/validate-binary-search-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s6ATEkipzow',
  order: 2,
  topic: 'Trees',
};
