/**
 * Symmetric Tree
 * Problem ID: symmetric-tree
 * Order: 5
 */

import { Problem } from '../../../types';

export const symmetric_treeProblem: Problem = {
  id: 'symmetric-tree',
  title: 'Symmetric Tree',
  difficulty: 'Easy',
  topic: 'Trees',
  description: `Given the \`root\` of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).`,
  examples: [
    {
      input: 'root = [1,2,2,3,4,4,3]',
      output: 'true',
    },
    {
      input: 'root = [1,2,2,null,3,null,3]',
      output: 'false',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 1000]',
    '-100 <= Node.val <= 100',
  ],
  hints: [
    'A tree is symmetric if left subtree is mirror of right subtree',
    'Compare left.left with right.right, left.right with right.left',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_symmetric(root: Optional[TreeNode]) -> bool:
    """
    Check if binary tree is symmetric.
    
    Args:
        root: Root of tree
        
    Returns:
        True if tree is symmetric
    """
    # Write your code here
    pass
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
    {
      input: [[1]],
      expected: true,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/symmetric-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Mao9uzxwvmc',
};
