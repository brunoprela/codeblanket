/**
 * Balanced Binary Tree
 * Problem ID: balanced-binary-tree
 * Order: 4
 */

import { Problem } from '../../../types';

export const balanced_binary_treeProblem: Problem = {
  id: 'balanced-binary-tree',
  title: 'Balanced Binary Tree',
  difficulty: 'Easy',
  topic: 'Depth-First Search (DFS)',
  description: `Given a binary tree, determine if it is **height-balanced**.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.`,
  examples: [
    {
      input: 'root = [3,9,20,null,null,15,7]',
      output: 'true',
    },
    {
      input: 'root = [1,2,2,3,3,null,null,4,4]',
      output: 'false',
    },
    {
      input: 'root = []',
      output: 'true',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 5000]',
    '-10^4 <= Node.val <= 10^4',
  ],
  hints: [
    'Use DFS to calculate height',
    'Return -1 if unbalanced',
    'Check height difference at each node',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_balanced(root: Optional[TreeNode]) -> bool:
    """
    Check if binary tree is height-balanced.
    
    Args:
        root: Root of tree
        
    Returns:
        True if balanced
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 9, 20, null, null, 15, 7]],
      expected: true,
    },
    {
      input: [[1, 2, 2, 3, 3, null, null, 4, 4]],
      expected: false,
    },
    {
      input: [[]],
      expected: true,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/balanced-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=QfJsau0ItOY',
};
