/**
 * Binary Tree Right Side View
 * Problem ID: binary-tree-right-side-view
 * Order: 6
 */

import { Problem } from '../../../types';

export const binary_tree_right_side_viewProblem: Problem = {
  id: 'binary-tree-right-side-view',
  title: 'Binary Tree Right Side View',
  difficulty: 'Medium',
  topic: 'Depth-First Search (DFS)',
  description: `Given the \`root\` of a binary tree, imagine yourself standing on the **right side** of it, return the values of the nodes you can see ordered from top to bottom.`,
  examples: [
    {
      input: 'root = [1,2,3,null,5,null,4]',
      output: '[1,3,4]',
    },
    {
      input: 'root = [1,null,3]',
      output: '[1,3]',
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
    'DFS with level tracking',
    'Visit right subtree before left',
    'First node at each level is visible',
  ],
  starterCode: `from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def right_side_view(root: Optional[TreeNode]) -> List[int]:
    """
    Get right side view of binary tree.
    
    Args:
        root: Root of tree
        
    Returns:
        List of rightmost nodes at each level
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, null, 5, null, 4]],
      expected: [1, 3, 4],
    },
    {
      input: [[1, null, 3]],
      expected: [1, 3],
    },
    {
      input: [[]],
      expected: [],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/binary-tree-right-side-view/',
  youtubeUrl: 'https://www.youtube.com/watch?v=d4zLyf32e3I',
};
