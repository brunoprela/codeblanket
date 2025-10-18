/**
 * Diameter of Binary Tree
 * Problem ID: diameter-binary-tree
 * Order: 6
 */

import { Problem } from '../../../types';

export const diameter_binary_treeProblem: Problem = {
  id: 'diameter-binary-tree',
  title: 'Diameter of Binary Tree',
  difficulty: 'Easy',
  topic: 'Trees',
  description: `Given the \`root\` of a binary tree, return the length of the **diameter** of the tree.

The **diameter** of a binary tree is the **length** of the longest path between any two nodes in a tree. This path may or may not pass through the \`root\`.

The **length** of a path between two nodes is represented by the number of edges between them.`,
  examples: [
    {
      input: 'root = [1,2,3,4,5]',
      output: '3',
      explanation: 'The path [4,2,1,3] or [5,2,1,3] has length 3.',
    },
    {
      input: 'root = [1,2]',
      output: '1',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 10^4]',
    '-100 <= Node.val <= 100',
  ],
  hints: [
    'For each node, diameter through it is left_height + right_height',
    'Track maximum diameter while calculating heights',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """
    Find diameter of binary tree.
    
    Args:
        root: Root of tree
        
    Returns:
        Length of longest path
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5]],
      expected: 3,
    },
    {
      input: [[1, 2]],
      expected: 1,
    },
    {
      input: [[1]],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/diameter-of-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=bkxqA8Rfv04',
};
