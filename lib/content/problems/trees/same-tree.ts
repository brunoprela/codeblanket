/**
 * Same Tree
 * Problem ID: same-tree
 * Order: 4
 */

import { Problem } from '../../../types';

export const same_treeProblem: Problem = {
  id: 'same-tree',
  title: 'Same Tree',
  difficulty: 'Easy',
  topic: 'Trees',
  description: `Given the roots of two binary trees \`p\` and \`q\`, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.`,
  examples: [
    {
      input: 'p = [1,2,3], q = [1,2,3]',
      output: 'true',
    },
    {
      input: 'p = [1,2], q = [1,null,2]',
      output: 'false',
    },
  ],
  constraints: [
    'The number of nodes in both trees is in the range [0, 100]',
    '-10^4 <= Node.val <= 10^4',
  ],
  hints: [
    'Use recursion to compare nodes',
    'Base cases: both null (true), one null (false)',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    Check if two binary trees are identical.
    
    Args:
        p: Root of first tree
        q: Root of second tree
        
    Returns:
        True if trees are identical
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [1, 2, 3],
        [1, 2, 3],
      ],
      expected: true,
    },
    {
      input: [
        [1, 2],
        [1, null, 2],
      ],
      expected: false,
    },
    {
      input: [[], []],
      expected: true,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/same-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=vRbbcKXCxOw',
};
