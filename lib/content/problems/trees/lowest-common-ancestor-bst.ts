/**
 * Lowest Common Ancestor of a Binary Search Tree
 * Problem ID: lowest-common-ancestor-bst
 * Order: 7
 */

import { Problem } from '../../../types';

export const lowest_common_ancestor_bstProblem: Problem = {
  id: 'lowest-common-ancestor-bst',
  title: 'Lowest Common Ancestor of a Binary Search Tree',
  difficulty: 'Medium',
  topic: 'Trees',
  description: `Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA: "The lowest common ancestor is defined between two nodes \`p\` and \`q\` as the lowest node in T that has both \`p\` and \`q\` as descendants (where we allow **a node to be a descendant of itself**)."`,
  examples: [
    {
      input: 'root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8',
      output: '6',
      explanation: 'The LCA of nodes 2 and 8 is 6.',
    },
    {
      input: 'root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4',
      output: '2',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [2, 10^5]',
    '-10^9 <= Node.val <= 10^9',
    'All Node.val are unique',
    'p != q',
    'p and q will exist in the BST',
  ],
  hints: [
    'Use BST property: left < root < right',
    'If both p and q are less than root, go left',
    'If both p and q are greater than root, go right',
    'Otherwise, current node is LCA',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA of two nodes in BST.
    
    Args:
        root: Root of BST
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
      input: [[6, 2, 8, 0, 4, 7, 9, null, null, 3, 5], 2, 8],
      expected: 6,
    },
    {
      input: [[6, 2, 8, 0, 4, 7, 9, null, null, 3, 5], 2, 4],
      expected: 2,
    },
  ],
  timeComplexity: 'O(h)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=gs2LMfuOR9k',
};
