/**
 * Construct Binary Tree from Preorder and Inorder Traversal
 * Problem ID: construct-tree-preorder-inorder
 * Order: 8
 */

import { Problem } from '../../../types';

export const construct_tree_preorder_inorderProblem: Problem = {
  id: 'construct-tree-preorder-inorder',
  title: 'Construct Binary Tree from Preorder and Inorder Traversal',
  difficulty: 'Medium',
  topic: 'Trees',
  description: `Given two integer arrays \`preorder\` and \`inorder\` where \`preorder\` is the preorder traversal of a binary tree and \`inorder\` is the inorder traversal of the same tree, construct and return the binary tree.`,
  examples: [
    {
      input: 'preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]',
      output: '[3,9,20,null,null,15,7]',
    },
    {
      input: 'preorder = [-1], inorder = [-1]',
      output: '[-1]',
    },
  ],
  constraints: [
    '1 <= preorder.length <= 3000',
    'inorder.length == preorder.length',
    '-3000 <= preorder[i], inorder[i] <= 3000',
    'preorder and inorder consist of unique values',
    'Each value of inorder also appears in preorder',
    'preorder is guaranteed to be the preorder traversal of the tree',
    'inorder is guaranteed to be the inorder traversal of the tree',
  ],
  hints: [
    'First element in preorder is always root',
    'Find root in inorder to split left and right subtrees',
    'Recursively build left and right subtrees',
  ],
  starterCode: `from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Construct binary tree from preorder and inorder traversals.
    
    Args:
        preorder: Preorder traversal
        inorder: Inorder traversal
        
    Returns:
        Root of constructed tree
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [3, 9, 20, 15, 7],
        [9, 3, 15, 20, 7],
      ],
      expected: [3, 9, 20, null, null, 15, 7],
    },
    {
      input: [[-1], [-1]],
      expected: [-1],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl:
    'https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ihj4IQGZ2zc',
};
