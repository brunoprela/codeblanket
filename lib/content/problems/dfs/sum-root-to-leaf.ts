/**
 * Sum Root to Leaf Numbers
 * Problem ID: sum-root-to-leaf
 * Order: 5
 */

import { Problem } from '../../../types';

export const sum_root_to_leafProblem: Problem = {
  id: 'sum-root-to-leaf',
  title: 'Sum Root to Leaf Numbers',
  difficulty: 'Easy',
  topic: 'Depth-First Search (DFS)',
  description: `You are given the \`root\` of a binary tree containing digits from \`0\` to \`9\` only.

Each root-to-leaf path in the tree represents a number.

- For example, the root-to-leaf path \`1 -> 2 -> 3\` represents the number \`123\`.

Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a **32-bit** integer.

A **leaf** node is a node with no children.`,
  examples: [
    {
      input: 'root = [1,2,3]',
      output: '25',
      explanation:
        'The root-to-leaf path 1->2 represents the number 12. The root-to-leaf path 1->3 represents the number 13. Therefore, sum = 12 + 13 = 25.',
    },
    {
      input: 'root = [4,9,0,5,1]',
      output: '1026',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 1000]',
    '0 <= Node.val <= 9',
    'The depth of the tree will not exceed 10',
  ],
  hints: [
    'DFS with current number as parameter',
    'At each node: current = current * 10 + node.val',
    'Add to sum when reaching leaf',
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def sum_numbers(root: Optional[TreeNode]) -> int:
    """
    Sum all root-to-leaf path numbers.
    
    Args:
        root: Root of tree
        
    Returns:
        Sum of all path numbers
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: 25,
    },
    {
      input: [[4, 9, 0, 5, 1]],
      expected: 1026,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(h)',
  leetcodeUrl: 'https://leetcode.com/problems/sum-root-to-leaf-numbers/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Jg4E4KZstFE',
};
