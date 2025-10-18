/**
 * Average of Levels in Binary Tree
 * Problem ID: average-of-levels
 * Order: 5
 */

import { Problem } from '../../../types';

export const average_of_levelsProblem: Problem = {
  id: 'average-of-levels',
  title: 'Average of Levels in Binary Tree',
  difficulty: 'Easy',
  topic: 'Breadth-First Search (BFS)',
  description: `Given the \`root\` of a binary tree, return the average value of the nodes on each level in the form of an array. Answers within \`10^-5\` of the actual answer will be accepted.`,
  examples: [
    {
      input: 'root = [3,9,20,null,null,15,7]',
      output: '[3.00000,14.50000,11.00000]',
      explanation: 'Level 0: 3, Level 1: (9+20)/2=14.5, Level 2: (15+7)/2=11',
    },
    {
      input: 'root = [3,9,20,15,7]',
      output: '[3.00000,14.50000,11.00000]',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [1, 10^4]',
    '-2^31 <= Node.val <= 2^31 - 1',
  ],
  hints: ['Use BFS level by level', 'Calculate average for each level'],
  starterCode: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def average_of_levels(root: Optional[TreeNode]) -> List[float]:
    """
    Calculate average value at each level.
    
    Args:
        root: Root of tree
        
    Returns:
        List of averages per level
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 9, 20, null, null, 15, 7]],
      expected: [3.0, 14.5, 11.0],
    },
    {
      input: [[3, 9, 20, 15, 7]],
      expected: [3.0, 14.5, 11.0],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl:
    'https://leetcode.com/problems/average-of-levels-in-binary-tree/',
  youtubeUrl: 'https://www.youtube.com/watch?v=3yxZMxG4Vdk',
};
