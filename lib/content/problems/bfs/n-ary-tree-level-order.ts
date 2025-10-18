/**
 * N-ary Tree Level Order Traversal
 * Problem ID: n-ary-tree-level-order
 * Order: 4
 */

import { Problem } from '../../../types';

export const n_ary_tree_level_orderProblem: Problem = {
  id: 'n-ary-tree-level-order',
  title: 'N-ary Tree Level Order Traversal',
  difficulty: 'Easy',
  topic: 'Breadth-First Search (BFS)',
  description: `Given an n-ary tree, return the level order traversal of its nodes' values.

*Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value.*`,
  examples: [
    {
      input: 'root = [1,null,3,2,4,null,5,6]',
      output: '[[1],[3,2,4],[5,6]]',
    },
    {
      input:
        'root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]',
      output: '[[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]',
    },
  ],
  constraints: [
    'The height of the n-ary tree is less than or equal to 1000',
    'The total number of nodes is between [0, 10^4]',
  ],
  hints: [
    'Similar to binary tree level order',
    'Each node has list of children',
  ],
  starterCode: `from typing import List
from collections import deque

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

def level_order_n_ary(root: Node) -> List[List[int]]:
    """
    Level order traversal of n-ary tree.
    
    Args:
        root: Root of n-ary tree
        
    Returns:
        List of levels
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, null, 3, 2, 4, null, 5, 6]],
      expected: [[1], [3, 2, 4], [5, 6]],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl:
    'https://leetcode.com/problems/n-ary-tree-level-order-traversal/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZfgOYcJcFwg',
};
