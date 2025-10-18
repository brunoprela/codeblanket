/**
 * Binary Tree Level Order Traversal
 * Problem ID: binary-tree-level-order
 * Order: 1
 */

import { Problem } from '../../../types';

export const binary_tree_level_orderProblem: Problem = {
  id: 'binary-tree-level-order',
  title: 'Binary Tree Level Order Traversal',
  difficulty: 'Easy',
  description: `Given the \`root\` of a binary tree, return the **level order traversal** of its nodes' values. (i.e., from left to right, level by level).


**Approach:**
Use **BFS (Breadth-First Search)** with a queue. Process nodes level by level:
1. Start with root in queue
2. For each level, capture queue size
3. Process all nodes in current level
4. Add their children to queue for next level

**Key Insight:**
The trick is to capture \`len(queue)\` **before** the inner loop to know exactly how many nodes are in the current level.`,
  examples: [
    {
      input: 'root = [3,9,20,null,null,15,7]',
      output: '[[3],[9,20],[15,7]]',
      explanation: 'Level 1: [3], Level 2: [9, 20], Level 3: [15, 7]',
    },
    {
      input: 'root = [1]',
      output: '[[1]]',
      explanation: 'Single node tree',
    },
    {
      input: 'root = []',
      output: '[]',
      explanation: 'Empty tree',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 2000]',
    '-1000 <= Node.val <= 1000',
  ],
  hints: [
    'Use a queue for BFS traversal',
    'Capture queue size before processing level',
    'Process exactly that many nodes for current level',
    'Add children to queue for next level',
  ],
  starterCode: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return level order traversal as list of levels.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        List of lists, each inner list represents one level
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 9, 20, null, null, 15, 7]],
      expected: [[3], [9, 20], [15, 7]],
    },
    {
      input: [[1]],
      expected: [[1]],
    },
    {
      input: [[]],
      expected: [],
    },
  ],
  solution: `from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS solution - process level by level.
    Time: O(N), Space: O(W) where W is max width
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)  # Capture size before loop!
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


# Alternative: Without capturing level size (flatten instead of levels)
def level_order_flat(root: Optional[TreeNode]) -> List[int]:
    """Return flattened level-order traversal"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result`,
  timeComplexity: 'O(N)',
  spaceComplexity: 'O(W) where W is maximum width',
  order: 1,
  topic: 'Breadth-First Search (BFS)',
  leetcodeUrl:
    'https://leetcode.com/problems/binary-tree-level-order-traversal/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6ZnyEApgFYg',
};
