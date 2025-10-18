/**
 * Path Sum
 * Problem ID: path-sum
 * Order: 2
 */

import { Problem } from '../../../types';

export const path_sumProblem: Problem = {
  id: 'path-sum',
  title: 'Path Sum',
  difficulty: 'Medium',
  description: `Given the \`root\` of a binary tree and an integer \`targetSum\`, return \`true\` if the tree has a **root-to-leaf** path such that adding up all the values along the path equals \`targetSum\`.

A **leaf** is a node with no children.


**Approach:**
Use **DFS** to explore all root-to-leaf paths. At each node:
1. Subtract node's value from remaining sum
2. If leaf node, check if remaining sum equals current value
3. Otherwise, recursively check left and right subtrees

**Key Insight:**
This is a top-down DFS where we pass information (remaining sum) down the tree. We return boolean results up the tree using OR logic.`,
  examples: [
    {
      input: 'root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22',
      output: 'true',
      explanation: 'Path exists: 5 → 4 → 11 → 2 (sum = 22)',
    },
    {
      input: 'root = [1,2,3], targetSum = 5',
      output: 'false',
      explanation: 'No root-to-leaf path sums to 5',
    },
    {
      input: 'root = [], targetSum = 0',
      output: 'false',
      explanation: 'Empty tree has no paths',
    },
  ],
  constraints: [
    'The number of nodes in the tree is in the range [0, 5000]',
    '-1000 <= Node.val <= 1000',
    '-1000 <= targetSum <= 1000',
  ],
  hints: [
    'Use DFS to explore all paths from root to leaves',
    'Subtract current node value from target as you go down',
    "Check if you've reached target when at a leaf node",
    'Return true if ANY path works (use OR)',
    "Don't forget: must be root-to-LEAF path",
  ],
  starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    Check if there exists a root-to-leaf path with sum equal to targetSum.
    
    Args:
        root: Root node of the binary tree
        target_sum: Target sum to find
        
    Returns:
        True if such path exists, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[5, 4, 8, 11, null, 13, 4, 7, 2, null, null, null, 1], 22],
      expected: true,
    },
    {
      input: [[1, 2, 3], 5],
      expected: false,
    },
    {
      input: [[], 0],
      expected: false,
    },
  ],
  solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def has_path_sum(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    Recursive DFS solution - top-down approach.
    Time: O(N), Space: O(H)
    """
    # Base case: empty tree
    if not root:
        return False
    
    # Leaf node: check if remaining sum equals node value
    if not root.left and not root.right:
        return target_sum == root.val
    
    # Recursive case: check left or right subtree
    # Subtract current value from remaining sum
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))


# Alternative: Iterative DFS with stack
def has_path_sum_iterative(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    Iterative DFS using stack of (node, remaining_sum) pairs.
    Time: O(N), Space: O(H)
    """
    if not root:
        return False
    
    stack = [(root, target_sum)]
    
    while stack:
        node, remaining = stack.pop()
        
        # Check if leaf with correct sum
        if not node.left and not node.right:
            if remaining == node.val:
                return True
        
        # Add children with updated remaining sum
        if node.right:
            stack.append((node.right, remaining - node.val))
        if node.left:
            stack.append((node.left, remaining - node.val))
    
    return False


# Alternative: Track full path
def has_path_sum_with_path(root: Optional[TreeNode], target_sum: int) -> bool:
    """
    DFS that also tracks the actual path.
    """
    def dfs(node, remaining, path):
        if not node:
            return False
        
        path.append(node.val)
        
        if not node.left and not node.right:
            if remaining == node.val:
                print(f"Found path: {path}")
                return True
        
        remaining -= node.val
        found = dfs(node.left, remaining, path) or dfs(node.right, remaining, path)
        
        path.pop()  # Backtrack
        return found
    
    return dfs(root, target_sum, [])`,
  timeComplexity: 'O(N)',
  spaceComplexity: 'O(H) where H is height',

  leetcodeUrl: 'https://leetcode.com/problems/path-sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=LSKQyOz_P8I',
  order: 2,
  topic: 'Depth-First Search (DFS)',
};
