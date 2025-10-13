import { Problem } from '../types';

export const dfsProblems: Problem[] = [
  {
    id: 'max-depth-binary-tree',
    title: 'Maximum Depth of Binary Tree',
    difficulty: 'Easy',
    description: `Given the \`root\` of a binary tree, return its **maximum depth**.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.


**Approach:**
Use **DFS (Depth-First Search)** to recursively calculate the depth of left and right subtrees. The maximum depth is 1 (current node) plus the maximum of left and right subtree depths.

**Base Case:** If the node is null, return 0.

**Recursive Case:** Return 1 + max(left_depth, right_depth)

**Key Insight:**
This is a classic bottom-up DFS where we return information from children up to the parent.`,
    examples: [
      {
        input: 'root = [3,9,20,null,null,15,7]',
        output: '3',
        explanation: 'The maximum depth is 3 (3 → 20 → 7 or 3 → 20 → 15)',
      },
      {
        input: 'root = [1,null,2]',
        output: '2',
        explanation: 'The maximum depth is 2 (1 → 2)',
      },
    ],
    constraints: [
      'The number of nodes in the tree is in the range [0, 10^4]',
      '-100 <= Node.val <= 100',
    ],
    hints: [
      'Think recursively: what information do children return?',
      'Base case: what is the depth of an empty tree?',
      "Recursive case: how does parent use children's depths?",
      'This is a postorder traversal (process children first)',
    ],
    starterCode: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root: Optional[TreeNode]) -> int:
    """
    Return the maximum depth of the binary tree.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        Maximum depth as an integer
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[3, 9, 20, null, null, 15, 7]],
        expected: 3,
      },
      {
        input: [[1, null, 2]],
        expected: 2,
      },
      {
        input: [[]],
        expected: 0,
      },
    ],
    solution: `from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root: Optional[TreeNode]) -> int:
    """
    Recursive DFS solution.
    Time: O(N), Space: O(H) where H is height
    """
    # Base case: empty tree has depth 0
    if not root:
        return 0
    
    # Recursively get depth of left and right subtrees
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    # Current depth is 1 + max of children
    return 1 + max(left_depth, right_depth)


# Alternative: Iterative DFS with stack
def max_depth_iterative(root: Optional[TreeNode]) -> int:
    """
    Iterative DFS using stack of (node, depth) pairs.
    Time: O(N), Space: O(H)
    """
    if not root:
        return 0
    
    stack = [(root, 1)]
    max_depth = 0
    
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        
        if node.left:
            stack.append((node.left, depth + 1))
        if node.right:
            stack.append((node.right, depth + 1))
    
    return max_depth`,
    timeComplexity: 'O(N)',
    spaceComplexity: 'O(H) where H is height',
    order: 1,
    topic: 'DFS',
    leetcodeUrl: 'https://leetcode.com/problems/maximum-depth-of-binary-tree/',
    youtubeUrl: 'https://www.youtube.com/watch?v=hTM3phVI6YQ',
  },
  {
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
        input:
          'root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22',
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
    order: 2,
    topic: 'DFS',
    leetcodeUrl: 'https://leetcode.com/problems/path-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=LSKQyOz_P8I',
  },
  {
    id: 'pacific-atlantic-water-flow',
    title: 'Pacific Atlantic Water Flow',
    difficulty: 'Hard',
    description: `There is an \`m x n\` rectangular island that borders both the **Pacific Ocean** and **Atlantic Ocean**. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an \`m x n\` integer matrix \`heights\` where \`heights[r][c]\` represents the **height above sea level** of the cell at coordinate \`(r, c)\`.

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is **less than or equal to** the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a **2D list** of grid coordinates \`result\` where \`result[i] = [ri, ci]\` denotes that rain water can flow from cell \`(ri, ci)\` to **both** the Pacific and Atlantic oceans.

**Approach:**
Instead of checking where water flows FROM each cell, work backwards - find which cells can be reached FROM each ocean:
1. Start DFS from all Pacific border cells (top row + left column)
2. Start DFS from all Atlantic border cells (bottom row + right column)
3. Find the intersection of cells reachable from both oceans

**Key Insight:**
Water flows from high to low, so reverse the problem: start from oceans and move to cells with height >= current height. This avoids checking all possible paths from each cell.`,
    examples: [
      {
        input: `heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]`,
        output: '[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]',
        explanation:
          'Water from these cells can flow to both oceans via height-decreasing paths',
      },
      {
        input: `heights = [[1]]`,
        output: '[[0,0]]',
        explanation:
          'The only cell touches both oceans so water flows to both',
      },
    ],
    constraints: [
      'm == heights.length',
      'n == heights[r].length',
      '1 <= m, n <= 200',
      '0 <= heights[r][c] <= 10^5',
    ],
    hints: [
      'Think backwards: which cells can reach each ocean?',
      'Start DFS from all border cells of each ocean',
      'Water can only flow from higher or equal height to lower',
      'When starting from ocean, reverse the flow direction',
      'The answer is the intersection of Pacific-reachable and Atlantic-reachable cells',
    ],
    starterCode: `from typing import List

def pacific_atlantic(heights: List[List[int]]) -> List[List[int]]:
    """
    Find all cells where water can flow to both Pacific and Atlantic oceans.
    
    Args:
        heights: Matrix of heights above sea level
        
    Returns:
        List of [row, col] coordinates
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 2, 3, 5],
            [3, 2, 3, 4, 4],
            [2, 4, 5, 3, 1],
            [6, 7, 1, 4, 5],
            [5, 1, 1, 2, 4],
          ],
        ],
        expected: [
          [0, 4],
          [1, 3],
          [1, 4],
          [2, 2],
          [3, 0],
          [3, 1],
          [4, 0],
        ],
      },
      {
        input: [[[1]]],
        expected: [[0, 0]],
      },
    ],
    solution: `from typing import List


def pacific_atlantic(heights: List[List[int]]) -> List[List[int]]:
    """
    Reverse DFS: Start from oceans, find which cells can reach each ocean.
    Time: O(M × N), Space: O(M × N)
    """
    if not heights or not heights[0]:
        return []
    
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, visited):
        """DFS to mark all cells reachable from current ocean"""
        visited.add((r, c))
        
        # Check all 4 directions
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            # Can only move to cells with height >= current (water flows down)
            if (0 <= nr < rows and 0 <= nc < cols and 
                (nr, nc) not in visited and 
                heights[nr][nc] >= heights[r][c]):
                dfs(nr, nc, visited)
    
    # Run DFS from all Pacific border cells (top row and left column)
    for c in range(cols):
        dfs(0, c, pacific)
    for r in range(rows):
        dfs(r, 0, pacific)
    
    # Run DFS from all Atlantic border cells (bottom row and right column)
    for c in range(cols):
        dfs(rows - 1, c, atlantic)
    for r in range(rows):
        dfs(r, cols - 1, atlantic)
    
    # Find intersection: cells that can reach both oceans
    return [[r, c] for r, c in pacific & atlantic]`,
    timeComplexity: 'O(M × N) - visit each cell at most twice',
    spaceComplexity: 'O(M × N) for recursion stack and visited sets',
    order: 3,
    topic: 'DFS',
    leetcodeUrl: 'https://leetcode.com/problems/pacific-atlantic-water-flow/',
    youtubeUrl: 'https://www.youtube.com/watch?v=s-VkcjHqkGI',
  },
];

