import { Problem } from '../types';

export const dfsProblems: Problem[] = [
  {
    id: 'max-depth-binary-tree',
    title: 'Maximum Depth of Binary Tree',
    difficulty: 'Easy',
    description: `Given the \`root\` of a binary tree, return its **maximum depth**.

A binary tree's **maximum depth** is the number of nodes along the longest path from the root node down to the farthest leaf node.

**LeetCode:** [104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
**YouTube:** [NeetCode - Maximum Depth of Binary Tree](https://www.youtube.com/watch?v=hTM3phVI6YQ)

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

**LeetCode:** [112. Path Sum](https://leetcode.com/problems/path-sum/)
**YouTube:** [NeetCode - Path Sum](https://www.youtube.com/watch?v=LSKQyOz_P8I)

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
    id: 'number-of-islands',
    title: 'Number of Islands',
    difficulty: 'Hard',
    description: `Given an \`m x n\` 2D binary grid \`grid\` which represents a map of \`'1'\`s (land) and \`'0'\`s (water), return the **number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**LeetCode:** [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
**YouTube:** [NeetCode - Number of Islands](https://www.youtube.com/watch?v=pV2kpPD66nE)

**Approach:**
Use **DFS** to explore each island:
1. Iterate through each cell in the grid
2. When you find a '1', increment island count
3. Use DFS to mark all connected '1's as visited
4. Continue until all cells are processed

**Key Insight:**
This is a classic connected components problem. Each DFS call explores one complete island. The number of DFS calls equals the number of islands.

**Optimization:** Mark visited cells in-place by changing '1' to '0' or use a separate visited set.`,
    examples: [
      {
        input: `grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]`,
        output: '1',
        explanation: 'All connected 1s form a single island',
      },
      {
        input: `grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]`,
        output: '3',
        explanation: 'Three separate islands',
      },
    ],
    constraints: [
      'm == grid.length',
      'n == grid[i].length',
      '1 <= m, n <= 300',
      "grid[i][j] is '0' or '1'",
    ],
    hints: [
      'Use DFS to explore each island completely',
      'Mark visited cells to avoid counting same island twice',
      'Count how many times you start a new DFS',
      'Check all 4 directions: up, down, left, right',
      'Can modify grid in-place or use separate visited set',
    ],
    starterCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    Count the number of islands in a 2D grid.
    
    Args:
        grid: 2D grid where '1' is land and '0' is water
        
    Returns:
        Number of islands
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            ['1', '1', '1', '1', '0'],
            ['1', '1', '0', '1', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '0', '0', '0'],
          ],
        ],
        expected: 1,
      },
      {
        input: [
          [
            ['1', '1', '0', '0', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '1', '0', '0'],
            ['0', '0', '0', '1', '1'],
          ],
        ],
        expected: 3,
      },
    ],
    solution: `from typing import List


def num_islands(grid: List[List[str]]) -> int:
    """
    DFS solution - mark visited cells in-place.
    Time: O(M × N), Space: O(M × N) for recursion stack
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        """Explore entire island starting from (r, c)"""
        # Base cases: out of bounds or water
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left
    
    # Iterate through each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                # Found new island
                count += 1
                # Mark entire island as visited
                dfs(r, c)
    
    return count


# Alternative: Using separate visited set (preserves grid)
def num_islands_visited_set(grid: List[List[str]]) -> int:
    """
    DFS with separate visited set.
    Time: O(M × N), Space: O(M × N)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0' or (r, c) in visited):
            return
        
        visited.add((r, c))
        
        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                count += 1
                dfs(r, c)
    
    return count


# Alternative: Iterative DFS with stack
def num_islands_iterative(grid: List[List[str]]) -> int:
    """
    Iterative DFS using explicit stack.
    Time: O(M × N), Space: O(M × N)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                
                # DFS with stack
                stack = [(r, c)]
                grid[r][c] = '0'
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    # Check all 4 directions
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr][nc] == '1'):
                            grid[nr][nc] = '0'
                            stack.append((nr, nc))
    
    return count`,
    timeComplexity: 'O(M × N)',
    spaceComplexity: 'O(M × N) for recursion stack in worst case',
    order: 3,
    topic: 'DFS',
    leetcodeUrl: 'https://leetcode.com/problems/number-of-islands/',
    youtubeUrl: 'https://www.youtube.com/watch?v=pV2kpPD66nE',
  },
];
