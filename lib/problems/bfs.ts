import { Problem } from '../types';

export const bfsProblems: Problem[] = [
  {
    id: 'binary-tree-level-order',
    title: 'Binary Tree Level Order Traversal',
    difficulty: 'Easy',
    description: `Given the \`root\` of a binary tree, return the **level order traversal** of its nodes' values. (i.e., from left to right, level by level).

**LeetCode:** [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
**YouTube:** [NeetCode - Binary Tree Level Order Traversal](https://www.youtube.com/watch?v=6ZnyEApgFYg)

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
    topic: 'BFS',
    leetcodeUrl:
      'https://leetcode.com/problems/binary-tree-level-order-traversal/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6ZnyEApgFYg',
  },
  {
    id: 'shortest-path-binary-matrix',
    title: 'Shortest Path in Binary Matrix',
    difficulty: 'Medium',
    description: `Given an \`n x n\` binary matrix \`grid\`, return the length of the **shortest clear path** in the matrix. If there is no clear path, return \`-1\`.

A **clear path** in a binary matrix is a path from the **top-left** cell (0, 0) to the **bottom-right** cell (n-1, n-1) such that:
- All visited cells are \`0\`
- All adjacent cells are **8-directionally** connected

The **length of a clear path** is the number of visited cells.

**LeetCode:** [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
**YouTube:** [NeetCode - Shortest Path in Binary Matrix](https://www.youtube.com/watch?v=caXJJOMLyHk)

**Approach:**
Use **BFS** to find the shortest path in an unweighted grid:
1. Start from (0, 0) if it's 0
2. Explore all 8 directions
3. Track visited cells
4. Return distance when reaching (n-1, n-1)

**Key Insight:**
BFS guarantees the shortest path in an unweighted graph. First time you reach the destination is the shortest path.`,
    examples: [
      {
        input: 'grid = [[0,1],[1,0]]',
        output: '2',
        explanation: 'Path: (0,0) → (1,1)',
      },
      {
        input: 'grid = [[0,0,0],[1,1,0],[1,1,0]]',
        output: '4',
        explanation: 'Path: (0,0) → (0,1) → (0,2) → (1,2) → (2,2)',
      },
      {
        input: 'grid = [[1,0,0],[1,1,0],[1,1,0]]',
        output: '-1',
        explanation: 'Start cell is blocked',
      },
    ],
    constraints: [
      'n == grid.length',
      'n == grid[i].length',
      '1 <= n <= 100',
      'grid[i][j] is 0 or 1',
    ],
    hints: [
      'Use BFS for shortest path in unweighted grid',
      'Check all 8 directions (including diagonals)',
      'Mark cells as visited to avoid revisiting',
      'Track distance/path length as you go',
      'First arrival at destination is shortest',
    ],
    starterCode: `from typing import List
from collections import deque

def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """
    Find shortest clear path in binary matrix.
    
    Args:
        grid: n x n binary matrix (0 = clear, 1 = blocked)
        
    Returns:
        Length of shortest path, or -1 if no path exists
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [0, 1],
            [1, 0],
          ],
        ],
        expected: 2,
      },
      {
        input: [
          [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
          ],
        ],
        expected: 4,
      },
      {
        input: [
          [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
          ],
        ],
        expected: -1,
      },
    ],
    solution: `from typing import List
from collections import deque


def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """
    BFS solution for shortest path in 8-directional grid.
    Time: O(N²), Space: O(N²)
    """
    n = len(grid)
    
    # Check if start or end is blocked
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # Single cell case
    if n == 1:
        return 1
    
    # 8 directions: up, down, left, right, and 4 diagonals
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    visited = {(0, 0)}
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    
    while queue:
        r, c, dist = queue.popleft()
        
        # Check all 8 directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if reached destination
            if nr == n - 1 and nc == n - 1:
                return dist + 1
            
            # Check bounds and if cell is valid
            if (0 <= nr < n and 0 <= nc < n and 
                grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1  # No path found


# Alternative: Modify grid in-place to mark visited
def shortest_path_inplace(grid: List[List[int]]) -> int:
    """
    Mark visited cells in grid itself (more space efficient).
    Time: O(N²), Space: O(N²) for queue only
    """
    n = len(grid)
    
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    if n == 1:
        return 1
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    queue = deque([(0, 0, 1)])
    grid[0][0] = 1  # Mark as visited
    
    while queue:
        r, c, dist = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if nr == n - 1 and nc == n - 1:
                return dist + 1
            
            if (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0):
                grid[nr][nc] = 1  # Mark visited
                queue.append((nr, nc, dist + 1))
    
    return -1`,
    timeComplexity: 'O(N²)',
    spaceComplexity: 'O(N²) for visited set and queue',
    order: 2,
    topic: 'BFS',
    leetcodeUrl:
      'https://leetcode.com/problems/shortest-path-in-binary-matrix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=caXJJOMLyHk',
  },
  {
    id: 'rotting-oranges',
    title: 'Rotting Oranges',
    difficulty: 'Hard',
    description: `You are given an \`m x n\` grid where each cell can have one of three values:
- \`0\` representing an empty cell
- \`1\` representing a fresh orange
- \`2\` representing a rotten orange

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return the **minimum number of minutes** that must elapse until no cell has a fresh orange. If this is impossible, return \`-1\`.

**LeetCode:** [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)
**YouTube:** [NeetCode - Rotting Oranges](https://www.youtube.com/watch?v=y704fEOx0s0)

**Approach:**
Use **Multi-Source BFS** - start from ALL rotten oranges simultaneously:
1. Add all initial rotten oranges to queue
2. Count total fresh oranges
3. BFS: spread rot to adjacent fresh oranges
4. Track minutes elapsed
5. Check if any fresh oranges remain

**Key Insight:**
This is multi-source BFS where rot spreads simultaneously from all sources. Process one "round" of rot per minute.`,
    examples: [
      {
        input: 'grid = [[2,1,1],[1,1,0],[0,1,1]]',
        output: '4',
        explanation:
          'Minute 0: [[2,1,1],[1,1,0],[0,1,1]]\n' +
          'Minute 1: [[2,2,1],[2,1,0],[0,1,1]]\n' +
          'Minute 2: [[2,2,2],[2,2,0],[0,1,1]]\n' +
          'Minute 3: [[2,2,2],[2,2,0],[0,2,1]]\n' +
          'Minute 4: [[2,2,2],[2,2,0],[0,2,2]]',
      },
      {
        input: 'grid = [[2,1,1],[0,1,1],[1,0,1]]',
        output: '-1',
        explanation: 'Orange at bottom left cannot be reached',
      },
      {
        input: 'grid = [[0,2]]',
        output: '0',
        explanation: 'No fresh oranges to rot',
      },
    ],
    constraints: [
      'm == grid.length',
      'n == grid[i].length',
      '1 <= m, n <= 10',
      'grid[i][j] is 0, 1, or 2',
    ],
    hints: [
      'Use multi-source BFS starting from all rotten oranges',
      'Count fresh oranges at the start',
      'Process one level (minute) at a time',
      'Decrease fresh count as oranges rot',
      'Check if any fresh oranges remain at the end',
    ],
    starterCode: `from typing import List
from collections import deque

def oranges_rotting(grid: List[List[int]]) -> int:
    """
    Find minimum minutes for all oranges to rot.
    
    Args:
        grid: m x n grid (0=empty, 1=fresh, 2=rotten)
        
    Returns:
        Minimum minutes, or -1 if impossible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [2, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
          ],
        ],
        expected: 4,
      },
      {
        input: [
          [
            [2, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
          ],
        ],
        expected: -1,
      },
      {
        input: [[[0, 2]]],
        expected: 0,
      },
    ],
    solution: `from typing import List
from collections import deque


def oranges_rotting(grid: List[List[int]]) -> int:
    """
    Multi-source BFS solution.
    Time: O(M × N), Space: O(M × N)
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh_count += 1
    
    # If no fresh oranges, return 0
    if fresh_count == 0:
        return 0
    
    minutes = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # BFS - process level by level (minute by minute)
    while queue:
        level_size = len(queue)
        
        # Process all oranges that rot in this minute
        for _ in range(level_size):
            r, c = queue.popleft()
            
            # Try to rot adjacent fresh oranges
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 1):
                    grid[nr][nc] = 2  # Rot it
                    fresh_count -= 1
                    queue.append((nr, nc))
        
        # Increment minutes only if we processed some oranges
        if queue:
            minutes += 1
    
    # Check if any fresh oranges remain
    return minutes if fresh_count == 0 else -1


# Alternative: Track minutes per cell
def oranges_rotting_alt(grid: List[List[int]]) -> int:
    """
    Alternative: track minutes with each cell.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, minute)
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    max_minutes = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        r, c, minute = queue.popleft()
        max_minutes = max(max_minutes, minute)
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] == 1):
                grid[nr][nc] = 2
                fresh_count -= 1
                queue.append((nr, nc, minute + 1))
    
    return max_minutes if fresh_count == 0 else -1`,
    timeComplexity: 'O(M × N)',
    spaceComplexity: 'O(M × N)',
    order: 3,
    topic: 'BFS',
    leetcodeUrl: 'https://leetcode.com/problems/rotting-oranges/',
    youtubeUrl: 'https://www.youtube.com/watch?v=y704fEOx0s0',
  },
];
