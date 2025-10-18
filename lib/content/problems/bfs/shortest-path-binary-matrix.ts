/**
 * Shortest Path in Binary Matrix
 * Problem ID: shortest-path-binary-matrix
 * Order: 2
 */

import { Problem } from '../../../types';

export const shortest_path_binary_matrixProblem: Problem = {
  id: 'shortest-path-binary-matrix',
  title: 'Shortest Path in Binary Matrix',
  difficulty: 'Medium',
  description: `Given an \`n x n\` binary matrix \`grid\`, return the length of the **shortest clear path** in the matrix. If there is no clear path, return \`-1\`.

A **clear path** in a binary matrix is a path from the **top-left** cell (0, 0) to the **bottom-right** cell (n-1, n-1) such that:
- All visited cells are \`0\`
- All adjacent cells are **8-directionally** connected

The **length of a clear path** is the number of visited cells.


**Approach:**
Use **BFS** to find the shortest path in an unweighted grid:
1. Start from (0, 0) if it is 0
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

  leetcodeUrl: 'https://leetcode.com/problems/shortest-path-in-binary-matrix/',
  youtubeUrl: 'https://www.youtube.com/watch?v=caXJJOMLyHk',
  order: 2,
  topic: 'Breadth-First Search (BFS)',
};
