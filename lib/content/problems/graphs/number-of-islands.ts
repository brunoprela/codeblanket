/**
 * Number of Islands
 * Problem ID: number-of-islands
 * Order: 1
 */

import { Problem } from '../../../types';

export const number_of_islandsProblem: Problem = {
  id: 'number-of-islands',
  title: 'Number of Islands',
  difficulty: 'Easy',
  description: `Given an \`m x n\` 2D binary grid \`grid\` which represents a map of \`'1'\`s (land) and \`'0'\`s (water), return **the number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


**Approach:**
This is a **connected components** problem. Use DFS or BFS to explore each island, marking visited cells. Each time we start a new DFS/BFS, we've found a new island.

**Key Insight:**
Think of the grid as an implicit graph where each land cell is connected to its adjacent land cells (up, down, left, right).`,
  examples: [
    {
      input:
        'grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]',
      output: '1',
      explanation: 'All the 1s are connected, forming one island.',
    },
    {
      input:
        'grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]',
      output: '3',
      explanation: 'There are three separate islands.',
    },
  ],
  constraints: [
    'm == grid.length',
    'n == grid[i].length',
    '1 <= m, n <= 300',
    'grid[i][j] is "0" or "1"',
  ],
  hints: [
    'Iterate through each cell in the grid',
    'When you find a "1" (unvisited land), increment island count',
    'Run DFS/BFS from that cell to mark all connected land as visited',
    'DFS will explore all 4 directions (up, down, left, right)',
    'Mark visited cells by changing "1" to "0" or use a visited set',
  ],
  starterCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    Count the number of islands in a 2D grid.
    
    Args:
        grid: 2D grid of "1" (land) and "0" (water)
        
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
    {
      input: [[['1']]],
      expected: 1,
    },
    {
      input: [[['0']]],
      expected: 0,
    },
  ],
  solution: `from typing import List


def num_islands(grid: List[List[str]]) -> int:
    """
    DFS approach (modifies grid in-place).
    Time: O(M * N), Space: O(M * N) for recursion
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        # Boundary checks
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0'):
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left
    
    # Iterate through all cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)  # Mark entire island
    
    return count


# Alternative: BFS approach
def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    BFS approach (doesn't modify input).
    Time: O(M * N), Space: O(min(M, N)) for queue
    """
    if not grid:
        return 0
    
    from collections import deque
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    
    def bfs(r, c):
        queue = deque([(r, c)])
        visited.add((r, c))
        
        while queue:
            row, col = queue.popleft()
            
            # Check all 4 directions
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = row + dr, col + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr][nc] == '1' and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                count += 1
                bfs(r, c)
    
    return count


# Alternative: Union-Find approach
def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    Union-Find approach.
    Time: O(M * N * α(M*N)), Space: O(M * N)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    class UnionFind:
        def __init__(self, grid):
            self.count = 0
            self.parent = {}
            self.rank = {}
            
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == '1':
                        key = r * cols + c
                        self.parent[key] = key
                        self.rank[key] = 0
                        self.count += 1
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            root_x, root_y = self.find(x), self.find(y)
            
            if root_x != root_y:
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                elif self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1
                self.count -= 1
    
    uf = UnionFind(grid)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                # Union with right and down neighbors
                if c + 1 < cols and grid[r][c + 1] == '1':
                    uf.union(r * cols + c, r * cols + c + 1)
                if r + 1 < rows and grid[r + 1][c] == '1':
                    uf.union(r * cols + c, (r + 1) * cols + c)
    
    return uf.count`,
  timeComplexity: 'O(M * N)',
  spaceComplexity: 'O(M * N) for recursion/visited set',

  leetcodeUrl: 'https://leetcode.com/problems/number-of-islands/',
  youtubeUrl: 'https://www.youtube.com/watch?v=pV2kpPD66nE',
  order: 1,
  topic: 'Graphs',
};
