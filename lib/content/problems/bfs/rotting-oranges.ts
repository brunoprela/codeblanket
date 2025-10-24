/**
 * Rotting Oranges
 * Problem ID: rotting-oranges
 * Order: 3
 */

import { Problem } from '../../../types';

export const rotting_orangesProblem: Problem = {
  id: 'rotting-oranges',
  title: 'Rotting Oranges',
  difficulty: 'Hard',
  description: `You are given an \`m x n\` grid where each cell can have one of three values:
- \`0\` representing an empty cell
- \`1\` representing a fresh orange
- \`2\` representing a rotten orange

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return the **minimum number of minutes** that must elapse until no cell has a fresh orange. If this is impossible, return \`-1\`.


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
        'Minute 0: [[2,1,1],[1,1,0],[0,1,1]]\\n' +
        'Minute 1: [[2,2,1],[2,1,0],[0,1,1]]\\n' +
        'Minute 2: [[2,2,2],[2,2,0],[0,1,1]]\\n' +
        'Minute 3: [[2,2,2],[2,2,0],[0,2,1]]\\n' +
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

  leetcodeUrl: 'https://leetcode.com/problems/rotting-oranges/',
  youtubeUrl: 'https://www.youtube.com/watch?v=y704fEOx0s0',
  order: 3,
  topic: 'Breadth-First Search (BFS)',
};
