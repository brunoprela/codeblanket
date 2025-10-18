/**
 * Walls and Gates (Multi-Source BFS)
 * Problem ID: queue-walls-gates
 * Order: 6
 */

import { Problem } from '../../../types';

export const walls_gatesProblem: Problem = {
  id: 'queue-walls-gates',
  title: 'Walls and Gates (Multi-Source BFS)',
  difficulty: 'Medium',
  topic: 'Queue',
  description: `You are given an m x n grid initialized with these three possible values:
- **-1** - A wall or an obstacle
- **0** - A gate
- **INF** - An empty room (represented as 2³¹ - 1 = 2147483647)

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

Use multi-source BFS: start from all gates simultaneously and propagate distances outward.`,
  examples: [
    {
      input: '[[INF,-1,0,INF],[INF,INF,INF,-1],[INF,-1,INF,-1],[0,-1,INF,INF]]',
      output: '[[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]',
    },
  ],
  constraints: [
    'm == rooms.length',
    'n == rooms[i].length',
    '1 <= m, n <= 250',
  ],
  hints: [
    'Multi-source BFS: add all gates to queue initially',
    'Process level by level, updating distances',
    'Each cell visited once, so O(mn) time',
    'Directions: up, down, left, right',
  ],
  starterCode: `def walls_and_gates(rooms):
    """
    Fill each empty room with distance to nearest gate using BFS.
    
    Args:
        rooms: 2D grid where -1=wall, 0=gate, INF=empty room
        
    Returns:
        None (modifies rooms in-place)
        
    Note: INF = 2147483647
    """
    pass


# Test case
INF = 2147483647
rooms = [
  [INF, -1, 0, INF],
  [INF, INF, INF, -1],
  [INF, -1, INF, -1],
  [0, -1, INF, INF]
]
walls_and_gates(rooms)
# Expected result:
# [[3, -1, 0, 1],
#  [2, 2, 1, -1],
#  [1, -1, 2, -1],
#  [0, -1, 3, 4]]
`,
  testCases: [
    {
      input: [
        [
          [2147483647, -1, 0, 2147483647],
          [2147483647, 2147483647, 2147483647, -1],
          [2147483647, -1, 2147483647, -1],
          [0, -1, 2147483647, 2147483647],
        ],
      ],
      expected: [
        [3, -1, 0, 1],
        [2, 2, 1, -1],
        [1, -1, 2, -1],
        [0, -1, 3, 4],
      ],
    },
  ],
  solution: `from collections import deque

def walls_and_gates(rooms):
    """Multi-source BFS from all gates"""
    if not rooms or not rooms[0]:
        return
    
    m, n = len(rooms), len(rooms[0])
    INF = 2147483647
    queue = deque()
    
    # Add all gates to queue (multi-source BFS)
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                queue.append((i, j, 0))  # (row, col, distance)
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS
    while queue:
        row, col, dist = queue.popleft()
        
        # Explore 4 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds and if cell is empty room
            if (0 <= new_row < m and 0 <= new_col < n and 
                rooms[new_row][new_col] == INF):
                
                # Update distance
                rooms[new_row][new_col] = dist + 1
                
                # Add to queue for further exploration
                queue.append((new_row, new_col, dist + 1))


# Why multi-source BFS?
# Start from all gates simultaneously
# First time we reach a room = shortest distance
# No need to compare distances - BFS guarantees shortest

# Example walkthrough:
# Initial gates at (0,2) and (3,0)
# Level 0: process gates
# Level 1: cells 1 step from gates
# Level 2: cells 2 steps from gates
# Continue until all reachable cells visited

# Time Complexity: O(m*n) - each cell visited once
# Space Complexity: O(m*n) - queue can have all cells`,
  timeComplexity: 'O(m*n)',
  spaceComplexity: 'O(m*n)',
  followUp: [
    'How would you modify for diagonal movement?',
    'What if walls could be broken (with cost)?',
    'Can you solve with dynamic programming?',
  ],
};
