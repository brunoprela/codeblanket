/**
 * Path With Minimum Effort
 * Problem ID: path-with-minimum-effort
 * Order: 3
 */

import { Problem } from '../../../types';

export const path_with_minimum_effortProblem: Problem = {
  id: 'path-with-minimum-effort',
  title: 'Path With Minimum Effort',
  difficulty: 'Hard',
  description: `You are a hiker preparing for an upcoming hike. You are given a 2D \`heights\` array where \`heights[row][col]\` represents the height of cell \`(row, col)\`. You start at \`(0, 0)\` and want to reach \`(rows-1, cols-1)\`.

A route's **effort** is the maximum absolute difference in heights between two consecutive cells.

Return **the minimum effort** required to travel from the top-left to the bottom-right cell.


**Approach:**
Modified **Dijkstra's** where instead of summing distances, we track the maximum difference encountered on the path. Use min-heap prioritized by maximum effort so far.

**Key Insight:**
This is a "minimax" path problem - minimize the maximum edge weight on the path. Dijkstra works by always exploring the path with smallest maximum effort first.`,
  examples: [
    {
      input: 'heights = [[1,2,2],[3,8,2],[5,3,5]]',
      output: '2',
      explanation:
        'Path [1,3,5,3,5] has maximum difference 2 (from 3 to 5 or 5 to 3).',
    },
    {
      input: 'heights = [[1,2,3],[3,8,4],[5,3,5]]',
      output: '1',
      explanation: 'Path [1,2,3,4,5] has maximum difference 1.',
    },
  ],
  constraints: [
    'rows == heights.length',
    'columns == heights[i].length',
    '1 <= rows, columns <= 100',
    '1 <= heights[i][j] <= 10^6',
  ],
  hints: [
    "Use Dijkstra's but track maximum difference, not sum",
    'Priority queue: (max_effort_so_far, row, col)',
    'For each neighbor, effort = max(current_effort, abs(heights[r][c] - heights[nr][nc]))',
    'Update neighbor if we found a path with smaller maximum effort',
    'Can also use binary search + BFS on effort threshold',
  ],
  starterCode: `from typing import List

def minimum_effort_path(heights: List[List[int]]) -> int:
    """
    Find path with minimum maximum effort.
    
    Args:
        heights: 2D grid of heights
        
    Returns:
        Minimum effort required
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2, 2],
          [3, 8, 2],
          [5, 3, 5],
        ],
      ],
      expected: 2,
    },
    {
      input: [
        [
          [1, 2, 3],
          [3, 8, 4],
          [5, 3, 5],
        ],
      ],
      expected: 1,
    },
    {
      input: [
        [
          [1, 2],
          [3, 8],
        ],
      ],
      expected: 2,
    },
  ],
  solution: `from typing import List
import heapq


def minimum_effort_path(heights: List[List[int]]) -> int:
    """
    Modified Dijkstra for minimax path.
    Time: O(m*n*log(m*n)), Space: O(m*n)
    """
    rows, cols = len(heights), len(heights[0])
    
    # Track minimum effort to reach each cell
    efforts = [[float('inf')] * cols for _ in range(rows)]
    efforts[0][0] = 0
    
    # Min-heap: (max_effort, row, col)
    pq = [(0, 0, 0)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while pq:
        effort, r, c = heapq.heappop(pq)
        
        # Reached destination
        if r == rows - 1 and c == cols - 1:
            return effort
        
        # Skip if we've found better path
        if effort > efforts[r][c]:
            continue
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                # Effort is max difference on path
                new_effort = max(effort, abs(heights[r][c] - heights[nr][nc]))
                
                if new_effort < efforts[nr][nc]:
                    efforts[nr][nc] = new_effort
                    heapq.heappush(pq, (new_effort, nr, nc))
    
    return efforts[rows-1][cols-1]


# Alternative: Binary search + BFS
def minimum_effort_path_binary_search(heights: List[List[int]]) -> int:
    """
    Binary search on effort threshold.
    Time: O(m*n*log(max_height)), Space: O(m*n)
    """
    from collections import deque
    
    rows, cols = len(heights), len(heights[0])
    
    def can_reach(max_effort):
        """Check if destination reachable with effort <= max_effort."""
        visited = set([(0, 0)])
        queue = deque([(0, 0)])
        
        while queue:
            r, c = queue.popleft()
            
            if r == rows - 1 and c == cols - 1:
                return True
            
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    (nr, nc) not in visited):
                    diff = abs(heights[r][c] - heights[nr][nc])
                    if diff <= max_effort:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return False
    
    # Binary search on effort
    left, right = 0, max(max(row) for row in heights)
    
    while left < right:
        mid = (left + right) // 2
        if can_reach(mid):
            right = mid
        else:
            left = mid + 1
    
    return left`,
  timeComplexity: 'O(m*n*log(m*n)) with Dijkstra',
  spaceComplexity: 'O(m*n)',

  leetcodeUrl: 'https://leetcode.com/problems/path-with-minimum-effort/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XQlxCCx2vI4',
  order: 3,
  topic: 'Advanced Graphs',
};
