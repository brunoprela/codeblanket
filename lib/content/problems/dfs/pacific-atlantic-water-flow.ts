/**
 * Pacific Atlantic Water Flow
 * Problem ID: pacific-atlantic-water-flow
 * Order: 3
 */

import { Problem } from '../../../types';

export const pacific_atlantic_water_flowProblem: Problem = {
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
      explanation: 'The only cell touches both oceans so water flows to both',
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

  leetcodeUrl: 'https://leetcode.com/problems/pacific-atlantic-water-flow/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s-VkcjHqkGI',
  order: 3,
  topic: 'Depth-First Search (DFS)',
};
