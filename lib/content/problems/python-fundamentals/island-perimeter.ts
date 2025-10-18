/**
 * Island Perimeter
 * Problem ID: fundamentals-island-perimeter
 * Order: 89
 */

import { Problem } from '../../../types';

export const island_perimeterProblem: Problem = {
  id: 'fundamentals-island-perimeter',
  title: 'Island Perimeter',
  difficulty: 'Easy',
  description: `Calculate perimeter of island in a grid.

Grid: 1 = land, 0 = water
Island: connected 1s (no diagonal connections)

**Example:** 
\`\`\`
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]
\`\`\`
Perimeter = 16

This tests:
- 2D array traversal
- Neighbor checking
- Counting`,
  examples: [
    {
      input: 'grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]',
      output: '16',
    },
  ],
  constraints: ['1 <= rows, cols <= 100', 'Only one island'],
  hints: [
    'Each land cell contributes 4 to perimeter',
    'Subtract 2 for each shared edge',
    'Check 4 neighbors',
  ],
  starterCode: `def island_perimeter(grid):
    """
    Calculate island perimeter.
    
    Args:
        grid: 2D array (1=land, 0=water)
        
    Returns:
        Perimeter of island
        
    Examples:
        >>> island_perimeter([[0,1,0,0],[1,1,1,0]])
        12
    """
    pass


# Test
print(island_perimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]))
`,
  testCases: [
    {
      input: [
        [
          [0, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 1, 0, 0],
          [1, 1, 0, 0],
        ],
      ],
      expected: 16,
    },
    {
      input: [[[1]]],
      expected: 4,
    },
  ],
  solution: `def island_perimeter(grid):
    perimeter = 0
    rows, cols = len(grid), len(grid[0])
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                # Start with 4 sides
                perimeter += 4
                
                # Subtract shared edges
                if i > 0 and grid[i-1][j] == 1:  # Up
                    perimeter -= 2
                if j > 0 and grid[i][j-1] == 1:  # Left
                    perimeter -= 2
    
    return perimeter`,
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(1)',
  order: 89,
  topic: 'Python Fundamentals',
};
