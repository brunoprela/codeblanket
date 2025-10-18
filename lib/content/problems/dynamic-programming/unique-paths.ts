/**
 * Unique Paths
 * Problem ID: unique-paths
 * Order: 9
 */

import { Problem } from '../../../types';

export const unique_pathsProblem: Problem = {
  id: 'unique-paths',
  title: 'Unique Paths',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `There is a robot on an \`m x n\` grid. The robot is initially located at the **top-left corner** (i.e., \`grid[0][0]\`). The robot tries to move to the **bottom-right corner** (i.e., \`grid[m - 1][n - 1]\`). The robot can only move either down or right at any point in time.

Given the two integers \`m\` and \`n\`, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to \`2 * 10^9\`.`,
  examples: [
    {
      input: 'm = 3, n = 7',
      output: '28',
    },
    {
      input: 'm = 3, n = 2',
      output: '3',
      explanation:
        'From the top-left corner, there are a total of 3 ways to reach the bottom-right corner: Right -> Down -> Down, Down -> Down -> Right, Down -> Right -> Down.',
    },
  ],
  constraints: ['1 <= m, n <= 100'],
  hints: [
    'dp[i][j] = ways to reach cell (i, j)',
    'dp[i][j] = dp[i-1][j] + dp[i][j-1]',
    'Base case: dp[0][j] = dp[i][0] = 1',
  ],
  starterCode: `def unique_paths(m: int, n: int) -> int:
    """
    Find number of unique paths in m x n grid.
    
    Args:
        m: Number of rows
        n: Number of columns
        
    Returns:
        Number of unique paths
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [3, 7],
      expected: 28,
    },
    {
      input: [3, 2],
      expected: 3,
    },
    {
      input: [1, 1],
      expected: 1,
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(n) with space optimization',
  leetcodeUrl: 'https://leetcode.com/problems/unique-paths/',
  youtubeUrl: 'https://www.youtube.com/watch?v=IlEsdxuD4lY',
};
