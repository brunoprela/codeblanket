/**
 * Longest Increasing Path in a Matrix
 * Problem ID: longest-increasing-path
 * Order: 7
 */

import { Problem } from '../../../types';

export const longest_increasing_pathProblem: Problem = {
  id: 'longest-increasing-path',
  title: 'Longest Increasing Path in a Matrix',
  difficulty: 'Medium',
  topic: 'Depth-First Search (DFS)',
  description: `Given an \`m x n\` integers \`matrix\`, return the length of the longest increasing path in \`matrix\`.

From each cell, you can either move in four directions: left, right, up, or down. You **may not** move **diagonally** or move **outside the boundary** (i.e., wrap-around is not allowed).`,
  examples: [
    {
      input: 'matrix = [[9,9,4],[6,6,8],[2,1,1]]',
      output: '4',
      explanation: 'The longest path is [1, 2, 6, 9].',
    },
    {
      input: 'matrix = [[3,4,5],[3,2,6],[2,2,1]]',
      output: '4',
      explanation: 'The longest path is [3, 4, 5, 6].',
    },
  ],
  constraints: [
    'm == matrix.length',
    'n == matrix[i].length',
    '1 <= m, n <= 200',
    '0 <= matrix[i][j] <= 2^31 - 1',
  ],
  hints: [
    'DFS with memoization',
    'Cache longest path from each cell',
    'Only move to cells with larger value',
  ],
  starterCode: `from typing import List

def longest_increasing_path(matrix: List[List[int]]) -> int:
    """
    Find longest increasing path in matrix.
    
    Args:
        matrix: 2D matrix
        
    Returns:
        Length of longest path
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [9, 9, 4],
          [6, 6, 8],
          [2, 1, 1],
        ],
      ],
      expected: 4,
    },
    {
      input: [
        [
          [3, 4, 5],
          [3, 2, 6],
          [2, 2, 1],
        ],
      ],
      expected: 4,
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(m * n)',
  leetcodeUrl:
    'https://leetcode.com/problems/longest-increasing-path-in-a-matrix/',
  youtubeUrl: 'https://www.youtube.com/watch?v=wCc_nd-GiEc',
};
