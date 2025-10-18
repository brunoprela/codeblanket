/**
 * Set Matrix Zeroes
 * Problem ID: set-matrix-zeroes
 * Order: 18
 */

import { Problem } from '../../../types';

export const set_matrix_zeroesProblem: Problem = {
  id: 'set-matrix-zeroes',
  title: 'Set Matrix Zeroes',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  order: 18,
  description: `Given an \`m x n\` integer matrix \`matrix\`, if an element is \`0\`, set its entire row and column to \`0\`'s.

You must do it **in place**.`,
  examples: [
    {
      input: 'matrix = [[1,1,1],[1,0,1],[1,1,1]]',
      output: '[[1,0,1],[0,0,0],[1,0,1]]',
    },
    {
      input: 'matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]',
      output: '[[0,0,0,0],[0,4,5,0],[0,3,1,0]]',
    },
  ],
  constraints: [
    'm == matrix.length',
    'n == matrix[0].length',
    '1 <= m, n <= 200',
    '-2^31 <= matrix[i][j] <= 2^31 - 1',
  ],
  hints: [
    'Use first row and first column as markers',
    'Need separate variable for first cell',
    'Process matrix in two passes',
  ],
  starterCode: `from typing import List

def set_zeroes(matrix: List[List[int]]) -> None:
    """
    Set entire row and column to zero if element is zero.
    Modify matrix in-place.
    
    Args:
        matrix: m x n matrix
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1],
        ],
      ],
      expected: [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1],
      ],
    },
    {
      input: [
        [
          [0, 1, 2, 0],
          [3, 4, 5, 2],
          [1, 3, 1, 5],
        ],
      ],
      expected: [
        [0, 0, 0, 0],
        [0, 4, 5, 0],
        [0, 3, 1, 0],
      ],
    },
  ],
  solution: `from typing import List

def set_zeroes(matrix: List[List[int]]) -> None:
    """
    Use first row/col as markers.
    Time: O(m * n), Space: O(1)
    """
    m, n = len(matrix), len(matrix[0])
    first_row_zero = False
    first_col_zero = False
    
    # Check if first row has zero
    for j in range(n):
        if matrix[0][j] == 0:
            first_row_zero = True
            break
    
    # Check if first column has zero
    for i in range(m):
        if matrix[i][0] == 0:
            first_col_zero = True
            break
    
    # Use first row and column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
`,
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/set-matrix-zeroes/',
  youtubeUrl: 'https://www.youtube.com/watch?v=T41rL0L3Pnw',
};
