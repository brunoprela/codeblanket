/**
 * Spiral Matrix
 * Problem ID: spiral-matrix
 * Order: 9
 */

import { Problem } from '../../../types';

export const spiral_matrixProblem: Problem = {
  id: 'spiral-matrix',
  title: 'Spiral Matrix',
  difficulty: 'Medium',
  topic: 'Math & Geometry',
  description: `Given an \`m x n\` matrix, return all elements of the matrix in spiral order.`,
  examples: [
    {
      input: 'matrix = [[1,2,3],[4,5,6],[7,8,9]]',
      output: '[1,2,3,6,9,8,7,4,5]',
    },
    {
      input: 'matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]',
      output: '[1,2,3,4,8,12,11,10,9,5,6,7]',
    },
  ],
  constraints: [
    'm == matrix.length',
    'n == matrix[i].length',
    '1 <= m, n <= 10',
    '-100 <= matrix[i][j] <= 100',
  ],
  hints: [
    'Track boundaries: top, bottom, left, right',
    'Move right, down, left, up in sequence',
    'Shrink boundaries after each direction',
  ],
  starterCode: `from typing import List

def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    Traverse matrix in spiral order.
    
    Args:
        matrix: 2D matrix
        
    Returns:
        Elements in spiral order
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      ],
      expected: [1, 2, 3, 6, 9, 8, 7, 4, 5],
    },
    {
      input: [
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
      ],
      expected: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7],
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/spiral-matrix/',
  youtubeUrl: 'https://www.youtube.com/watch?v=BJnMZNwUk1M',
};
