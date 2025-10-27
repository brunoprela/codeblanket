/**
 * Rotate Image
 * Problem ID: rotate-image
 * Order: 1
 */

import { Problem } from '../../../types';

export const rotate_imageProblem: Problem = {
  id: 'rotate-image',
  title: 'Rotate Image',
  difficulty: 'Easy',
  description: `You are given an \`n x n\` 2D matrix representing an image, rotate the image by **90 degrees (clockwise)**.

You have to rotate the image **in-place**, which means you have to modify the input 2D matrix directly.


**Approach:**1. Transpose the matrix (swap rows with columns)
2. Reverse each row

**Key Insight:**
90° clockwise rotation = transpose + reverse each row`,
  examples: [
    {
      input: 'matrix = [[1,2,3],[4,5,6],[7,8,9]]',
      output: '[[7,4,1],[8,5,2],[9,6,3]]',
    },
  ],
  constraints: ['n == matrix.length == matrix[i].length', '1 <= n <= 20'],
  hints: [
    'Transpose: swap matrix[i][j] with matrix[j][i]',
    'Then reverse each row',
  ],
  starterCode: `from typing import List

def rotate(matrix: List[List[int]]) -> None:
    """Rotate matrix 90° clockwise in-place."""
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
      expected: [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3],
      ],
    },
  ],
  solution: `from typing import List

def rotate(matrix: List[List[int]]) -> None:
    """Time: O(n²), Space: O(1)"""
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()
`,
  timeComplexity: 'O(n²)',
  spaceComplexity: 'O(1)',
  order: 1,
  topic: 'Math & Geometry',
  leetcodeUrl: 'https://leetcode.com/problems/rotate-image/',
  youtubeUrl: 'https://www.youtube.com/watch?v=fMSJSS7eO1w',
};
