/**
 * Count Negative Numbers in a Sorted Matrix
 * Problem ID: count-negative-numbers-sorted-matrix
 * Order: 8
 */

import { Problem } from '../../../types';

export const count_negative_numbers_sorted_matrixProblem: Problem = {
  id: 'count-negative-numbers-sorted-matrix',
  title: 'Count Negative Numbers in a Sorted Matrix',
  difficulty: 'Easy',
  topic: 'Binary Search',
  description: `Given a \`m x n\` matrix \`grid\` which is sorted in non-increasing order both row-wise and column-wise, return the number of negative numbers in \`grid\`.`,
  examples: [
    {
      input: 'grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]',
      output: '8',
      explanation: 'There are 8 negatives number in the matrix.',
    },
    {
      input: 'grid = [[3,2],[1,0]]',
      output: '0',
    },
  ],
  constraints: [
    'm == grid.length',
    'n == grid[i].length',
    '1 <= m, n <= 100',
    '-100 <= grid[i][j] <= 100',
  ],
  hints: [
    'Use binary search for each row',
    'The matrix is sorted, so once you find a negative number, all numbers to its right are also negative',
  ],
  starterCode: `from typing import List

def count_negatives(grid: List[List[int]]) -> int:
    """
    Count negative numbers in sorted matrix.
    
    Args:
        grid: Matrix sorted in non-increasing order
        
    Returns:
        Count of negative numbers
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          [4, 3, 2, -1],
          [3, 2, 1, -1],
          [1, 1, -1, -2],
          [-1, -1, -2, -3],
        ],
      ],
      expected: 8,
    },
    {
      input: [
        [
          [3, 2],
          [1, 0],
        ],
      ],
      expected: 0,
    },
    {
      input: [
        [
          [1, -1],
          [-1, -1],
        ],
      ],
      expected: 3,
    },
  ],
  timeComplexity: 'O(m log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/',
  youtubeUrl: 'https://www.youtube.com/watch?v=5BI4BxoVlLo',
};
