/**
 * 2D Range Sum Query - Mutable
 * Problem ID: range-sum-2d-mutable
 * Order: 3
 */

import { Problem } from '../../../types';

export const range_sum_2d_mutableProblem: Problem = {
  id: 'range-sum-2d-mutable',
  title: '2D Range Sum Query - Mutable',
  difficulty: 'Hard',
  description: `Implement a 2D Fenwick Tree for range sum queries and updates on a matrix.

Operations:
- \`update(row, col, val)\`: Update matrix[row][col] to val
- \`sumRegion(row1, col1, row2, col2)\`: Sum of rectangle

**Key Insight:**
2D Fenwick Tree extends 1D idea - update and query both use nested loops with bit manipulation.`,
  examples: [
    {
      input: 'matrix = [[1,2],[3,4]], sumRegion(0,0,1,1)',
      output: '10',
      explanation: '1+2+3+4 = 10',
    },
  ],
  constraints: ['1 <= matrix.length <= 200', '1 <= matrix[0].length <= 200'],
  hints: [
    'Extend 1D Fenwick to 2D',
    'Both row and column use bit manipulation',
    'Use inclusion-exclusion for rectangle sum',
  ],
  starterCode: `from typing import List

class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        pass
    
    def update(self, row: int, col: int, val: int) -> None:
        pass
    
    def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
        pass
`,
  testCases: [
    {
      input: [
        [
          [1, 2],
          [3, 4],
        ],
      ],
      expected: [10],
    },
  ],
  solution: `from typing import List


class NumMatrix:
    """2D Fenwick Tree"""
    
    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            return
        
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.matrix = [[0] * self.cols for _ in range(self.rows)]
        self.tree = [[0] * (self.cols + 1) for _ in range(self.rows + 1)]
        
        for r in range(self.rows):
            for c in range(self.cols):
                self.update(r, c, matrix[r][c])
    
    def _add(self, r: int, c: int, delta: int):
        """Add delta to (r, c) in tree"""
        i = r + 1
        while i <= self.rows:
            j = c + 1
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i
    
    def _query(self, r: int, c: int) -> int:
        """Sum of rectangle from (0,0) to (r,c)"""
        total = 0
        i = r + 1
        while i > 0:
            j = c + 1
            while j > 0:
                total += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return total
    
    def update(self, row: int, col: int, val: int) -> None:
        delta = val - self.matrix[row][col]
        self.matrix[row][col] = val
        self._add(row, col, delta)
    
    def sum_region(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Sum of rectangle using inclusion-exclusion"""
        return (self._query(row2, col2) - 
                self._query(row1-1, col2) - 
                self._query(row2, col1-1) + 
                self._query(row1-1, col1-1))`,
  timeComplexity: 'O(log N × log M) per operation',
  spaceComplexity: 'O(N × M)',
  order: 3,
  topic: 'Fenwick Tree (Binary Indexed Tree)',
  leetcodeUrl: 'https://leetcode.com/problems/range-sum-query-2d-mutable/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ufZinoNHaUU',
};
