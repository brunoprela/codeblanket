import { Problem } from '../types';

export const fenwickTreeProblems: Problem[] = [
  {
    id: 'range-sum-query-fenwick',
    title: 'Range Sum Query - Fenwick Tree',
    difficulty: 'Medium',
    description: `Implement a data structure that supports range sum queries and point updates using a Fenwick Tree (Binary Indexed Tree).

Implement the \`NumArray\` class:
- \`update(index, val)\`: Update the value at index to val
- \`sumRange(left, right)\`: Return the sum of elements between indices left and right


**Key Insight:**
Fenwick Tree provides O(log N) for both operations using bit manipulation tricks.`,
    examples: [
      {
        input:
          '["NumArray","sumRange","update","sumRange"] [[[1,3,5]],[0,2],[1,2],[0,2]]',
        output: '[null,9,null,8]',
      },
    ],
    constraints: ['1 <= nums.length <= 3 * 10^4'],
    hints: [
      'Use Fenwick Tree (Binary Indexed Tree)',
      'Remember 1-indexing for Fenwick Tree',
      'Use i & -i to get last set bit',
    ],
    starterCode: `class NumArray:
    def __init__(self, nums: list[int]):
        pass
    
    def update(self, index: int, val: int) -> None:
        pass
    
    def sum_range(self, left: int, right: int) -> int:
        pass
`,
    testCases: [
      {
        input: [[1, 3, 5]],
        expected: [9, 8],
      },
    ],
    solution: `class NumArray:
    """Fenwick Tree implementation"""
    
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.nums = [0] + nums  # 1-indexed original array
        self.tree = [0] * (self.n + 1)
        
        # Build tree
        for i in range(1, self.n + 1):
            self._add(i, nums[i-1])
    
    def _add(self, i: int, delta: int):
        """Add delta to index i"""
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
    
    def _prefix_sum(self, i: int) -> int:
        """Get sum from 1 to i"""
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total
    
    def update(self, index: int, val: int) -> None:
        """Update to new value (not add delta)"""
        delta = val - self.nums[index + 1]
        self.nums[index + 1] = val
        self._add(index + 1, delta)
    
    def sum_range(self, left: int, right: int) -> int:
        """Sum from left to right (0-indexed)"""
        return self._prefix_sum(right + 1) - self._prefix_sum(left)`,
    timeComplexity: 'O(log N)',
    spaceComplexity: 'O(N)',
    order: 1,
    topic: 'Fenwick Tree (Binary Indexed Tree)',
    leetcodeUrl: 'https://leetcode.com/problems/range-sum-query-mutable/',
    youtubeUrl: 'https://www.youtube.com/watch?v=CWDQJGaN1gY',
  },
  {
    id: 'count-inversions',
    title: 'Count of Inversions',
    difficulty: 'Hard',
    description: `Count the number of inversions in an array. An inversion is a pair of indices (i, j) where i < j but arr[i] > arr[j].

**Approach:**
Use Fenwick Tree with coordinate compression:
1. Process array from right to left
2. For each element, query count of smaller values seen
3. Add current element to Fenwick Tree

**Key Insight:**
Fenwick Tree efficiently counts how many smaller elements we've seen so far.`,
    examples: [
      {
        input: 'arr = [5, 3, 2, 4, 1]',
        output: '8',
        explanation:
          'Inversions: (5,3), (5,2), (5,4), (5,1), (3,2), (3,1), (2,1), (4,1)',
      },
    ],
    constraints: ['1 <= arr.length <= 10^5', '-10^9 <= arr[i] <= 10^9'],
    hints: [
      'Process from right to left',
      'Use coordinate compression for large values',
      'Fenwick tree counts smaller values seen',
    ],
    starterCode: `from typing import List

def count_inversions(arr: List[int]) -> int:
    """Count number of inversions in array"""
    pass
`,
    testCases: [
      {
        input: [[5, 3, 2, 4, 1]],
        expected: 8,
      },
      {
        input: [[1, 2, 3, 4, 5]],
        expected: 0,
      },
    ],
    solution: `from typing import List


class FenwickTree:
    def __init__(self, n):
        self.tree = [0] * (n + 1)
    
    def update(self, i):
        while i < len(self.tree):
            self.tree[i] += 1
            i += i & -i
    
    def query(self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def count_inversions(arr: List[int]) -> int:
    """
    Count inversions using Fenwick Tree.
    Time: O(N log N), Space: O(N)
    """
    # Coordinate compression
    sorted_arr = sorted(set(arr))
    rank = {v: i+1 for i, v in enumerate(sorted_arr)}
    
    n = len(sorted_arr)
    ft = FenwickTree(n)
    inversions = 0
    
    # Process from right to left
    for num in reversed(arr):
        r = rank[num]
        # Count smaller elements seen so far
        if r > 1:
            inversions += ft.query(r - 1)
        # Add current element
        ft.update(r)
    
    return inversions`,
    timeComplexity: 'O(N log N)',
    spaceComplexity: 'O(N)',

    leetcodeUrl: 'https://leetcode.com/problems/count-of-range-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=kPaJfAUwViY',
    order: 2,
    topic: 'Fenwick Tree (Binary Indexed Tree)',
    leetcodeUrl: 'https://leetcode.com/problems/count-of-range-sum/',
    youtubeUrl: 'https://www.youtube.com/watch?v=kPaJfAUwViY',
  },
  {
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
  },
];
