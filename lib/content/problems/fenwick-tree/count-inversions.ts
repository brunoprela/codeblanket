/**
 * Count of Inversions
 * Problem ID: count-inversions
 * Order: 2
 */

import { Problem } from '../../../types';

export const count_inversionsProblem: Problem = {
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
};
