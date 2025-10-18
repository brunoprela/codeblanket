/**
 * Range Sum Query - Fenwick Tree
 * Problem ID: range-sum-query-fenwick
 * Order: 1
 */

import { Problem } from '../../../types';

export const range_sum_query_fenwickProblem: Problem = {
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
};
