/**
 * Range Minimum Query
 * Problem ID: range-minimum-query
 * Order: 2
 */

import { Problem } from '../../../types';

export const range_minimum_queryProblem: Problem = {
  id: 'range-minimum-query',
  title: 'Range Minimum Query',
  difficulty: 'Medium',
  description: `Implement a data structure that supports range minimum queries and updates.

**Operations:**
- \`update(index, val)\`: Update element at index to val
- \`rangeMin(left, right)\`: Return minimum element in range [left, right]


**Key Insight:**
Use Segment Tree with min operation instead of sum. Return infinity for no-overlap case.`,
  examples: [
    {
      input: 'nums = [1,3,5,7,9], rangeMin(1,3)',
      output: '3',
      explanation: 'Minimum of [3,5,7] is 3',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
  hints: [
    'Similar to range sum, but use min operation',
    'Return float("inf") for no-overlap case',
    'Merge children using min(left, right)',
  ],
  starterCode: `class RangeMinQuery:
    def __init__(self, nums: list[int]):
        pass
    
    def update(self, index: int, val: int) -> None:
        pass
    
    def range_min(self, left: int, right: int) -> int:
        pass
`,
  testCases: [
    {
      input: [
        [1, 3, 5, 7, 9],
        ['rangeMin', 1, 3],
      ],
      expected: [3],
    },
  ],
  solution: `class RangeMinQuery:
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [float('inf')] * (4 * self.n)
        self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
            return
        
        mid = (start + end) // 2
        self._build(nums, 2*node+1, start, mid)
        self._build(nums, 2*node+2, mid+1, end)
        self.tree[node] = min(self.tree[2*node+1], self.tree[2*node+2])
    
    def update(self, index: int, val: int) -> None:
        self._update(0, 0, self.n-1, index, val)
    
    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2*node+1, start, mid, idx, val)
        else:
            self._update(2*node+2, mid+1, end, idx, val)
        
        self.tree[node] = min(self.tree[2*node+1], self.tree[2*node+2])
    
    def range_min(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n-1, left, right)
    
    def _query(self, node, start, end, L, R):
        if R < start or L > end:
            return float('inf')
        if L <= start and end <= R:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_min = self._query(2*node+1, start, mid, L, R)
        right_min = self._query(2*node+2, mid+1, end, L, R)
        return min(left_min, right_min)`,
  timeComplexity: 'O(log N)',
  spaceComplexity: 'O(N)',
  order: 2,
  topic: 'Segment Tree',
  leetcodeUrl: 'https://leetcode.com/problems/range-minimum-query-mutable/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Oq2E2yGadnU',
};
