/**
 * Range Sum Query - Mutable
 * Problem ID: range-sum-query-mutable
 * Order: 1
 */

import { Problem } from '../../../types';

export const range_sum_query_mutableProblem: Problem = {
  id: 'range-sum-query-mutable',
  title: 'Range Sum Query - Mutable',
  difficulty: 'Medium',
  description: `Given an integer array \`nums\`, handle multiple queries of the following types:

1. **Update** the value of an element in \`nums\`.
2. Calculate the **sum** of the elements of \`nums\` between indices \`left\` and \`right\` **inclusive**.

Implement the \`NumArray\` class with \`update(index, val)\` and \`sumRange(left, right)\` methods.


**Key Insight:**
Use a Segment Tree to achieve O(log N) for both updates and range sum queries.`,
  examples: [
    {
      input:
        '["NumArray", "sumRange", "update", "sumRange"]\n[[[1,3,5]], [0,2], [1,2], [0,2]]',
      output: '[null, 9, null, 8]',
      explanation:
        'sumRange(0,2)=9, update(1,2) changes [1,3,5] to [1,2,5], sumRange(0,2)=8',
    },
  ],
  constraints: [
    '1 <= nums.length <= 3 * 10^4',
    '-100 <= nums[i] <= 100',
    '0 <= index < nums.length',
    '-100 <= val <= 100',
    '0 <= left <= right < nums.length',
    'At most 3 * 10^4 calls to update and sumRange',
  ],
  hints: [
    'Segment Tree provides O(log N) for both operations',
    'Each node stores sum of its range',
    'Build tree in O(N), query/update in O(log N)',
  ],
  starterCode: `class NumArray:
    def __init__(self, nums: list[int]):
        """Initialize with array nums"""
        pass
    
    def update(self, index: int, val: int) -> None:
        """Update nums[index] to val"""
        pass
    
    def sum_range(self, left: int, right: int) -> int:
        """Return sum of nums[left..right]"""
        pass
`,
  testCases: [
    {
      input: [
        [1, 3, 5],
        ['sumRange', 0, 2],
        ['update', 1, 2],
        ['sumRange', 0, 2],
      ],
      expected: [9, null, 8],
    },
  ],
  solution: `class NumArray:
    """Segment Tree implementation for range sum with updates"""
    
    def __init__(self, nums: list[int]):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self._build(nums, 0, 0, self.n - 1)
    
    def _build(self, nums, node, start, end):
        if start == end:
            self.tree[node] = nums[start]
            return
        
        mid = (start + end) // 2
        self._build(nums, 2*node+1, start, mid)
        self._build(nums, 2*node+2, mid+1, end)
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
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
        
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def sum_range(self, left: int, right: int) -> int:
        return self._query(0, 0, self.n-1, left, right)
    
    def _query(self, node, start, end, L, R):
        if R < start or L > end:
            return 0
        if L <= start and end <= R:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_sum = self._query(2*node+1, start, mid, L, R)
        right_sum = self._query(2*node+2, mid+1, end, L, R)
        return left_sum + right_sum`,
  timeComplexity: 'O(log N) per operation',
  spaceComplexity: 'O(N)',
  order: 1,
  topic: 'Segment Tree',
  leetcodeUrl: 'https://leetcode.com/problems/range-sum-query-mutable/',
  youtubeUrl: 'https://www.youtube.com/watch?v=rYBtViWXYeI',
};
