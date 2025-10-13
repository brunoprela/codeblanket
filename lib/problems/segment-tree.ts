import { Problem } from '../types';

export const segmentTreeProblems: Problem[] = [
  {
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
  },
  {
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
  },
  {
    id: 'count-of-smaller-after-self',
    title: 'Count of Smaller Numbers After Self',
    difficulty: 'Hard',
    description: `Given an integer array \`nums\`, return an integer array \`counts\` where \`counts[i]\` is the number of smaller elements to the right of \`nums[i]\`.


**Key Insight:**
Process array from right to left, building a segment tree of seen values. For each element, query how many smaller values have been seen.`,
    examples: [
      {
        input: 'nums = [5,2,6,1]',
        output: '[2,1,1,0]',
        explanation:
          '5: 2 smaller (2,1), 2: 1 smaller (1), 6: 1 smaller (1), 1: 0 smaller',
      },
    ],
    constraints: ['1 <= nums.length <= 10^5', '-10^4 <= nums[i] <= 10^4'],
    hints: [
      'Process array from right to left',
      'Use segment tree to count values seen so far',
      'Coordinate compression for large range',
      'Query range [min_val, nums[i]-1] for count',
    ],
    starterCode: `from typing import List

def count_smaller(nums: List[int]) -> List[int]:
    """
    Return count of smaller elements to the right.
    
    Args:
        nums: Input array
        
    Returns:
        Array where result[i] = count of smaller to right
    """
    pass
`,
    testCases: [
      {
        input: [[5, 2, 6, 1]],
        expected: [2, 1, 1, 0],
      },
      {
        input: [[-1]],
        expected: [0],
      },
    ],
    solution: `from typing import List


class SegmentTree:
    """Segment tree for counting elements"""
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (4 * size)
    
    def update(self, node, start, end, idx):
        """Increment count at position idx"""
        if start == end:
            self.tree[node] += 1
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self.update(2*node+1, start, mid, idx)
        else:
            self.update(2*node+2, mid+1, end, idx)
        
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, node, start, end, L, R):
        """Count elements in range [L, R]"""
        if R < start or L > end or L > R:
            return 0
        if L <= start and end <= R:
            return self.tree[node]
        
        mid = (start + end) // 2
        return (self.query(2*node+1, start, mid, L, R) + 
                self.query(2*node+2, mid+1, end, L, R))


def count_smaller(nums: List[int]) -> List[int]:
    """
    Use segment tree with coordinate compression.
    Time: O(N log N), Space: O(N)
    """
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    rank = {v: i for i, v in enumerate(sorted_nums)}
    
    n = len(nums)
    result = [0] * n
    st = SegmentTree(len(sorted_nums))
    
    # Process from right to left
    for i in range(n - 1, -1, -1):
        # Count smaller elements seen so far
        r = rank[nums[i]]
        if r > 0:
            result[i] = st.query(0, 0, st.size-1, 0, r-1)
        
        # Add current element
        st.update(0, 0, st.size-1, r)
    
    return result`,
    timeComplexity: 'O(N log N)',
    spaceComplexity: 'O(N)',
    
    leetcodeUrl: 'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
    youtubeUrl: 'https://www.youtube.com/watch?v=2SVLYsq5W8M',
    order: 3,
    topic: 'Segment Tree',
    leetcodeUrl:
      'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ZBHKZF5w4YU',
  },
];
