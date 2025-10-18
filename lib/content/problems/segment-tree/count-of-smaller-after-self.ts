/**
 * Count of Smaller Numbers After Self
 * Problem ID: count-of-smaller-after-self
 * Order: 3
 */

import { Problem } from '../../../types';

export const count_of_smaller_after_selfProblem: Problem = {
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
  order: 3,
  topic: 'Segment Tree',
  leetcodeUrl:
    'https://leetcode.com/problems/count-of-smaller-numbers-after-self/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ZBHKZF5w4YU',
};
