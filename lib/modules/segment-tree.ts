import { Module } from '@/lib/types';

export const segmentTreeModule: Module = {
  id: 'segment-tree',
  title: 'Segment Tree',
  description:
    'Master segment trees for efficient range queries and updates on arrays.',
  icon: '🌲',
  timeComplexity: 'O(log N) for queries and updates',
  spaceComplexity: 'O(N)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Segment Trees',
      content: `**Segment Tree** is a tree data structure for storing intervals (segments). It allows efficient querying and updating of array intervals.

**Problem it Solves:**
Given an array, perform these operations efficiently:
- **Range Query**: Find sum/min/max/gcd of elements in range [L, R]
- **Point Update**: Update a single element
- **Range Update**: Update all elements in a range (with lazy propagation)

**Why Segment Trees:**
- Range queries: O(log N) instead of O(N)
- Point updates: O(log N) instead of O(1) simple array
- Balance between query and update efficiency

**When to Use:**
- Multiple range queries on static or dynamic arrays
- Need both queries and updates
- Range sum, min, max, GCD, etc.
- Problems with intervals/segments

**Alternative Data Structures:**
- **Prefix Sum**: Fast queries (O(1)) but can't handle updates
- **Fenwick Tree**: Simpler, but limited to associative operations
- **Sqrt Decomposition**: Simpler but slower (O(√N))`,
    },
    {
      id: 'structure',
      title: 'Segment Tree Structure',
      content: `**Tree Representation:**
- Each node represents an interval/segment
- Leaf nodes represent single elements
- Internal nodes represent unions of child intervals
- Root represents entire array [0, N-1]

**Example:** Array = [1, 3, 5, 7, 9, 11]

\`\`\`
                  [0,5] = 36
                 /          \\
            [0,2] = 9      [3,5] = 27
           /      \\         /      \\
       [0,1]=4  [2]=5  [3,4]=16  [5]=11
       /    \\           /    \\
    [0]=1  [1]=3    [3]=7  [4]=9
\`\`\`

**Array Representation:**
Store tree in array of size 4N (to be safe).
\`\`\`python
tree = [0] * (4 * n)
\`\`\`

**Index Mapping:**
- Node i has:
  - Left child: 2*i + 1
  - Right child: 2*i + 2
  - Parent: (i-1) // 2

**Building the Tree:**
\`\`\`python
def build(arr, tree, node, start, end):
    """Build segment tree recursively"""
    if start == end:
        # Leaf node
        tree[node] = arr[start]
        return
    
    mid = (start + end) // 2
    left_child = 2 * node + 1
    right_child = 2 * node + 2
    
    # Build left and right subtrees
    build(arr, tree, left_child, start, mid)
    build(arr, tree, right_child, mid + 1, end)
    
    # Internal node = merge of children
    tree[node] = tree[left_child] + tree[right_child]
\`\`\`

**Complexity:**
- Build: O(N)
- Space: O(N)`,
    },
    {
      id: 'operations',
      title: 'Core Operations',
      content: `**1. Range Query**
Find value for range [L, R].

**Three Cases:**
1. **No overlap**: Range completely outside [L, R]
2. **Complete overlap**: Range completely inside [L, R]
3. **Partial overlap**: Range partially overlaps [L, R]

\`\`\`python
def query(tree, node, start, end, L, R):
    """Query sum in range [L, R]"""
    # No overlap
    if R < start or L > end:
        return 0
    
    # Complete overlap
    if L <= start and end <= R:
        return tree[node]
    
    # Partial overlap - query both children
    mid = (start + end) // 2
    left_sum = query(tree, 2*node+1, start, mid, L, R)
    right_sum = query(tree, 2*node+2, mid+1, end, L, R)
    return left_sum + right_sum
\`\`\`

**2. Point Update**
Update single element at index.

\`\`\`python
def update(tree, node, start, end, idx, value):
    """Update element at index idx to value"""
    if start == end:
        # Leaf node
        tree[node] = value
        return
    
    mid = (start + end) // 2
    
    if idx <= mid:
        update(tree, 2*node+1, start, mid, idx, value)
    else:
        update(tree, 2*node+2, mid+1, end, idx, value)
    
    # Update current node
    tree[node] = tree[2*node+1] + tree[2*node+2]
\`\`\`

**Complexity:**
- Query: O(log N)
- Update: O(log N)`,
      codeExample: `class SegmentTree:
    """Segment Tree for range sum queries"""
    
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr, node, start, end):
        """Build tree recursively"""
        if start == end:
            self.tree[node] = arr[start]
            return
        
        mid = (start + end) // 2
        self._build(arr, 2*node+1, start, mid)
        self._build(arr, 2*node+2, mid+1, end)
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, L, R):
        """Query sum in range [L, R]"""
        return self._query(0, 0, self.n-1, L, R)
    
    def _query(self, node, start, end, L, R):
        # No overlap
        if R < start or L > end:
            return 0
        
        # Complete overlap
        if L <= start and end <= R:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_sum = self._query(2*node+1, start, mid, L, R)
        right_sum = self._query(2*node+2, mid+1, end, L, R)
        return left_sum + right_sum
    
    def update(self, idx, value):
        """Update element at index to value"""
        self._update(0, 0, self.n-1, idx, value)
    
    def _update(self, node, start, end, idx, value):
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2*node+1, start, mid, idx, value)
        else:
            self._update(2*node+2, mid+1, end, idx, value)
        
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]


# Example usage
arr = [1, 3, 5, 7, 9, 11]
st = SegmentTree(arr)
print(st.query(1, 3))  # Sum of [3, 5, 7] = 15
st.update(1, 10)        # Change arr[1] = 3 to 10
print(st.query(1, 3))  # Sum of [10, 5, 7] = 22`,
    },
    {
      id: 'variations',
      title: 'Segment Tree Variations',
      content: `**Range Minimum Query (RMQ):**
\`\`\`python
# In build and update, use:
tree[node] = min(tree[left_child], tree[right_child])

# In query for no overlap, return:
return float('inf')  # Instead of 0
\`\`\`

**Range Maximum Query:**
\`\`\`python
# Use max instead of min/sum
tree[node] = max(tree[left_child], tree[right_child])

# In query for no overlap:
return float('-inf')
\`\`\`

**Range GCD:**
\`\`\`python
from math import gcd

# In merge:
tree[node] = gcd(tree[left_child], tree[right_child])

# In query for no overlap:
return 0
\`\`\`

**Lazy Propagation (Range Updates):**
For efficient range updates, use lazy propagation to defer updates until needed.

\`\`\`python
class LazySegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)
    
    def _push(self, node, start, end):
        """Push lazy value to children"""
        if self.lazy[node] != 0:
            self.tree[node] += (end - start + 1) * self.lazy[node]
            
            if start != end:  # Not leaf
                self.lazy[2*node+1] += self.lazy[node]
                self.lazy[2*node+2] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def range_update(self, L, R, value):
        """Add value to all elements in [L, R]"""
        self._range_update(0, 0, self.n-1, L, R, value)
    
    def _range_update(self, node, start, end, L, R, value):
        self._push(node, start, end)
        
        if R < start or L > end:
            return
        
        if L <= start and end <= R:
            self.lazy[node] += value
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._range_update(2*node+1, start, mid, L, R, value)
        self._range_update(2*node+2, mid+1, end, L, R, value)
        
        self._push(2*node+1, start, mid)
        self._push(2*node+2, mid+1, end)
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
\`\`\``,
    },
    {
      id: 'complexity',
      title: 'Complexity Analysis',
      content: `**Time Complexity:**

**Build:** O(N)
- Visit each node once
- Total nodes = 2N - 1 ≈ O(N)

**Query:** O(log N)
- At most visit O(log N) levels
- Each level: constant work
- Can visit at most 4 nodes per level

**Update:** O(log N)
- Path from root to leaf
- Height = log N

**Range Update (with lazy):** O(log N)
- Defer actual updates
- Push down only when needed

**Space Complexity:**
- **Array size:** 4N
- **Why 4N?**
  - Complete binary tree needs 2N - 1 nodes
  - Array representation wastes some space
  - 4N is safe upper bound
  - For exact: use 2^(ceil(log2(N)) + 1)

**Comparison:**

| Operation | Array | Prefix Sum | Segment Tree | Fenwick Tree |
|-----------|-------|------------|--------------|--------------|
| Build | O(N) | O(N) | O(N) | O(N log N) |
| Query | O(N) | O(1) | O(log N) | O(log N) |
| Update | O(1) | O(N) | O(log N) | O(log N) |
| Space | O(N) | O(N) | O(N) | O(N) |`,
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Segment Tree when:**
- Multiple **range queries** on array
- Need **both queries and updates**
- Queries are sum/min/max/gcd
- Can't use simpler approaches

**Keywords:**
- "Range sum/min/max queries"
- "Efficient updates and queries"
- "Interval queries"
- "Dynamic array with queries"

**Problem-Solving Steps:**

**1. Identify if Segment Tree is Needed:**
- Can you use prefix sums? (query-only)
- Can you use Fenwick tree? (simpler, if applicable)
- Do you really need O(log N) for both operations?

**2. Choose Operation Type:**
- Sum? Min? Max? GCD?
- Point update or range update?
- Need lazy propagation?

**3. Implementation Strategy:**
- Start with basic point update
- Add range updates later if needed
- Test with small examples

**4. Edge Cases:**
- Single element array
- Query entire array
- Update and immediate query
- Multiple updates to same position

**Interview Tips:**
- Mention simpler alternatives first
- Explain why you need segment tree
- Draw tree structure for small example
- Discuss complexity trade-offs
- Code recursively (cleaner)

**Common Mistakes:**
- Forgetting to update parent nodes
- Wrong merge function
- Index calculations (2*i+1, 2*i+2)
- Not handling no-overlap case correctly
- Forgetting lazy propagation push`,
    },
  ],
  keyTakeaways: [
    'Segment Tree enables O(log N) range queries and updates on arrays',
    'Each node represents an interval; leaves are single elements',
    'Build in O(N), query in O(log N), update in O(log N)',
    'Works for sum, min, max, GCD - any associative operation',
    'Lazy propagation enables efficient range updates in O(log N)',
    'Space: 4N array is safe, actual usage is 2N-1 nodes',
    'Use when you need both efficient queries and updates',
    'Consider simpler alternatives: prefix sums (query-only) or Fenwick tree',
  ],
  relatedProblems: [
    'range-sum-query-mutable',
    'range-minimum-query',
    'count-of-smaller-after-self',
  ],
};
