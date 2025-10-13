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
      quiz: [
        {
          id: 'segment-intro-1',
          question:
            'What is the main advantage of Segment Tree over a simple array with recalculation?',
          hint: 'Consider the time complexity of range queries.',
          sampleAnswer:
            'Segment Tree reduces range query time from O(N) to O(log N). For example, finding the sum of a range in a simple array requires iterating through all elements, but Segment Tree precomputes and stores ranges, allowing you to combine at most log N nodes to answer any query.',
          keyPoints: [
            'Range queries: O(N) → O(log N)',
            'Precomputes and stores range information',
            'Combines at most log N nodes per query',
          ],
        },
        {
          id: 'segment-intro-2',
          question: 'Why would you choose Segment Tree over Fenwick Tree?',
          hint: 'Think about operations that do not have an inverse.',
          sampleAnswer:
            'Use Segment Tree when you need operations without an inverse, like min, max, or GCD. Fenwick Tree only works for operations with inverses (like addition/subtraction). Segment Tree also supports more complex operations and lazy propagation for range updates, making it more versatile despite being harder to implement.',
          keyPoints: [
            'Supports min, max, GCD (no inverse needed)',
            'Lazy propagation for range updates',
            'More versatile but more complex code',
          ],
        },
        {
          id: 'segment-intro-3',
          question:
            'When should you use Sqrt Decomposition instead of Segment Tree?',
          hint: 'Consider implementation complexity and time limits.',
          sampleAnswer:
            'Use Sqrt Decomposition when time limits are generous and you want simpler code. Sqrt Decomposition is much easier to implement (20 lines vs 50+ for Segment Tree) and has O(√N) complexity, which is acceptable for N up to 10^6. Use Segment Tree when you need O(log N) or the problem requires sophisticated range operations.',
          keyPoints: [
            'Sqrt Decomposition: simpler, O(√N)',
            'Segment Tree: more complex, O(log N)',
            'Choose based on time limits and problem complexity',
          ],
        },
      ],
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

\`\`
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
      quiz: [
        {
          id: 'segment-structure-1',
          question:
            'Why do we typically allocate 4N space for a Segment Tree when it has 2N-1 nodes?',
          hint: 'Think about how the tree is stored in an array.',
          sampleAnswer:
            'We allocate 4N to simplify implementation and handle worst-case scenarios. When storing a tree in an array with indices 2*i and 2*i+1 for children, some array positions remain unused. The 4N bound ensures we never run out of space regardless of how the tree is built, avoiding index calculations.',
          keyPoints: [
            'Simplifies implementation',
            'Handles worst-case array storage',
            'Avoids complex index calculations',
          ],
        },
        {
          id: 'segment-structure-2',
          question:
            'Explain how parent-child relationships work in the array representation of a Segment Tree.',
          hint: 'Consider the indices used for left and right children.',
          sampleAnswer:
            'In array representation, if a node is at index i, its left child is at 2*i and right child is at 2*i+1. The parent of node i is at i//2. This binary heap-like structure allows efficient navigation without storing explicit pointers.',
          keyPoints: [
            'Left child: 2*i, Right child: 2*i+1',
            'Parent: i // 2',
            'Similar to binary heap indexing',
          ],
        },
        {
          id: 'segment-structure-3',
          question:
            'How many leaf nodes and internal nodes does a Segment Tree have for an array of size N?',
          hint: 'Leaf nodes represent individual elements.',
          sampleAnswer:
            'A Segment Tree has exactly N leaf nodes (one for each array element) and N-1 internal nodes. The total is 2N-1 nodes. Each internal node represents the union of two child segments, and you need N-1 such combinations to cover all levels.',
          keyPoints: [
            'Leaf nodes: N',
            'Internal nodes: N-1',
            'Total: 2N-1 nodes',
          ],
        },
      ],
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

\`\`python
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
      quiz: [
        {
          id: 'segment-operations-1',
          question:
            'Why is building a Segment Tree O(N) instead of O(N log N)?',
          hint: 'Think about how many times each node is visited during build.',
          sampleAnswer:
            'Building is O(N) because we visit each of the 2N-1 nodes exactly once in a bottom-up or top-down traversal. Although the tree has log N levels, we do not repeat work - each node computes its value from its two children in O(1) time. Total work is O(2N-1) = O(N).',
          keyPoints: [
            'Visit each node exactly once',
            'Total nodes: 2N-1',
            'Each node: O(1) computation',
          ],
        },
        {
          id: 'segment-operations-2',
          question: 'Explain why range query in Segment Tree is O(log N).',
          hint: 'Consider how many nodes you visit at each level.',
          sampleAnswer:
            'At each level of the tree, the query range can intersect at most 4 nodes (2 at the boundaries). Since the tree has O(log N) levels, the total nodes visited is O(4 * log N) = O(log N). The key insight is that we prune branches that are completely inside or outside the query range.',
          keyPoints: [
            'At most 4 nodes per level',
            'Tree height: O(log N)',
            'Pruning reduces work dramatically',
          ],
        },
        {
          id: 'segment-operations-3',
          question:
            'What is the update process in a Segment Tree after modifying a leaf node?',
          hint: 'Think about which nodes are affected by a single element change.',
          sampleAnswer:
            'After updating a leaf, you must update all ancestors up to the root. Each ancestor recomputes its value from its two children. Since the tree height is log N, you update O(log N) nodes. The path from leaf to root visits one node per level.',
          keyPoints: [
            'Update all ancestors up to root',
            'Path from leaf to root: O(log N)',
            'Each ancestor recomputes from children',
          ],
        },
      ],
    },
    {
      id: 'variations',
      title: 'Segment Tree Variations',
      content: `**Range Minimum Query (RMQ):**
\`\`python
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
      quiz: [
        {
          id: 'segment-variations-1',
          question:
            'Explain how lazy propagation works and why it is necessary for range updates.',
          hint: 'Think about what happens if you update every element in a range individually.',
          sampleAnswer:
            'Without lazy propagation, updating a range of N elements would take O(N log N) time. Lazy propagation defers updates by marking nodes with pending changes instead of immediately propagating to children. Updates are pushed down only when needed during queries. This makes range updates O(log N).',
          keyPoints: [
            'Defers updates to avoid O(N log N)',
            'Marks nodes with pending changes',
            'Push down only when queried',
            'Range update becomes O(log N)',
          ],
        },
        {
          id: 'segment-variations-2',
          question:
            'What is the difference between a Range Sum Tree and a Range Min Tree?',
          hint: 'Consider the merge operation and the identity element.',
          sampleAnswer:
            'The difference is only in the merge operation. Range Sum Tree uses addition to combine children (left + right), with identity 0. Range Min Tree uses minimum (min(left, right)), with identity infinity. The tree structure and algorithms remain identical - only the combine function changes.',
          keyPoints: [
            'Sum Tree: merge = addition, identity = 0',
            'Min Tree: merge = minimum, identity = ∞',
            'Same structure, different merge function',
          ],
        },
        {
          id: 'segment-variations-3',
          question:
            'How do you handle a Segment Tree that needs to support both range sum and range min queries?',
          hint: 'Think about what information each node must store.',
          sampleAnswer:
            'Store both sum and min at each node. During build and update, compute both values. For queries, return both pieces of information. This doubles the space per node but does not change the time complexity. Each node becomes a struct/object with multiple fields.',
          keyPoints: [
            'Store multiple values per node',
            'Compute all values during build/update',
            'Space: doubles, Time complexity: unchanged',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'segment-complexity-1',
          question:
            'Why does Segment Tree use O(4N) space instead of O(2N-1) for 2N-1 nodes?',
          hint: 'Consider how we store the tree in an array.',
          sampleAnswer:
            'We use O(4N) for simplicity. When using array indexing (left child at 2*i, right child at 2*i+1), some positions remain unused. The tight bound is closer to 2N, but 4N is a safe upper bound that avoids careful analysis and makes implementation simpler. The space is still O(N).',
          keyPoints: [
            'Tight bound: ~2N, Safe bound: 4N',
            'Array storage leaves gaps',
            'Simplifies implementation',
          ],
        },
        {
          id: 'segment-complexity-2',
          question:
            'Compare the build time of Segment Tree (O(N)) versus Fenwick Tree (O(N log N)).',
          hint: 'Think about what each build process does.',
          sampleAnswer:
            'Segment Tree builds in O(N) by visiting each of its 2N-1 nodes once. Fenwick Tree builds in O(N log N) by calling update() N times, where each update is O(log N). In practice, both are fast enough, but Segment Tree has a faster build. Fenwick Tree compensates with less space.',
          keyPoints: [
            'Segment Tree: O(N) build, O(4N) space',
            'Fenwick Tree: O(N log N) build, O(N) space',
            'Tradeoff: build time vs space',
          ],
        },
        {
          id: 'segment-complexity-3',
          question: 'What is the space complexity of a 2D Segment Tree?',
          hint: 'Consider nesting two segment trees.',
          sampleAnswer:
            'A 2D Segment Tree for an M×N matrix uses O(M*N) space if implemented efficiently, or up to O(16*M*N) with simple array allocation. The structure is a tree of trees - each node of the outer tree contains an inner tree. Time complexity for queries/updates is O(log M * log N).',
          keyPoints: [
            'Efficient: O(M*N), Simple: O(16*M*N)',
            'Tree of trees structure',
            'Query/Update: O(log M * log N)',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'segment-interview-1',
          question: 'How do you recognize when a problem needs a Segment Tree?',
          hint: 'Look for patterns in the problem requirements.',
          sampleAnswer:
            'Key signals: "range queries" + "updates", operations like min/max/GCD that have no inverse, multiple queries on dynamic data, or explicit mention of intervals/segments. If the problem combines querying ranges and modifying data, and simpler approaches like prefix sums do not work, Segment Tree is likely needed.',
          keyPoints: [
            '"Range queries" + "updates"',
            'Min/max/GCD operations',
            'Dynamic data with multiple queries',
            'Intervals or segments mentioned',
          ],
        },
        {
          id: 'segment-interview-2',
          question:
            'What are the most common mistakes when implementing Segment Trees in interviews?',
          hint: 'Think about edge cases and implementation details.',
          sampleAnswer:
            'Common mistakes: 1) Off-by-one errors in range boundaries, 2) Forgetting to handle single-element ranges, 3) Wrong base case in recursion, 4) Not allocating enough space (use 4*N), 5) Incorrect merge logic for the specific operation. Always test with small examples and edge cases like N=1 or querying a single element.',
          keyPoints: [
            'Off-by-one errors in ranges',
            'Base case and single elements',
            'Space allocation (4*N)',
            'Correct merge operation',
          ],
        },
        {
          id: 'segment-interview-3',
          question:
            'Should you implement Segment Tree iteratively or recursively in an interview?',
          hint: 'Consider code clarity versus implementation speed.',
          sampleAnswer:
            'Use recursive implementation in interviews unless you are very comfortable with iterative. Recursive is more intuitive, easier to explain, and less error-prone. Iterative can be faster and saves stack space, but is harder to get right under pressure. Most interviewers prefer correct recursive code over buggy iterative code.',
          keyPoints: [
            'Recursive: more intuitive, easier to explain',
            'Iterative: faster, saves stack, harder to implement',
            'Choose based on confidence level',
          ],
        },
      ],
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
