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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What problem does Segment Tree solve?',
          options: [
            'Sorting',
            'Efficient range queries (sum/min/max) and updates - O(log N) for both',
            'Searching',
            'Hashing',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree enables O(log N) range queries and point/range updates. Useful when you need both operations efficiently, unlike prefix sum (fast query, slow update) or simple array (slow query, fast update).',
        },
        {
          id: 'mc2',
          question: 'When should you use Segment Tree over Fenwick Tree?',
          options: [
            'Always',
            'Need operations without inverse (min, max, GCD) or lazy propagation for range updates',
            'Never',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Use Segment Tree for: 1) Non-invertible operations (min, max, GCD), 2) Lazy propagation for efficient range updates, 3) More complex custom operations. Fenwick is simpler but limited to invertible operations.',
        },
        {
          id: 'mc3',
          question: 'What is the time complexity of Segment Tree operations?',
          options: [
            'O(N)',
            'Query: O(log N), Update: O(log N), Build: O(N)',
            'All O(1)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree: Build O(N), Query O(log N) (traverse tree height), Point Update O(log N) (update path to root). Space: O(N) for 4N array representation.',
        },
        {
          id: 'mc4',
          question: 'Why use Segment Tree over Prefix Sum for range queries?',
          options: [
            'Faster queries',
            'Supports updates - prefix sum O(N) rebuild after update, segment tree O(log N) update',
            'Less space',
            'Simpler',
          ],
          correctAnswer: 1,
          explanation:
            'Prefix sum: O(1) query but O(N) update (rebuild array). Segment Tree: O(log N) query and O(log N) update. Use segment tree when you need both queries and updates on dynamic data.',
        },
        {
          id: 'mc5',
          question: 'What types of operations can Segment Tree efficiently handle?',
          options: [
            'Only sum',
            'Any associative operation: sum, min, max, GCD, OR, AND, XOR',
            'Only min/max',
            'None',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree works for any associative operation where combine(a, combine(b, c)) = combine(combine(a, b), c). Examples: sum, min, max, GCD, bitwise OR/AND/XOR.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How is a Segment Tree typically stored?',
          options: [
            'Linked nodes',
            'Array of size 4N with tree[1] as root, children at 2i and 2i+1',
            'Hash map',
            'Stack',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree uses array representation: tree[1] is root, for node i, left child is 2i, right child is 2i+1. Size 4N ensures space for complete binary tree.',
        },
        {
          id: 'mc2',
          question: 'What does each node in a Segment Tree represent?',
          options: [
            'Single element',
            'An interval [L, R] with aggregate value (sum/min/max) for that range',
            'Random value',
            'Index',
          ],
          correctAnswer: 1,
          explanation:
            'Each node stores: 1) Interval [L, R] it represents, 2) Aggregate value (sum/min/max) for that range. Leaf nodes are single elements [i, i]. Internal nodes combine children intervals.',
        },
        {
          id: 'mc3',
          question: 'What is the height of a Segment Tree for array of size N?',
          options: [
            'O(N)',
            'O(log N) - complete binary tree',
            'O(√N)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree is complete binary tree, so height is O(log N). Each level doubles nodes until reaching N leaves. This log height enables O(log N) operations.',
        },
        {
          id: 'mc4',
          question: 'How do you find children of node i in array representation?',
          options: [
            'i+1, i+2',
            'Left child: 2i, Right child: 2i+1',
            'Random',
            'i-1, i-2',
          ],
          correctAnswer: 1,
          explanation:
            'Array heap property: for node at index i, left child at 2i, right child at 2i+1, parent at i//2. This allows O(1) navigation without pointers.',
        },
        {
          id: 'mc5',
          question: 'Why do we allocate 4N space for Segment Tree?',
          options: [
            'Random choice',
            'Ensures space for complete binary tree - worst case when N not power of 2',
            'Always need exactly 4N',
            'Optimization',
          ],
          correctAnswer: 1,
          explanation:
            'Complete binary tree with N leaves has at most 2N-1 nodes. For safety with any N (not just powers of 2) and simple indexing, 4N guarantees sufficient space.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What are the three cases when processing a range query?',
          options: [
            'Random',
            'No overlap (return identity), Complete overlap (return node value), Partial overlap (recurse both children)',
            'Always recurse',
            'Just return value',
          ],
          correctAnswer: 1,
          explanation:
            'Range query cases: 1) No overlap (query range doesn\'t intersect node) → return identity (0 for sum, INF for min), 2) Complete overlap (node fully in query) → return node value, 3) Partial overlap → recurse on both children and combine.',
        },
        {
          id: 'mc2',
          question: 'How do you update a single element in Segment Tree?',
          options: [
            'Update all nodes',
            'Traverse from root to leaf updating path nodes O(log N)',
            'Rebuild tree',
            'Update leaf only',
          ],
          correctAnswer: 1,
          explanation:
            'Point update: traverse path from root to target leaf (O(log N) height). At each node, check if it contains target index. Update leaf, then propagate changes up by recalculating parent values.',
        },
        {
          id: 'mc3',
          question: 'What is lazy propagation?',
          options: [
            'Slow algorithm',
            'Defer range updates using lazy array - only push changes when needed for O(log N) range update',
            'Random optimization',
            'Bad practice',
          ],
          correctAnswer: 1,
          explanation:
            'Lazy propagation: for range updates, store pending updates in lazy array instead of immediately updating all affected nodes. Push updates down only when accessing nodes. Reduces range update from O(N) to O(log N).',
        },
        {
          id: 'mc4',
          question: 'How do you build a Segment Tree?',
          options: [
            'Random',
            'Recursive: leaves get array values, parents combine children - O(N) time',
            'Iterative only',
            'Cannot build',
          ],
          correctAnswer: 1,
          explanation:
            'Build recursively: 1) Base case: leaf node [i,i] gets arr[i], 2) Recursive: compute left and right subtrees, combine values. Visits each of 2N-1 nodes once = O(N) time.',
        },
        {
          id: 'mc5',
          question: 'What does "combine" function do in Segment Tree?',
          options: [
            'Sorts nodes',
            'Merges two child values: sum (a+b), min (min(a,b)), max (max(a,b)), etc.',
            'Random',
            'Deletes nodes',
          ],
          correctAnswer: 1,
          explanation:
            'Combine function merges children based on operation: sum→a+b, min→min(a,b), max→max(a,b), GCD→gcd(a,b). Must be associative. Determines what aggregate the tree maintains.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What variation do you use for range minimum query?',
          options: [
            'Same as sum',
            'Change combine to min(left, right) instead of left + right',
            'Different tree structure',
            'Cannot do RMQ',
          ],
          correctAnswer: 1,
          explanation:
            'RMQ: just change combine function from sum (left+right) to min (min(left,right)). Identity changes from 0 to INF. Same structure, different operation.',
        },
        {
          id: 'mc2',
          question: 'How does lazy propagation work for range updates?',
          options: [
            'Updates immediately',
            'Store pending updates in lazy array, push down only when needed',
            'Rebuilds tree',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Lazy propagation: maintain lazy[node] array for pending updates. When updating range, mark affected nodes as lazy. Push updates down only when querying/updating those nodes. Reduces range update to O(log N).',
        },
        {
          id: 'mc3',
          question: 'What operations can Segment Tree handle?',
          options: [
            'Only sum',
            'Any associative operation: sum, min, max, GCD, XOR, etc.',
            'Only min/max',
            'None',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree works with any associative operation where order of combining doesn\'t matter: sum, product, min, max, GCD, LCM, XOR, OR, AND. Just change the combine function.',
        },
        {
          id: 'mc4',
          question: '2D Segment Tree is used for what?',
          options: [
            'Sorting',
            'Range queries on 2D matrix (rectangle sum, min, etc.)',
            'Graph traversal',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            '2D Segment Tree handles 2D range queries like rectangle sum in O(log²N) time. Build tree of trees: outer for rows, inner for columns. Updates and queries work in both dimensions.',
        },
        {
          id: 'mc5',
          question: 'When would you use Persistent Segment Tree?',
          options: [
            'Random',
            'Need to query previous versions of array after updates (version control)',
            'Always',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Persistent Segment Tree maintains all versions after updates by creating new nodes instead of modifying. Query any historical version. Used in time-travel queries, undo/redo, or range queries at specific timestamps.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of Segment Tree operations?',
          options: [
            'All O(N)',
            'Build: O(N), Query: O(log N), Update: O(log N)',
            'All O(log N)',
            'Build: O(log N), Query: O(N)',
          ],
          correctAnswer: 1,
          explanation:
            'Build: O(N) visits each node once. Query: O(log N) traverses tree height. Point Update: O(log N) updates path to root. Range Update with lazy prop: O(log N).',
        },
        {
          id: 'mc2',
          question: 'What is the space complexity of Segment Tree?',
          options: [
            'O(log N)',
            'O(N) - specifically 4N for array representation',
            'O(N²)',
            'O(1)',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree needs O(N) space. Array representation uses 4N to handle all cases safely (complete binary tree with N leaves needs at most 2N-1 nodes, 4N is safe upper bound).',
        },
        {
          id: 'mc3',
          question: 'Why is range query O(log N)?',
          options: [
            'Random',
            'At most O(log N) nodes visited per level, and tree height is O(log N)',
            'Always fast',
            'Uses binary search',
          ],
          correctAnswer: 1,
          explanation:
            'Range query visits at most 2 nodes per level (one for each subtree boundary). Tree height is O(log N), so total nodes visited is O(log N).',
        },
        {
          id: 'mc4',
          question: 'How does lazy propagation reduce range update complexity?',
          options: [
            'Makes it slower',
            'Defers updates - mark O(log N) nodes instead of updating O(N) affected leaves',
            'Random',
            'No difference',
          ],
          correctAnswer: 1,
          explanation:
            'Without lazy prop: range update touches O(N) leaves. With lazy prop: mark O(log N) ancestor nodes, push updates only when needed. Reduces from O(N) to O(log N).',
        },
        {
          id: 'mc5',
          question: 'What makes Segment Tree efficient?',
          options: [
            'Random',
            'Precomputed intervals at all levels enable O(log N) range queries by combining logarithmic nodes',
            'Always O(1)',
            'Uses sorting',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree precomputes all interval combinations. Any range can be decomposed into O(log N) precomputed intervals. Combining these is fast (single operation per interval).',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What signals suggest using Segment Tree in an interview?',
          options: [
            'Sorting only',
            'Multiple range queries + updates on array, keywords: range sum/min/max, dynamic array',
            'Binary search',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Segment Tree signals: 1) Multiple range queries needed, 2) Array is dynamic (updates), 3) Keywords like "range sum", "range minimum", 4) Need O(log N) for both query and update.',
        },
        {
          id: 'mc2',
          question: 'What should you clarify in a Segment Tree interview?',
          options: [
            'Nothing',
            'Operation type (sum/min/max), update type (point/range), constraints (N size, query count)',
            'Random',
            'Language only',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: 1) What operation (determines combine function), 2) Point or range updates (range needs lazy prop), 3) Constraints on N and Q (affects if segment tree needed), 4) Memory limits (4N space).',
        },
        {
          id: 'mc3',
          question: 'What is a common mistake when implementing Segment Tree?',
          options: [
            'Using recursion',
            'Off-by-one errors in range bounds, forgetting to push lazy updates',
            'Good naming',
            'Comments',
          ],
          correctAnswer: 1,
          explanation:
            'Common mistakes: 1) Off-by-one in [L, R] vs [L, R) ranges, 2) Forgetting to push lazy values before querying, 3) Wrong node indexing (2i vs 2i+1), 4) Not handling identity values for operations.',
        },
        {
          id: 'mc4',
          question: 'When would you use Fenwick Tree instead of Segment Tree?',
          options: [
            'Always',
            'Simpler code, operation has inverse (sum), don\'t need lazy propagation',
            'Never',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Use Fenwick when: 1) Operation is invertible (sum, XOR), 2) Only point updates (no lazy prop needed), 3) Want simpler code (half the lines). Use Segment Tree for min/max/GCD or range updates.',
        },
        {
          id: 'mc5',
          question: 'What is good interview communication for Segment Tree?',
          options: [
            'Just code',
            'Explain why Segment Tree (O(log N) query+update), clarify operation, walk through build/query, discuss complexity',
            'No explanation',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Communication: 1) Justify why Segment Tree over alternatives, 2) Explain tree structure briefly, 3) Walk through one query example, 4) Mention time O(log N) and space O(N), 5) Code with clear comments.',
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
