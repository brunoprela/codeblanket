import { Module } from '@/lib/types';

export const fenwickTreeModule: Module = {
  id: 'fenwick-tree',
  title: 'Fenwick Tree (Binary Indexed Tree)',
  description:
    'Master Fenwick Trees for efficient prefix sum queries and updates.',
  icon: '🎯',
  timeComplexity: 'O(log N) for queries and updates',
  spaceComplexity: 'O(N)',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Fenwick Tree',
      content: `**Fenwick Tree** (also called **Binary Indexed Tree** or **BIT**) is a data structure that efficiently calculates prefix sums and handles updates.

**What it Does:**
- **Prefix Sum Query**: Sum of elements from index 0 to i
- **Range Sum Query**: Sum from index L to R (= prefix[R] - prefix[L-1])
- **Point Update**: Add value to element at index i

**Why Fenwick Tree:**
- Simpler than Segment Tree
- Same complexity: O(log N) for both operations
- Less memory: Just an array of size N+1
- Elegant implementation using bit manipulation

**When to Use:**
- Need prefix sums with updates
- Range sum queries
- Cumulative frequency (counting inversions)
- 2D range sums

**Compared to Alternatives:**

| Data Structure | Build | Query | Update | Space |
|----------------|-------|-------|--------|-------|
| Prefix Sum | O(N) | O(1) | O(N) | O(N) |
| Fenwick Tree | O(N log N) | O(log N) | O(log N) | O(N) |
| Segment Tree | O(N) | O(log N) | O(log N) | O(4N) |

**Key Advantage:**
- Simpler to code than Segment Tree
- Works for any operation with an **inverse** (addition/subtraction, XOR)
- Can't do min/max (no inverse operation)`,
    },
    {
      id: 'structure',
      title: 'Fenwick Tree Structure',
      content: `**How it Works:**
Each index in BIT stores sum of a range of elements, not just one element.

**Index Responsibility:**
Index \`i\` stores sum of \`r(i)\` elements, where \`r(i)\` is the position of the last set bit.

**Example:** Array = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]

\`\`\`
Index (binary)  Stores sum of
1  (0001)       arr[1]           (1 element)
2  (0010)       arr[1..2]        (2 elements)
3  (0011)       arr[3]           (1 element)
4  (0100)       arr[1..4]        (4 elements)
5  (0101)       arr[5]           (1 element)
6  (0110)       arr[5..6]        (2 elements)
7  (0111)       arr[7]           (1 element)
8  (1000)       arr[1..8]        (8 elements)
\`\`\`

**Key Operations:**

**1. Get Last Set Bit (LSB):**
\`\`\`python
def lsb(i):
    return i & (-i)
# Examples:
# 6 (110) & -6 (010) = 2 (010)
# 12 (1100) & -12 (0100) = 4 (0100)
\`\`\`

**2. Parent (for update):**
\`\`\`python
def parent(i):
    return i + (i & -i)
# Move to next index that needs updating
\`\`\`

**3. Prefix (for query):**
\`\`\`python
def prefix_parent(i):
    return i - (i & -i)
# Move to previous range to sum
\`\`\``,
    },
    {
      id: 'operations',
      title: 'Core Operations',
      content: `**1. Point Update (Add delta to index i)**

\`\`\`python
def update(tree, i, delta):
    """Add delta to index i (1-indexed)"""
    while i < len(tree):
        tree[i] += delta
        i += i & -i  # Move to parent
\`\`\`

**2. Prefix Sum (Sum from 1 to i)**

\`\`\`python
def prefix_sum(tree, i):
    """Get sum of elements from 1 to i"""
    total = 0
    while i > 0:
        total += tree[i]
        i -= i & -i  # Move to previous range
    return total
\`\`\`

**3. Range Sum (Sum from L to R)**

\`\`\`python
def range_sum(tree, L, R):
    """Get sum from L to R (1-indexed)"""
    return prefix_sum(tree, R) - prefix_sum(tree, L - 1)
\`\`\`

**4. Build Tree**

\`\`\`python
def build(arr):
    """Build Fenwick tree from array (0-indexed input)"""
    n = len(arr)
    tree = [0] * (n + 1)  # 1-indexed
    
    for i, val in enumerate(arr):
        update(tree, i + 1, val)
    
    return tree
\`\`\`

**Complexity:**
- Update: O(log N)
- Query: O(log N)
- Build: O(N log N)
- Space: O(N)`,
      codeExample: `class FenwickTree:
    """Binary Indexed Tree for prefix sums"""
    
    def __init__(self, n: int):
        """Initialize with size n (supports indices 1 to n)"""
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed
    
    def update(self, i: int, delta: int):
        """
        Add delta to element at index i (1-indexed).
        Time: O(log N)
        """
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i  # Move to parent
    
    def prefix_sum(self, i: int) -> int:
        """
        Get sum of elements from 1 to i.
        Time: O(log N)
        """
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i  # Move to previous range
        return total
    
    def range_sum(self, L: int, R: int) -> int:
        """
        Get sum from L to R (1-indexed).
        Time: O(log N)
        """
        return self.prefix_sum(R) - self.prefix_sum(L - 1)
    
    @staticmethod
    def from_array(arr: list) -> 'FenwickTree':
        """Build from 0-indexed array"""
        ft = FenwickTree(len(arr))
        for i, val in enumerate(arr):
            ft.update(i + 1, val)  # Convert to 1-indexed
        return ft


# Example usage
arr = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]
ft = FenwickTree.from_array(arr)

print(ft.prefix_sum(5))      # Sum of first 5 elements
print(ft.range_sum(3, 7))    # Sum from index 3 to 7
ft.update(4, 3)               # Add 3 to element at index 4
print(ft.range_sum(3, 7))    # Updated sum`,
    },
    {
      id: 'advanced',
      title: 'Advanced Techniques',
      content: `**2D Fenwick Tree (Range Queries on 2D Matrix)**

\`\`\`python
class FenwickTree2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, r, c, delta):
        """Add delta to cell (r, c)"""
        i = r
        while i <= self.rows:
            j = c
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i
    
    def query(self, r, c):
        """Sum of rectangle from (1,1) to (r,c)"""
        total = 0
        i = r
        while i > 0:
            j = c
            while j > 0:
                total += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return total
    
    def range_query(self, r1, c1, r2, c2):
        """Sum of rectangle from (r1,c1) to (r2,c2)"""
        return (self.query(r2, c2) - 
                self.query(r1-1, c2) - 
                self.query(r2, c1-1) + 
                self.query(r1-1, c1-1))
\`\`\`

**Finding kth Element (Binary Search on Fenwick Tree)**

\`\`\`python
def find_kth(tree, k):
    """
    Find smallest index with prefix sum >= k.
    Requires tree to be cumulative (monotonic).
    """
    i = 0
    bit = 1 << 20  # Start with large power of 2
    
    while bit > 0:
        if i + bit < len(tree) and tree[i + bit] < k:
            i += bit
            k -= tree[i]
        bit >>= 1
    
    return i + 1  # 1-indexed result
\`\`\`

**Counting Inversions**

\`\`\`python
def count_inversions(arr):
    """
    Count pairs (i,j) where i < j but arr[i] > arr[j].
    Uses coordinate compression + Fenwick tree.
    """
    # Coordinate compression
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1])
    rank = [0] * len(arr)
    for i, (orig_idx, _) in enumerate(sorted_arr):
        rank[orig_idx] = i + 1
    
    # Count inversions
    ft = FenwickTree(len(arr))
    inversions = 0
    
    for i in range(len(arr) - 1, -1, -1):
        # Count how many smaller elements we've seen
        inversions += ft.prefix_sum(rank[i] - 1)
        ft.update(rank[i], 1)
    
    return inversions
\`\`\``,
    },
    {
      id: 'comparison',
      title: 'Fenwick Tree vs Segment Tree',
      content: `**When to Use Fenwick Tree:**
- ✅ Prefix sums, range sums
- ✅ Operations with **inverse** (add/subtract, XOR)
- ✅ Simpler to code
- ✅ Less memory

**When to Use Segment Tree:**
- ✅ Min/max queries (no inverse)
- ✅ GCD queries
- ✅ More complex operations
- ✅ Range updates with lazy propagation
- ✅ Easier to understand/debug

**Fenwick Limitations:**
- ❌ Can't do min/max (no inverse operation)
- ❌ Range updates more complex
- ❌ Less intuitive than Segment Tree
- ❌ 1-indexed (not 0-indexed)

**Code Comparison:**

**Fenwick Tree:** ~20 lines
\`\`\`python
class FenwickTree:
    def __init__(self, n):
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i
    
    def query(self, i):
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total
\`\`\`

**Segment Tree:** ~50 lines (more complex but more powerful)

**Decision Guide:**
- Need sum/prefix queries? → Fenwick Tree
- Need min/max/gcd? → Segment Tree
- Simple problem? → Fenwick Tree
- Complex operations? → Segment Tree`,
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Fenwick Tree when:**
- "Prefix sum with updates"
- "Range sum queries"
- "Cumulative frequencies"
- "Count inversions"
- "Dynamic ranking"

**Keywords:**
- "Prefix sum", "cumulative sum"
- "Range sum with updates"
- "Efficient queries and updates"
- "Counting smaller/larger elements"

**Problem-Solving Steps:**

**1. Identify if Fenwick applies:**
- Is it prefix/range sum?
- Do I need updates?
- Is there an inverse operation?

**2. Handle 1-indexing:**
- Fenwick uses 1-indexed arrays
- Convert input: \`update(i+1, val)\`
- Or use 0-indexed version

**3. Choose variant:**
- Basic: Point update, range query
- 2D: For matrix problems
- Binary search: For kth element

**4. Implementation tips:**
- Remember: \`i & -i\` gets last set bit
- Update: add last set bit (\`i += i & -i\`)
- Query: subtract last set bit (\`i -= i & -i\`)

**Common Mistakes:**
- Forgetting 1-indexing
- Wrong bit operation (\`i & i\` vs \`i & -i\`)
- Not initializing tree size correctly
- Using for min/max (need Segment Tree)

**Interview Tips:**
- Mention it's simpler than Segment Tree
- Explain bit manipulation trick
- Draw small example (n=8)
- Discuss limitations (no min/max)`,
    },
  ],
  keyTakeaways: [
    'Fenwick Tree (BIT) provides O(log N) prefix sums and updates',
    'Uses bit manipulation: i & -i gets last set bit',
    'Update: i += i & -i (move to parent), Query: i -= i & -i (move back)',
    'Simpler than Segment Tree but only works for operations with inverse',
    '1-indexed: tree[0] unused, indices 1 to N',
    'Perfect for: range sums, counting inversions, cumulative frequency',
    'Cannot do: min/max queries (no inverse operation)',
    'Space: O(N), much simpler code than Segment Tree (~20 lines)',
  ],
  relatedProblems: [
    'range-sum-query-fenwick',
    'count-inversions',
    'range-sum-2d-mutable',
  ],
};
