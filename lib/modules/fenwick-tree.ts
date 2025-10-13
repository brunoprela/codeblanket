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
      quiz: [
        {
          id: 'fenwick-intro-1',
          question:
            'Why would you choose a Fenwick Tree over a simple prefix sum array when you need both queries and updates?',
          hint: 'Think about what happens when you update a value in a prefix sum array.',
          sampleAnswer:
            'With a simple prefix sum array, queries are O(1) but updates are O(N) because you must recalculate all prefix sums after the update. Fenwick Tree balances this tradeoff by making both operations O(log N), which is much better when you have many updates.',
          keyPoints: [
            'Prefix sum array: O(1) query, O(N) update',
            'Fenwick Tree: O(log N) query, O(log N) update',
            'Better when you have frequent updates',
          ],
        },
        {
          id: 'fenwick-intro-2',
          question:
            'What is the key limitation of Fenwick Tree compared to Segment Tree, and why does this limitation exist?',
          hint: 'Consider what mathematical property is needed for a Fenwick Tree to work.',
          sampleAnswer:
            'Fenwick Tree cannot handle operations without an inverse, like min/max or GCD. This is because Fenwick Tree relies on being able to add and subtract ranges. For addition, subtraction is the inverse. But there is no inverse for min or max - you cannot "un-min" a value.',
          keyPoints: [
            'Fenwick Tree needs an inverse operation',
            'Works for: addition (inverse: subtraction), XOR (inverse: XOR)',
            'Does not work for: min, max, GCD (no inverse)',
          ],
        },
        {
          id: 'fenwick-intro-3',
          question:
            'When should you prefer Fenwick Tree over Segment Tree for a range sum problem?',
          hint: 'Think about code complexity and implementation time.',
          sampleAnswer:
            'Use Fenwick Tree when the problem only needs range sums or prefix sums with updates. Fenwick Tree is simpler to implement (around 20 lines vs 50+ for Segment Tree), uses less memory, and is less error-prone in interviews. Only use Segment Tree when you need operations like min/max or lazy propagation.',
          keyPoints: [
            'Fenwick Tree: simpler, fewer lines of code',
            'Same time complexity for range sums',
            'Less memory usage',
            'Easier to implement correctly under time pressure',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'fenwick-structure-1',
          question:
            'Explain what the operation "i & -i" does and why it is critical to Fenwick Tree.',
          hint: "Think about what -i looks like in binary (two's complement).",
          sampleAnswer:
            'The operation "i & -i" extracts the least significant bit (LSB) of i. In two\'s complement, -i flips all bits and adds 1, so when you AND them together, only the rightmost 1-bit survives. This tells us how many elements the current index is responsible for, and where to jump next during update or query.',
          keyPoints: [
            'Extracts the least significant set bit',
            'Determines range size at each index',
            'Used to navigate parent/child relationships',
          ],
        },
        {
          id: 'fenwick-structure-2',
          question:
            'Why does Fenwick Tree use 1-indexing instead of 0-indexing?',
          hint: 'Consider what happens when you try "0 & -0".',
          sampleAnswer:
            'Fenwick Tree uses 1-indexing because the bit manipulation "i & -i" does not work correctly for i=0. When i=0, "i & -i" equals 0, creating an infinite loop since adding or subtracting 0 does not change the index. Starting from index 1 avoids this issue.',
          keyPoints: [
            '"0 & -0" = 0, causes infinite loops',
            'tree[0] is unused, indices start at 1',
            'Need to convert: arr[i] → tree[i+1]',
          ],
        },
        {
          id: 'fenwick-structure-3',
          question:
            'How do you traverse up the tree during an update operation?',
          hint: 'Think about which indices depend on the current index.',
          sampleAnswer:
            'During update, you move up by adding the LSB: "i += i & -i". This takes you to the parent node that needs updating. You keep going until i exceeds the tree size. For example, updating index 5 (binary 101) affects indices 5→6→8→16...',
          keyPoints: [
            'Move up: i += i & -i',
            'Each parent covers a larger range',
            'Continue until i > n',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'fenwick-operations-1',
          question:
            'Why is building a Fenwick Tree O(N log N) instead of O(N) like a Segment Tree?',
          hint: 'Think about what happens when you call update() for each element.',
          sampleAnswer:
            'Building by calling update() N times gives O(N log N) because each update is O(log N). There is actually an O(N) build method that fills the tree directly without updates, but it is less commonly taught. The O(N log N) method is simpler and more intuitive, though slightly slower.',
          keyPoints: [
            'Naive build: N elements × O(log N) per update = O(N log N)',
            'Optimized build exists but rarely needed',
            'Still acceptable for most problems',
          ],
        },
        {
          id: 'fenwick-operations-2',
          question:
            'Explain how to compute range_sum(L, R) using only prefix_sum operations.',
          hint: 'This uses the principle of inclusion-exclusion.',
          sampleAnswer:
            'Range sum from L to R is computed as prefix_sum(R) - prefix_sum(L-1). This works because prefix_sum(R) includes everything from 1 to R, and prefix_sum(L-1) includes everything from 1 to L-1. Subtracting removes the unwanted prefix, leaving only elements from L to R.',
          keyPoints: [
            'range_sum(L, R) = prefix_sum(R) - prefix_sum(L-1)',
            'Inclusion-exclusion principle',
            'Both queries are O(log N), so total is O(log N)',
          ],
        },
        {
          id: 'fenwick-operations-3',
          question:
            'What is the update operation actually doing? Does it set a value or add a delta?',
          hint: 'Think about why it is called "add delta" not "set value".',
          sampleAnswer:
            'Fenwick Tree update adds a delta to an index, it does not set an absolute value. To set arr[i] to new_val, you must compute delta = new_val - arr[i], then call update(i, delta). This is because Fenwick Tree stores cumulative sums, not individual values, so it can only support additive updates efficiently.',
          keyPoints: [
            'Updates add a delta, not set a value',
            'To set: delta = new_value - old_value, then update(i, delta)',
            'Fenwick stores cumulative data, not raw values',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'fenwick-advanced-1',
          question:
            'How do you extend Fenwick Tree to handle 2D range queries?',
          hint: 'Think about nesting the update and query operations.',
          sampleAnswer:
            'A 2D Fenwick Tree uses nested loops. For update(r, c, delta), the outer loop iterates over rows using "r += r & -r", and for each row, the inner loop iterates over columns using "c += c & -c". Query works similarly with subtraction. This gives O(log M × log N) complexity for an M×N matrix.',
          keyPoints: [
            'Nest two Fenwick Trees: one for rows, one for columns',
            'Update/Query: O(log M × log N)',
            'Useful for 2D prefix sum problems',
          ],
        },
        {
          id: 'fenwick-advanced-2',
          question:
            'Explain how Fenwick Tree can be used to count inversions in an array.',
          hint: 'Think about processing elements in reverse and tracking what you have seen.',
          sampleAnswer:
            "To count inversions, iterate the array from right to left. Use coordinate compression to map values to ranks. For each element, query the Fenwick Tree for how many smaller elements you have already seen (prefix_sum of rank-1). Then update the tree by adding 1 at this element's rank. The sum of all queries is the inversion count.",
          keyPoints: [
            'Process array right to left',
            'Use coordinate compression for large values',
            'Query counts smaller elements seen so far',
            'Total inversions = sum of all queries',
          ],
        },
        {
          id: 'fenwick-advanced-3',
          question:
            'What is the binary search on Fenwick Tree technique and when is it useful?',
          hint: 'Think about finding the kth element in a frequency array.',
          sampleAnswer:
            'Binary search on Fenwick Tree finds the smallest index with prefix_sum >= k. You start with a large power of 2 and halve it each step, adding it to your position if it does not exceed k. This is useful for finding the kth smallest element in O(log N) when the tree represents cumulative frequencies.',
          keyPoints: [
            'Finds kth element in O(log N)',
            'Requires monotonic (cumulative) data',
            'Uses powers of 2 to navigate',
            'Useful for frequency/rank queries',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'fenwick-comparison-1',
          question:
            'Explain why Fenwick Tree cannot handle range minimum queries (RMQ) while Segment Tree can.',
          hint: 'Think about the inverse operation requirement.',
          sampleAnswer:
            'Fenwick Tree requires an inverse operation to compute range queries using prefix subtraction. For sums, range(L,R) = prefix(R) - prefix(L-1) works because subtraction is the inverse of addition. But for min, there is no inverse - you cannot "un-min" values. Segment Tree stores ranges directly without needing inverses, so it can handle min/max.',
          keyPoints: [
            'Fenwick needs inverse operation',
            'Min/max have no inverse',
            'Segment Tree stores ranges directly',
          ],
        },
        {
          id: 'fenwick-comparison-2',
          question:
            'When would you choose the simpler Fenwick Tree over Segment Tree in an interview?',
          hint: 'Consider time pressure and implementation complexity.',
          sampleAnswer:
            'Choose Fenwick Tree when the problem only needs sum queries with updates and you are under time pressure. Fenwick Tree is 20 lines vs 50+ for Segment Tree, easier to implement correctly, and has the same time complexity. Only use Segment Tree when you need operations Fenwick cannot do, like min/max or lazy propagation.',
          keyPoints: [
            'Fenwick: faster to code correctly',
            'Same O(log N) complexity for sums',
            'Less prone to implementation bugs',
            'Only sacrifice: cannot do min/max',
          ],
        },
        {
          id: 'fenwick-comparison-3',
          question:
            'What advantage does Segment Tree have in terms of memory and build time?',
          hint: 'Compare the build complexity and space usage.',
          sampleAnswer:
            'Segment Tree actually builds in O(N) time, while Fenwick Tree takes O(N log N) with the standard build method. However, Segment Tree uses O(4N) space versus O(N) for Fenwick. So Segment Tree is faster to build but uses more memory. For most problems, the O(N log N) build is acceptable.',
          keyPoints: [
            'Segment Tree: O(N) build, O(4N) space',
            'Fenwick Tree: O(N log N) build, O(N) space',
            'Trade-off: build speed vs memory',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'fenwick-interview-1',
          question:
            'What are the key phrases in a problem statement that should make you think of Fenwick Tree?',
          hint: 'Think about what operations Fenwick Tree excels at.',
          sampleAnswer:
            'Key phrases include "prefix sum with updates", "range sum queries", "cumulative frequency", "count inversions", or "dynamic ranking". Any time you see both queries and updates on cumulative/additive data, Fenwick Tree is a strong candidate. Also look for "count elements smaller than X" in a dynamic array.',
          keyPoints: [
            '"Prefix sum" + "updates"',
            '"Range sum queries"',
            '"Cumulative" or "frequency"',
            '"Count inversions" or "count smaller"',
          ],
        },
        {
          id: 'fenwick-interview-2',
          question:
            'How should you explain the 1-indexing requirement to an interviewer?',
          hint: 'Connect it to the bit manipulation.',
          sampleAnswer:
            'Explain that Fenwick Tree uses 1-indexing because the bit operation "i & -i" equals 0 when i is 0, which would cause infinite loops. Starting from index 1 makes the bit manipulation work correctly. Mention that tree[0] is unused and you convert 0-indexed input by adding 1 to all indices.',
          keyPoints: [
            '"i & -i" fails for i=0',
            'tree[0] unused, start at 1',
            'Convert: input_index + 1',
          ],
        },
        {
          id: 'fenwick-interview-3',
          question:
            'What common mistake do candidates make when implementing Fenwick Tree updates?',
          hint: 'Think about what update does versus what people expect it to do.',
          sampleAnswer:
            'A common mistake is treating update as "set value" when it actually "adds delta". To set arr[i] to new_val, you must calculate delta = new_val - old_val, then call update(i, delta). Candidates who forget this will get wrong answers. Always track the original array separately if you need to set values.',
          keyPoints: [
            'Update adds delta, does not set value',
            'To set: compute delta = new - old',
            'Keep original array if needed',
          ],
        },
      ],
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
