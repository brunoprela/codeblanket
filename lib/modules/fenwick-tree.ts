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
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the main advantage of Fenwick Tree over Segment Tree?',
          options: [
            'Faster',
            'Simpler code (~20 lines vs 50+), less memory (N vs 4N)',
            'More powerful',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree is simpler to implement (fewer lines, less complex), uses less memory (N vs 4N), and easier to get right under time pressure. Same O(log N) complexity for range sums.',
        },
        {
          id: 'mc2',
          question: 'What operations can Fenwick Tree handle?',
          options: [
            'Any operation',
            'Operations with inverse: addition (subtract), XOR (XOR)',
            'Only sum',
            'Min/max',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree requires invertible operations. Addition (inverse: subtraction) and XOR (inverse: XOR itself) work. Min/max/GCD don\'t work (no inverse to "undo" operation).',
        },
        {
          id: 'mc3',
          question:
            'When should you use Fenwick Tree over simple prefix sum array?',
          options: [
            'Never',
            'When you need both queries and updates - prefix sum has O(N) update cost',
            'Always',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Prefix sum: O(1) query but O(N) update (recalculate all sums). Fenwick Tree: O(log N) for both. Use Fenwick when updates are frequent.',
        },
        {
          id: 'mc4',
          question: 'What is the time complexity of Fenwick Tree operations?',
          options: [
            'O(N)',
            'Build: O(N log N), Query: O(log N), Update: O(log N)',
            'All O(1)',
            'O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree: Build O(N log N) with N updates, each O(log N). Query O(log N) sums logarithmic ranges. Point Update O(log N) updates logarithmic ancestors.',
        },
        {
          id: 'mc5',
          question: 'What does BIT stand for and why that name?',
          options: [
            'Basic Integer Tree',
            'Binary Indexed Tree - uses binary representation of indices',
            'Best Implementation Tool',
            'Random name',
          ],
          correctAnswer: 1,
          explanation:
            'BIT = Binary Indexed Tree. Uses bit manipulation on binary representation of indices to determine parent/child relationships. Each bit determines range responsibility.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does i & -i compute in Fenwick Tree?',
          options: [
            'Random value',
            'Isolates the last set bit (rightmost 1) - determines range size',
            'Doubles i',
            'Negates i',
          ],
          correctAnswer: 1,
          explanation:
            "i & -i isolates the last set bit using two's complement trick. This determines how many elements index i is responsible for. Example: 12 (1100) & -12 = 4 (0100), so index 12 covers 4 elements.",
        },
        {
          id: 'mc2',
          question: 'How do you find the parent of index i in Fenwick Tree?',
          options: ['i/2', 'i + (i & -i) - add last set bit', 'i-1', 'i*2'],
          correctAnswer: 1,
          explanation:
            'Parent of i is i + (i & -i). This adds the last set bit, moving to the next index responsible for a larger range. Example: parent of 4 (100) is 4+4=8 (1000).',
        },
        {
          id: 'mc3',
          question: 'What range does index i cover in Fenwick Tree?',
          options: [
            'Single element',
            'Range [i - (i & -i) + 1, i] with length (i & -i)',
            'Entire array',
            'Random range',
          ],
          correctAnswer: 1,
          explanation:
            'Index i covers (i & -i) elements ending at position i. Range is [i - (i & -i) + 1, i]. Example: index 6 (110) covers 6 & -6 = 2 elements: positions 5 and 6.',
        },
        {
          id: 'mc4',
          question: 'Why are Fenwick Tree indices 1-based?',
          options: [
            'Random choice',
            'Bit manipulation (i & -i) fails for index 0 - would be 0',
            'Easier to read',
            'Historical reasons',
          ],
          correctAnswer: 1,
          explanation:
            'Index 0 has no set bits, so 0 & -0 = 0, breaking the algorithm. Starting at index 1 ensures all indices have at least one set bit, making bit manipulation work correctly.',
        },
        {
          id: 'mc5',
          question: 'What makes Fenwick Tree elegant compared to Segment Tree?',
          options: [
            'Faster',
            'Uses bit manipulation tricks for parent/child navigation - no explicit tree structure',
            'More powerful',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree uses bit manipulation (i & -i, i + (i & -i), i - (i & -i)) to implicitly represent tree structure in a simple array. No pointers or complex indexing like Segment Tree.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How do you perform a point update in Fenwick Tree?',
          options: [
            'Update all indices',
            'Start at i, add delta, move to parent with i += i & -i until i > N',
            'Rebuild tree',
            'Update i only',
          ],
          correctAnswer: 1,
          explanation:
            'Point update: add delta to tree[i], then move up to parent (i += i & -i) and update all ancestors. This updates all ranges containing index i in O(log N) time.',
        },
        {
          id: 'mc2',
          question: 'How do you compute prefix sum for index i?',
          options: [
            'Sum all elements',
            'Start at i, sum tree[i], move down with i -= i & -i until i = 0',
            'tree[i] only',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Prefix sum: sum tree[i], then move to previous range (i -= i & -i) and continue until i = 0. This sums O(log N) precomputed ranges that combine to give prefix sum.',
        },
        {
          id: 'mc3',
          question: 'How do you compute range sum [L, R]?',
          options: [
            'Sum elements one by one',
            'prefix_sum(R) - prefix_sum(L-1) - uses difference of cumulative sums',
            'Impossible',
            'Binary search',
          ],
          correctAnswer: 1,
          explanation:
            'Range sum [L,R] = sum from 1 to R minus sum from 1 to L-1. Compute prefix_sum(R) - prefix_sum(L-1). Each prefix sum is O(log N), so range sum is O(log N).',
        },
        {
          id: 'mc4',
          question:
            'Why does update move up (i += i & -i) while query moves down (i -= i & -i)?',
          options: [
            'Random',
            'Update affects ancestors (larger ranges), query combines smaller ranges into prefix',
            'They are the same',
            'Error in algorithm',
          ],
          correctAnswer: 1,
          explanation:
            'Update: changing element affects all parent ranges containing it (move up). Query: prefix sum is built from smaller disjoint ranges (move down). Different traversals for different purposes.',
        },
        {
          id: 'mc5',
          question:
            'What is the time complexity of each Fenwick Tree operation?',
          options: [
            'O(N)',
            'Update: O(log N), Prefix sum: O(log N), Range sum: O(log N)',
            'All O(1)',
            'O(N log N)',
          ],
          correctAnswer: 1,
          explanation:
            'All operations are O(log N) because they traverse at most log N indices (tree height). Update moves up log N parents, query sums log N ranges.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a 2D Fenwick Tree used for?',
          options: [
            'Sorting 2D arrays',
            'Range sum queries on 2D matrix (rectangle sums) in O(log²N) time',
            'Graph traversal',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            '2D Fenwick Tree handles 2D range queries like rectangle sums. Uses nested Fenwick Trees: outer for rows, inner for columns. Update and query both O(log M × log N).',
        },
        {
          id: 'mc2',
          question: 'How do you count inversions using Fenwick Tree?',
          options: [
            'Cannot do',
            'For each element, count how many larger elements came before (coordinate compression + range query)',
            'Sort array',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Inversions: process left to right. For each element x, query sum of elements > x seen so far. Update tree at position x. With coordinate compression for space efficiency. O(N log N) time.',
        },
        {
          id: 'mc3',
          question: 'What does coordinate compression do for Fenwick Tree?',
          options: [
            'Compresses data',
            'Maps large value range to smaller [1, N] for space efficiency',
            'Random',
            'Deletes data',
          ],
          correctAnswer: 1,
          explanation:
            'When values are large (e.g., 10^9) but only N distinct values, map them to [1,N]. Sorts unique values, creates mapping. Fenwick Tree size becomes O(N) instead of O(max_value).',
        },
        {
          id: 'mc4',
          question: 'Can Fenwick Tree handle range updates efficiently?',
          options: [
            'No',
            'Yes with difference array technique - update endpoints O(log N)',
            'Only point updates',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Range update [L,R] += delta: use difference array. Update diff[L] += delta, diff[R+1] -= delta. Query reconstructs by prefix sum of differences. Range update becomes O(log N).',
        },
        {
          id: 'mc5',
          question: 'What makes Fenwick Tree implementation elegant?',
          options: [
            'Random',
            'Just 2 simple functions using bit manipulation - ~20 lines total',
            'Uses complex data structures',
            'Always fast',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree beauty: entire implementation is 2 functions (update, prefix_sum), each 3-5 lines with simple bit manipulation. No complex tree structures or recursion. Very interview-friendly.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'When should you use Fenwick Tree over Segment Tree?',
          options: [
            'Always',
            'Simpler problem (sum/XOR), want shorter code, operation has inverse',
            'Never',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            "Use Fenwick when: 1) Operation is invertible (sum, XOR), 2) Don't need min/max/GCD, 3) Want simpler code (~20 lines vs 50+), 4) Interview time pressure. Same O(log N) complexity.",
        },
        {
          id: 'mc2',
          question: 'When must you use Segment Tree instead of Fenwick Tree?',
          options: [
            'Random',
            'Need min/max/GCD (no inverse) or lazy propagation for range updates',
            'Always',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Use Segment Tree when: 1) Operation has no inverse (min, max, GCD), 2) Need lazy propagation for efficient range updates, 3) Complex custom operations. Fenwick is limited to invertible operations.',
        },
        {
          id: 'mc3',
          question: 'What are the code complexity differences?',
          options: [
            'Same',
            'Fenwick: ~20 lines simple, Segment: ~50+ lines complex',
            'Fenwick longer',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree: 2 simple functions, ~20 lines, straightforward bit manipulation. Segment Tree: recursive build/query/update, lazy propagation, ~50+ lines, more complex. Big difference in interview settings.',
        },
        {
          id: 'mc4',
          question: 'What memory differences exist?',
          options: [
            'Same',
            'Fenwick: N+1 array, Segment: 4N array - Fenwick uses 25% space',
            'Segment uses less',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree uses N+1 space (1-indexed array). Segment Tree uses 4N for safety. Fenwick is 4x more space-efficient, though both are O(N).',
        },
        {
          id: 'mc5',
          question: 'Performance in practice?',
          options: [
            'Segment always faster',
            'Similar O(log N), but Fenwick often faster due to simpler operations and better constants',
            'Fenwick always slower',
            'No difference',
          ],
          correctAnswer: 1,
          explanation:
            'Both O(log N), but Fenwick often faster in practice: simpler operations (just addition + bit manipulation), better cache locality (smaller structure), lower constant factors.',
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
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What keywords signal using Fenwick Tree in an interview?',
          options: [
            'Sorting',
            'Prefix sum, range sum, cumulative frequency, inversions - with updates',
            'Shortest path',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree signals: "prefix sum", "range sum", "cumulative frequency", "count inversions", "dynamic array queries". Key: need both queries AND updates efficiently.',
        },
        {
          id: 'mc2',
          question: 'What should you clarify in a Fenwick Tree interview?',
          options: [
            'Nothing',
            'Operation type (sum/XOR?), update frequency, can I use 1-indexing?',
            'Random',
            'Language only',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: 1) What operation (if min/max, need Segment Tree), 2) How frequent are updates (if none, prefix sum array faster), 3) Can use 1-indexing (standard for Fenwick), 4) Value range (coordinate compression?).',
        },
        {
          id: 'mc3',
          question: 'What is a common mistake when implementing Fenwick Tree?',
          options: [
            'Using bit manipulation',
            'Forgetting 1-based indexing, off-by-one in range queries',
            'Good naming',
            'Comments',
          ],
          correctAnswer: 1,
          explanation:
            'Common mistakes: 1) Using 0-based indexing (breaks algorithm), 2) Range query: prefix_sum(R) - prefix_sum(L-1), forgetting L-1, 3) Update vs set confusion (Fenwick adds delta, not sets value).',
        },
        {
          id: 'mc4',
          question: 'How should you communicate your Fenwick Tree solution?',
          options: [
            'Just code',
            'Explain why Fenwick (O(log N) for sum+update), mention bit manipulation briefly, walk through example, complexity',
            'No explanation',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Communication: 1) Why Fenwick over alternatives (simpler than Segment Tree for sums), 2) Briefly mention bit manipulation concept, 3) Walk through one update and query, 4) Time O(log N), space O(N), 5) Code with comments.',
        },
        {
          id: 'mc5',
          question: 'What makes Fenwick Tree interview-friendly?',
          options: [
            'Random',
            'Short code (~20 lines), easy to memorize, fewer bugs than Segment Tree',
            'Complex',
            'Slow',
          ],
          correctAnswer: 1,
          explanation:
            'Fenwick Tree is interview-friendly: 1) Only ~20 lines total, 2) 2 simple functions, easy to memorize, 3) Less error-prone than Segment Tree (50+ lines), 4) Same O(log N) performance.',
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
