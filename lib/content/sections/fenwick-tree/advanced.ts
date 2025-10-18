/**
 * Advanced Techniques Section
 */

export const advancedSection = {
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
};
