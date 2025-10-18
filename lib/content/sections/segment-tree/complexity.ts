/**
 * Complexity Analysis Section
 */

export const complexitySection = {
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
};
