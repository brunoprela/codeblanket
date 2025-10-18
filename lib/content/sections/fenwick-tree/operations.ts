/**
 * Core Operations Section
 */

export const operationsSection = {
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
};
