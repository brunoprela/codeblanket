/**
 * Core Operations Section
 */

export const operationsSection = {
  id: 'operations',
  title: 'Core Operations',
  content: `**1. Range Query**
Find value for range [L, R].

**Three Cases:**
1. **No overlap**: Range completely outside [L, R]
2. **Complete overlap**: Range completely inside [L, R]
3. **Partial overlap**: Range partially overlaps [L, R]

\`\`python
def query (tree, node, start, end, L, R):
    """Query sum in range [L, R]"""
    # No overlap
    if R < start or L > end:
        return 0
    
    # Complete overlap
    if L <= start and end <= R:
        return tree[node]
    
    # Partial overlap - query both children
    mid = (start + end) // 2
    left_sum = query (tree, 2*node+1, start, mid, L, R)
    right_sum = query (tree, 2*node+2, mid+1, end, L, R)
    return left_sum + right_sum
\`\`\`

**2. Point Update**
Update single element at index.

\`\`\`python
def update (tree, node, start, end, idx, value):
    """Update element at index idx to value"""
    if start == end:
        # Leaf node
        tree[node] = value
        return
    
    mid = (start + end) // 2
    
    if idx <= mid:
        update (tree, 2*node+1, start, mid, idx, value)
    else:
        update (tree, 2*node+2, mid+1, end, idx, value)
    
    # Update current node
    tree[node] = tree[2*node+1] + tree[2*node+2]
\`\`\`

**Complexity:**
- Query: O(log N)
- Update: O(log N)`,
};
