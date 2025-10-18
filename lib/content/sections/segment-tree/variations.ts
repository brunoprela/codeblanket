/**
 * Segment Tree Variations Section
 */

export const variationsSection = {
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
};
