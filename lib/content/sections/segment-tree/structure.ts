/**
 * Segment Tree Structure Section
 */

export const structureSection = {
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
def build (arr, tree, node, start, end):
    """Build segment tree recursively"""
    if start == end:
        # Leaf node
        tree[node] = arr[start]
        return
    
    mid = (start + end) // 2
    left_child = 2 * node + 1
    right_child = 2 * node + 2
    
    # Build left and right subtrees
    build (arr, tree, left_child, start, mid)
    build (arr, tree, right_child, mid + 1, end)
    
    # Internal node = merge of children
    tree[node] = tree[left_child] + tree[right_child]
\`\`\`

**Complexity:**
- Build: O(N)
- Space: O(N)`,
};
