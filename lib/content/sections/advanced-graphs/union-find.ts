/**
 * Union-Find (Disjoint Set Union) Section
 */

export const unionfindSection = {
  id: 'union-find',
  title: 'Union-Find (Disjoint Set Union)',
  content: `**Union-Find** (also called Disjoint Set Union or DSU) is a data structure that tracks elements partitioned into disjoint (non-overlapping) sets.

**Core Operations:**
- \`find (x)\`: Find which set x belongs to (returns representative/root)
- \`union (x, y)\`: Merge the sets containing x and y

**Applications:**
- **Kruskal's MST algorithm**
- Detecting cycles in undirected graphs
- Finding connected components
- Network connectivity
- Percolation problems

**Basic Implementation:**
\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list (range (n))  # Each node is its own parent

    def find (self, x):
        """Find root of x"""
        if self.parent[x] != x:
            return self.find (self.parent[x])
        return x

    def union (self, x, y):
        """Merge sets containing x and y"""
        root_x = self.find (x)
        root_y = self.find (y)
        if root_x != root_y:
            self.parent[root_x] = root_y
\`\`\`

**Optimization 1: Path Compression**
Make tree flatter by pointing nodes directly to root during find.

\`\`\`python
def find (self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find (self.parent[x])  # Path compression!
    return self.parent[x]
\`\`\`

**Optimization 2: Union by Rank**
Attach smaller tree under larger tree to keep trees balanced.

\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list (range (n))
        self.rank = [0] * n  # Tree height

    def find (self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find (self.parent[x])
        return self.parent[x]

    def union (self, x, y):
        root_x = self.find (x)
        root_y = self.find (y)

        if root_x == root_y:
            return False  # Already in same set

        # Attach smaller rank tree under larger rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True  # Successfully merged
\`\`\`

**Complexity with Both Optimizations:**
- Time: O(α(n)) ≈ O(1) where α is inverse Ackermann (effectively constant)
- Space: O(n)

**Common Pattern - Cycle Detection:**
\`\`\`python
def has_cycle (edges, n):
    uf = UnionFind (n)
    for u, v in edges:
        if uf.find (u) == uf.find (v):
            return True  # Cycle detected!
        uf.union (u, v)
    return False
\`\`\``,
};
