/**
 * Minimum Spanning Tree (MST) Section
 */

export const mstSection = {
  id: 'mst',
  title: 'Minimum Spanning Tree (MST)',
  content: `A **Minimum Spanning Tree** connects all vertices in a weighted graph with minimum total edge weight, with no cycles.

**Properties:**
- Connects all V vertices with exactly V-1 edges
- No cycles (it's a tree!)
- Minimizes sum of edge weights
- Not necessarily unique

**Two Main Algorithms:**

**1. Kruskal\'s Algorithm (Edge-based)**
- Sort edges by weight
- Use Union-Find to avoid cycles
- Add edges greedily if they don't form cycle

\`\`\`python
def kruskal_mst (edges, n):
    """
    edges: [(weight, u, v), ...]
    n: number of vertices
    """
    # Sort edges by weight
    edges.sort()

    uf = UnionFind (n)
    mst = []
    total_weight = 0

    for weight, u, v in edges:
        if uf.union (u, v):  # No cycle
            mst.append((u, v, weight))
            total_weight += weight

            if len (mst) == n - 1:  # Found MST
                break

    return mst, total_weight
\`\`\`

**Complexity:** O(E log E) for sorting + O(E α(V)) for union-find ≈ O(E log E)

**2. Prim's Algorithm (Vertex-based)**
- Start from any vertex
- Repeatedly add minimum-weight edge connecting tree to non-tree vertex
- Use min-heap for efficiency

\`\`\`python
import heapq

def prim_mst (graph, n):
    """
    graph: {node: [(weight, neighbor), ...]}
    n: number of vertices
    """
    visited = set()
    mst = []
    total_weight = 0

    # Start from vertex 0
    visited.add(0)
    heap = graph[0][:]  # edges from start
    heapq.heapify (heap)

    while heap and len (visited) < n:
        weight, u, v = heapq.heappop (heap)

        if v in visited:
            continue

        # Add edge to MST
        visited.add (v)
        mst.append((u, v, weight))
        total_weight += weight

        # Add edges from newly added vertex
        for w, neighbor in graph[v]:
            if neighbor not in visited:
                heapq.heappush (heap, (w, v, neighbor))

    return mst, total_weight
\`\`\`

**Complexity:** O((V + E) log V) with binary heap

**Kruskal vs Prim:**

| Aspect | Kruskal | Prim |
|--------|---------|------|
| Approach | Edge-based | Vertex-based |
| Data Structure | Union-Find | Min-Heap |
| Complexity | O(E log E) | O((V+E) log V) |
| Best for | Sparse graphs | Dense graphs |
| Edge list | Yes | Adjacency list better |

**When to use which:**
- **Kruskal**: Sparse graph, have edge list, simpler to code
- **Prim**: Dense graph, adjacency list available`,
};
