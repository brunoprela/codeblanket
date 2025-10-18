/**
 * Complexity Analysis Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Complexity Analysis',
  content: `**Graph Algorithm Complexities:**

**Traversals (BFS/DFS):**
- **Time**: O(V + E)
  - Visit each vertex once: O(V)
  - Explore each edge once: O(E)
- **Space**: 
  - BFS: O(W) where W is maximum width
  - DFS: O(H) where H is height (recursion)

**Connected Components:**
- **Time**: O(V + E) - DFS/BFS on all nodes
- **Space**: O(V) for visited set

**Cycle Detection:**
- **Time**: O(V + E)
- **Space**: O(V)

**Topological Sort:**
- **Time**: O(V + E)
- **Space**: O(V)

**Shortest Path (Unweighted - BFS):**
- **Time**: O(V + E)
- **Space**: O(V)

**Dijkstra's Algorithm:**
- **Time**: O((V + E) log V) with min heap
- **Space**: O(V)

**Bellman-Ford (Negative Weights):**
- **Time**: O(V * E)
- **Space**: O(V)

**Floyd-Warshall (All Pairs):**
- **Time**: O(V³)
- **Space**: O(V²)

**Union-Find:**
- **Find**: O(α(N)) ≈ O(1) amortized with path compression
- **Union**: O(α(N)) ≈ O(1) amortized with rank/size
- **Space**: O(N)

**Graph Representation Space:**
- **Adjacency List**: O(V + E)
- **Adjacency Matrix**: O(V²)
- **Edge List**: O(E)

**Dense vs Sparse Graphs:**
- **Dense**: E ≈ V² → Use adjacency matrix
- **Sparse**: E << V² → Use adjacency list

**Key Insights:**
- Most algorithms linear in graph size: O(V + E)
- BFS optimal for unweighted shortest paths
- DFS uses less memory than BFS
- Union-Find nearly constant time operations`,
};
