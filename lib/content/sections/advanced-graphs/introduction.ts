/**
 * Advanced Graph Algorithms Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Advanced Graph Algorithms',
  content: `Beyond basic traversals (BFS/DFS), advanced graph algorithms solve optimization problems like finding shortest paths, minimum spanning trees, and maximum flow.

**Categories of Advanced Algorithms:**

**1. Shortest Path Algorithms**
- **Single-source shortest path**: One start to all destinations
  - Dijkstra's (non-negative weights)
  - Bellman-Ford (negative weights allowed)
- **All-pairs shortest path**: Every pair of vertices
  - Floyd-Warshall

**2. Minimum Spanning Tree (MST)**
- Connect all vertices with minimum total edge weight
  - Prim's Algorithm
  - Kruskal's Algorithm

**3. Network Flow**
- Maximum flow through a network
  - Ford-Fulkerson
  - Edmonds-Karp

**4. Advanced Traversals**
- Strongly Connected Components (Kosaraju, Tarjan)
- Articulation Points and Bridges
  
**When to Use:**
- GPS navigation → Dijkstra's
- Network routing → Bellman-Ford
- City planning → MST
- Resource allocation → Max Flow
- Web crawling → SCC`,
};
