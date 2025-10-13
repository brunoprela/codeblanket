import { Module } from '@/lib/types';

export const advancedGraphsModule: Module = {
  id: 'advanced-graphs',
  title: 'Advanced Graphs',
  description:
    'Master advanced graph algorithms including shortest paths, minimum spanning trees, and network flow.',
  icon: '🗺️',
  timeComplexity: 'Varies by algorithm (O(E log V) to O(V³))',
  spaceComplexity: 'O(V) to O(V²)',
  sections: [
    {
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
    },
    {
      id: 'dijkstra',
      title: "Dijkstra's Algorithm",
      content: `**Dijkstra's Algorithm** finds the shortest path from a source to all other vertices in a **weighted graph with non-negative weights**.

**Algorithm:**
1. Initialize distances: source = 0, others = ∞
2. Use min-heap to always process closest unvisited vertex
3. For each neighbor, try to relax (improve) distance
4. Mark vertex as visited once processed

**Key Insight:**
Greedy approach - always extend the shortest known path.

**Complexity:**
- Time: O((V + E) log V) with min-heap
- Space: O(V)

**When to Use:**
- Non-negative edge weights only!
- Single-source shortest path
- GPS, network routing

**Example:**
\`\`\`
Graph:
A --1--> B
|        |
2        3
|        |
v        v
C --1--> D

Shortest paths from A:
A → A: 0
A → B: 1
A → C: 2
A → D: 3 (via A→B→D or A→C→D)
\`\`\`

**Why Non-Negative Weights:**
Negative weights can create shorter paths after a node is "finalized", breaking the greedy assumption.`,
      codeExample: `import heapq
from typing import Dict, List, Tuple


def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
    """
    Dijkstra's shortest path algorithm.
    graph: adjacency list {node: [(neighbor, weight), ...]}
    Time: O((V + E) log V), Space: O(V)
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Min-heap: (distance, node)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        visited.add(node)
        
        # Relax edges
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances


# With path reconstruction
def dijkstra_with_path(graph, start):
    """
    Returns both distances and predecessors for path reconstruction.
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {node: None for node in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = node
                heapq.heappush(pq, (distance, neighbor))
    
    return distances, predecessors


def reconstruct_path(predecessors, start, end):
    """Reconstruct shortest path from start to end."""
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    path.reverse()
    return path if path[0] == start else []`,
    },
    {
      id: 'bellman-ford',
      title: 'Bellman-Ford Algorithm',
      content: `**Bellman-Ford** finds shortest paths even with **negative edge weights**, and detects **negative cycles**.

**Algorithm:**
1. Initialize distances: source = 0, others = ∞
2. Relax all edges V-1 times
3. Check for negative cycles (one more relaxation)

**Key Insight:**
In a graph with V vertices, shortest path has at most V-1 edges. Relax all edges V-1 times guarantees finding shortest paths.

**Complexity:**
- Time: O(V * E)
- Space: O(V)

**Advantages over Dijkstra:**
- ✅ Handles negative weights
- ✅ Detects negative cycles
- ❌ Slower than Dijkstra

**Example with Negative Weight:**
\`\`\`
A --2--> B
|        |
1       -3
|        |
v        v
C <--1-- D

Shortest A → D:
Via A→B→D: 2 + (-3) = -1 (shortest!)
Via A→C→D: 1 + 1 = 2
\`\`\`

**Negative Cycle:**
If relaxation happens in Vth iteration, negative cycle exists (distances keep decreasing infinitely).`,
      codeExample: `from typing import Dict, List, Tuple


def bellman_ford(
    graph: Dict[int, List[Tuple[int, int]]], 
    start: int
) -> Tuple[Dict[int, int], bool]:
    """
    Bellman-Ford algorithm.
    Returns (distances, has_negative_cycle)
    Time: O(V * E), Space: O(V)
    """
    # Get all vertices
    vertices = set(graph.keys())
    for node in graph:
        for neighbor, _ in graph[node]:
            vertices.add(neighbor)
    
    # Initialize
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0
    
    # Relax all edges V-1 times
    for _ in range(len(vertices) - 1):
        for node in graph:
            if distances[node] == float('inf'):
                continue
            
            for neighbor, weight in graph[node]:
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
    
    # Check for negative cycles
    for node in graph:
        if distances[node] == float('inf'):
            continue
        
        for neighbor, weight in graph[node]:
            if distances[node] + weight < distances[neighbor]:
                return distances, True  # Negative cycle exists
    
    return distances, False


# Edge list version (simpler)
def bellman_ford_edges(edges: List[Tuple[int, int, int]], n: int, start: int):
    """
    edges: [(from, to, weight), ...]
    n: number of vertices (0 to n-1)
    """
    distances = [float('inf')] * n
    distances[start] = 0
    
    # Relax n-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
    
    # Check negative cycle
    for u, v, w in edges:
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            return None  # Negative cycle
    
    return distances`,
    },
    {
      id: 'floyd-warshall',
      title: 'Floyd-Warshall Algorithm',
      content: `**Floyd-Warshall** finds shortest paths between **all pairs** of vertices using **dynamic programming**.

**Algorithm:**
For each intermediate vertex k:
  For each pair (i, j):
    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

**Key Insight:**
Try using each vertex as intermediate point. If path through k is shorter, use it.

**Complexity:**
- Time: O(V³)
- Space: O(V²)

**When to Use:**
- Need all-pairs shortest paths
- Dense graph (many edges)
- Small number of vertices (≤ 400)

**Example:**
\`\`\`
Initial:
  A  B  C
A 0  1  ∞
B ∞  0  1
C 1  ∞  0

After considering each vertex:
  A  B  C
A 0  1  2
B 2  0  1
C 1  2  0
\`\`\`

**Advantages:**
- Simple to implement
- Finds all pairs at once
- Handles negative weights
- Can detect negative cycles (dist[i][i] < 0)

**Disadvantages:**
- O(V³) time (slow for large graphs)
- O(V²) space (stores all pairs)`,
      codeExample: `from typing import List


def floyd_warshall(graph: List[List[int]]) -> List[List[int]]:
    """
    Floyd-Warshall all-pairs shortest path.
    graph: adjacency matrix (graph[i][j] = weight from i to j)
           use float('inf') for no edge
    Time: O(V³), Space: O(V²)
    """
    n = len(graph)
    
    # Copy graph (don't modify input)
    dist = [row[:] for row in graph]
    
    # Try each vertex as intermediate
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Can we go through k?
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist


def floyd_warshall_with_path(graph):
    """
    With path reconstruction.
    Returns (distances, next_vertex_on_path)
    """
    n = len(graph)
    dist = [row[:] for row in graph]
    next_vertex = [[j if graph[i][j] != float('inf') else None 
                    for j in range(n)] 
                   for i in range(n)]
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]
    
    return dist, next_vertex


def has_negative_cycle(dist):
    """Check if graph has negative cycle."""
    n = len(dist)
    for i in range(n):
        if dist[i][i] < 0:
            return True
    return False


def reconstruct_path_fw(next_vertex, i, j):
    """Reconstruct path from i to j."""
    if next_vertex[i][j] is None:
        return []
    
    path = [i]
    while i != j:
        i = next_vertex[i][j]
        path.append(i)
    
    return path`,
    },
    {
      id: 'union-find',
      title: 'Union-Find (Disjoint Set Union)',
      content: `**Union-Find** (also called Disjoint Set Union or DSU) is a data structure that tracks elements partitioned into disjoint (non-overlapping) sets.

**Core Operations:**
- \`find(x)\`: Find which set x belongs to (returns representative/root)
- \`union(x, y)\`: Merge the sets containing x and y

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
        self.parent = list(range(n))  # Each node is its own parent
    
    def find(self, x):
        """Find root of x"""
        if self.parent[x] != x:
            return self.find(self.parent[x])
        return x
    
    def union(self, x, y):
        """Merge sets containing x and y"""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
\`\`\`

**Optimization 1: Path Compression**
Make tree flatter by pointing nodes directly to root during find.

\`\`\`python
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # Path compression!
    return self.parent[x]
\`\`\`

**Optimization 2: Union by Rank**
Attach smaller tree under larger tree to keep trees balanced.

\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n  # Tree height
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
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
def has_cycle(edges, n):
    uf = UnionFind(n)
    for u, v in edges:
        if uf.find(u) == uf.find(v):
            return True  # Cycle detected!
        uf.union(u, v)
    return False
\`\`\``,
      codeExample: `class UnionFind:
    """Optimized Union-Find with path compression and union by rank"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n  # Track number of disjoint sets
    
    def find(self, x: int) -> int:
        """Find root with path compression. O(α(n)) ≈ O(1)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union by rank. Returns True if merged, False if already connected.
        O(α(n)) ≈ O(1)
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
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
        
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set. O(α(n))"""
        return self.find(x) == self.find(y)
    
    def count_components(self) -> int:
        """Get number of disjoint sets. O(1)"""
        return self.components


# Example: Detect cycle in undirected graph
def has_cycle_undirected(edges, n):
    """Returns True if graph has a cycle"""
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return True  # Edge connects already-connected nodes
    return False


# Example: Count connected components
def count_components(edges, n):
    """Count number of connected components"""
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count_components()`,
    },
    {
      id: 'mst',
      title: 'Minimum Spanning Tree (MST)',
      content: `A **Minimum Spanning Tree** connects all vertices in a weighted graph with minimum total edge weight, with no cycles.

**Properties:**
- Connects all V vertices with exactly V-1 edges
- No cycles (it's a tree!)
- Minimizes sum of edge weights
- Not necessarily unique

**Two Main Algorithms:**

**1. Kruskal's Algorithm (Edge-based)**
- Sort edges by weight
- Use Union-Find to avoid cycles
- Add edges greedily if they don't form cycle

\`\`\`python
def kruskal_mst(edges, n):
    """
    edges: [(weight, u, v), ...]
    n: number of vertices
    """
    # Sort edges by weight
    edges.sort()
    
    uf = UnionFind(n)
    mst = []
    total_weight = 0
    
    for weight, u, v in edges:
        if uf.union(u, v):  # No cycle
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == n - 1:  # Found MST
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

def prim_mst(graph, n):
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
    heapq.heapify(heap)
    
    while heap and len(visited) < n:
        weight, u, v = heapq.heappop(heap)
        
        if v in visited:
            continue
        
        # Add edge to MST
        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight
        
        # Add edges from newly added vertex
        for w, neighbor in graph[v]:
            if neighbor not in visited:
                heapq.heappush(heap, (w, v, neighbor))
    
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
      codeExample: `import heapq
from typing import List, Tuple


class UnionFind:
    """For Kruskal's algorithm"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True


def kruskal_mst(edges: List[Tuple[int, int, int]], n: int):
    """
    Kruskal's MST algorithm.
    edges: [(u, v, weight), ...]
    Returns: (mst_edges, total_weight)
    Time: O(E log E), Space: O(V)
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == n - 1:
                break
    
    return mst, total_weight


def prim_mst(graph: dict, n: int, start: int = 0):
    """
    Prim's MST algorithm.
    graph: {node: [(neighbor, weight), ...]}
    Returns: (mst_edges, total_weight)
    Time: O((V+E) log V), Space: O(V)
    """
    visited = set([start])
    mst = []
    total_weight = 0
    
    # Min-heap: (weight, from_node, to_node)
    heap = [(weight, start, neighbor) 
            for neighbor, weight in graph[start]]
    heapq.heapify(heap)
    
    while heap and len(visited) < n:
        weight, u, v = heapq.heappop(heap)
        
        if v in visited:
            continue
        
        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight
        
        # Add edges from newly added vertex
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush(heap, (w, v, neighbor))
    
    return mst, total_weight


# Example usage
if __name__ == "__main__":
    # Example graph
    edges = [
        (0, 1, 4),
        (0, 2, 3),
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 4),
    ]
    
    n = 4
    mst, weight = kruskal_mst(edges, n)
    print(f"Kruskal MST: {mst}, Total weight: {weight}")
    
    # Convert to adjacency list for Prim
    graph = {i: [] for i in range(n)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    mst, weight = prim_mst(graph, n)
    print(f"Prim MST: {mst}, Total weight: {weight}")`,
    },
    {
      id: 'comparison',
      title: 'Algorithm Comparison',
      content: `**Shortest Path Algorithm Selection:**

| Algorithm | Use Case | Time | Space | Negative Weights | Negative Cycles |
|-----------|----------|------|-------|------------------|-----------------|
| BFS | Unweighted | O(V+E) | O(V) | N/A | No |
| Dijkstra | Non-negative weights | O((V+E)logV) | O(V) | ❌ No | N/A |
| Bellman-Ford | Negative weights OK | O(VE) | O(V) | ✅ Yes | ✅ Detects |
| Floyd-Warshall | All pairs | O(V³) | O(V²) | ✅ Yes | ✅ Detects |

**Decision Tree:**

\`\`\`
Need shortest path?
│
├─ Unweighted graph?
│  └─ Use BFS (O(V+E))
│
├─ Single source?
│  │
│  ├─ Non-negative weights?
│  │  └─ Use Dijkstra (O((V+E)logV))
│  │
│  └─ Negative weights possible?
│     └─ Use Bellman-Ford (O(VE))
│
└─ All pairs?
   │
   ├─ Sparse graph, many queries?
   │  └─ Run Dijkstra V times
   │
   └─ Dense graph or small V?
      └─ Use Floyd-Warshall (O(V³))
\`\`\`

**Practical Guidelines:**

**Use Dijkstra when:**
- GPS/navigation (non-negative distances)
- Network routing (non-negative costs)
- Most real-world shortest path problems

**Use Bellman-Ford when:**
- Arbitrage detection (negative cycles)
- Some financial models
- Need to detect negative cycles

**Use Floyd-Warshall when:**
- Small graph (V ≤ 400)
- Need many shortest paths
- Transitive closure problems

**Use BFS when:**
- Unweighted graph (or all weights equal)
- Simplest and fastest for this case`,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Advanced Graphs when you see:**
- "Shortest path", "minimum cost", "cheapest route"
- "Weighted graph", "edge costs"
- "Network", "cities connected", "flights"
- "Negative weights", "negative cycle"
- "All pairs shortest path"

---

**Problem-Solving Steps:**

**Step 1: Identify Problem Type (2 min)**
- Single source or all pairs?
- Weighted or unweighted?
- Negative weights possible?
- Need to detect cycles?

**Step 2: Choose Algorithm (2 min)**
- Follow decision tree above
- Consider time/space constraints

**Step 3: Implementation (15 min)**
- Dijkstra: Min-heap + relaxation
- Bellman-Ford: Relax V-1 times
- Floyd-Warshall: Triple nested loop

**Step 4: Edge Cases (2 min)**
- Disconnected graph
- Negative cycles
- Source unreachable
- Self-loops

---

**Common Mistakes:**

**1. Using Dijkstra with Negative Weights**
Will give wrong answer! Use Bellman-Ford.

**2. Not Checking for Negative Cycles**
Bellman-Ford can detect them - use it!

**3. Wrong Heap Priority**
Dijkstra: always pop minimum distance.

**4. Forgetting Visited Set**
Dijkstra: mark as visited after popping.

---

**Interview Communication:**

*Interviewer: Find shortest path in weighted graph.*

**You:**
1. "Are all weights non-negative?" → Use Dijkstra
2. "Can weights be negative?" → Use Bellman-Ford
3. "Need all pairs?" → Consider Floyd-Warshall

4. **Dijkstra Explanation:**
   - "Use min-heap with (distance, node)."
   - "Always extend shortest known path (greedy)."
   - "Relax neighbors: update if shorter path found."
   - "O((V+E)logV) time, O(V) space."`,
    },
  ],
  keyTakeaways: [
    "Dijkstra's: Fastest for single-source with non-negative weights - O((V+E)logV)",
    'Bellman-Ford: Handles negative weights and detects cycles - O(VE)',
    'Floyd-Warshall: All-pairs shortest path using DP - O(V³)',
    'Use BFS for unweighted graphs (simplest)',
    'Dijkstra uses min-heap and greedy approach (always extend shortest path)',
    'Bellman-Ford relaxes all edges V-1 times',
    'Floyd-Warshall tries each vertex as intermediate point',
    'Never use Dijkstra with negative weights - it will fail!',
  ],
  relatedProblems: [
    'network-delay-time',
    'cheapest-flights-within-k-stops',
    'path-with-minimum-effort',
  ],
};
