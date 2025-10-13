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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between basic graph algorithms and advanced graph algorithms. What makes an algorithm "advanced"?',
          sampleAnswer:
            'Basic algorithms (BFS, DFS, topological sort) work on unweighted graphs or simple problems, run in O(V+E). Advanced algorithms handle weighted graphs, find optimal paths, detect special structures. "Advanced" means: dealing with edge weights (Dijkstra, Bellman-Ford), all-pairs problems (Floyd-Warshall), graph connectivity (Union-Find, MST), optimization (shortest path, minimum spanning tree). For example, BFS finds if path exists (basic), Dijkstra finds shortest weighted path (advanced). DFS visits nodes (basic), Tarjan finds strongly connected components (advanced). Advanced algorithms are more complex, often involve greedy choices or dynamic programming, have higher complexity O(E log V) or O(V³). They solve optimization and structural problems that basic algorithms cannot.',
          keyPoints: [
            'Basic: unweighted, simple, O(V+E)',
            'Advanced: weighted, optimization, O(E log V) or O(V³)',
            'Handle: edge weights, all-pairs, connectivity',
            'Examples: Dijkstra, Floyd-Warshall, Union-Find',
            'Solve: shortest paths, MST, special structures',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare single-source vs all-pairs shortest path problems. When would you use each?',
          sampleAnswer:
            'Single-source finds shortest paths from one node to all others. Use Dijkstra O(E log V) or Bellman-Ford O(VE). All-pairs finds shortest paths between every pair of nodes. Use Floyd-Warshall O(V³). Choose single-source when: start node known (GPS navigation from current location), run multiple times for different sources (still faster than all-pairs if sources < V). Choose all-pairs when: need distances between all pairs (distance matrix), running single-source V times anyway, V is small (few hundred nodes). For example, GPS routing is single-source (from your location). City distance table is all-pairs (all city pairs). Time comparison: Dijkstra V times is O(VE log V), Floyd-Warshall is O(V³). For sparse graphs (E << V²), V × Dijkstra faster.',
          keyPoints: [
            'Single-source: one node to all, O(E log V)',
            'All-pairs: every pair, O(V³)',
            'Single when: known start, sparse graph',
            'All-pairs when: need full matrix, small V',
            'V × Dijkstra vs Floyd-Warshall depends on graph density',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe MST (Minimum Spanning Tree). What real-world problems does it solve?',
          sampleAnswer:
            'MST is tree connecting all vertices with minimum total edge weight. No cycles, n-1 edges for n vertices. Algorithms: Kruskal (sort edges, union-find) O(E log E), Prim (greedy with heap) O(E log V). Real-world: network design (minimum cable to connect all buildings), circuit design (minimum wire to connect pins), clustering (threshold on MST edges), approximation for TSP. For example, connecting cities with roads: MST gives minimum total road length while ensuring all cities connected. Utility network design: minimum pipe/cable to reach all customers. Key property: locally optimal edge choices (greedy) lead to globally optimal tree. MST is unique if all edge weights distinct. Multiple MSTs possible with duplicate weights.',
          keyPoints: [
            'Tree connecting all vertices, minimum total weight',
            'n-1 edges, no cycles',
            'Algorithms: Kruskal O(E log E), Prim O(E log V)',
            'Real-world: network design, clustering, TSP approx',
            'Greedy choices lead to global optimum',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What distinguishes advanced graph algorithms from basic traversals?',
          options: [
            'Just faster',
            'Solve optimization problems (shortest path, MST, max flow) vs simple exploration',
            'Random',
            'Use more memory',
          ],
          correctAnswer: 1,
          explanation:
            'Advanced algorithms solve optimization: shortest paths (Dijkstra, Bellman-Ford), MST (Prim, Kruskal), max flow, SCC. Basic (BFS/DFS) just explore/traverse. Different goals and complexities.',
        },
        {
          id: 'mc2',
          question: 'When should you use Dijkstra vs Bellman-Ford?',
          options: [
            'Always Dijkstra',
            'Dijkstra: non-negative weights O(E log V). Bellman-Ford: negative weights allowed O(VE)',
            'Random',
            'Same algorithm',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra faster O(E log V) but requires non-negative weights. Bellman-Ford slower O(VE) but handles negative weights and detects negative cycles. BFS for unweighted.',
        },
        {
          id: 'mc3',
          question: 'What is a Minimum Spanning Tree?',
          options: [
            'Shortest path',
            'Tree connecting all vertices with minimum total edge weight - no cycles',
            'Largest tree',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'MST: tree (no cycles) spanning all vertices with minimum sum of edge weights. Used in network design, clustering. Algorithms: Prim O(E log V), Kruskal O(E log E).',
        },
        {
          id: 'mc4',
          question: 'What problem does Floyd-Warshall solve?',
          options: [
            'Single-source shortest path',
            'All-pairs shortest paths - shortest between every pair of vertices',
            'MST',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd-Warshall: all-pairs shortest paths in O(V³). Computes shortest path between every pair. Good for dense graphs or when need all distances. Handles negative weights.',
        },
        {
          id: 'mc5',
          question: 'What is network flow used for?',
          options: [
            'Traversal',
            'Maximum flow through network (capacity constraints) - resource allocation, matching',
            'Sorting',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Network flow: maximum flow from source to sink respecting edge capacities. Applications: resource allocation, bipartite matching, assignment problems. Algorithms: Ford-Fulkerson, Edmonds-Karp.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain Dijkstra algorithm step by step. Why does it need non-negative weights?',
          sampleAnswer:
            'Dijkstra finds shortest paths from source using greedy approach. Steps: 1) Initialize distances (source=0, others=infinity), 2) Use min-heap with distances, 3) Pop node with minimum distance, 4) For each neighbor, if source→current→neighbor is shorter than current best to neighbor, update neighbor distance and add to heap, 5) Repeat until heap empty. Needs non-negative weights because greedy assumption: once a node is popped from heap (minimum distance found), that distance is final - no future path can be shorter. With negative weights, later path through negative edge could be shorter, breaking the guarantee. For example, A→B cost 5, B→C cost -10, A→C cost 3. Greedy picks A→C=3 as final, but A→B→C=-5 is actually shorter. Non-negative ensures processed nodes are finalized.',
          keyPoints: [
            'Greedy: always extend shortest known path',
            'Min-heap for next node with minimum distance',
            'Relax edges: update if shorter path found',
            'Needs non-negative: greedy assumption breaks otherwise',
            'Negative weights can create shorter paths later',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare Dijkstra implementation with priority queue vs simple array. What are the tradeoffs?',
          sampleAnswer:
            'Priority queue (min-heap) implementation: O((V+E) log V) time, O(V) space. Each vertex added/removed from heap O(log V), each edge relaxation potentially updates heap O(log V). Simple array: O(V²) time, O(V) space. Each iteration scans all vertices O(V) to find minimum, V iterations gives O(V²). Tradeoffs: heap is better for sparse graphs (E << V²), array is better for dense graphs (E ≈ V²) or small graphs. For example, 1000 vertices, 5000 edges (sparse): heap is O(5000 log 1000) ≈ 50K, array is O(1M). For complete graph with 1000 vertices, 500K edges (dense): heap is O(500K log 1000) ≈ 5M, array is O(1M). In practice, heap almost always preferred due to modern sparse graphs.',
          keyPoints: [
            'Heap: O((V+E) log V), better for sparse',
            'Array: O(V²), better for dense',
            'Heap: log V per operation, array: V per iteration',
            'Sparse graphs: heap wins',
            'Modern graphs usually sparse, prefer heap',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through reconstructing the shortest path from Dijkstra. Why do we need a parent/predecessor array?',
          sampleAnswer:
            'Dijkstra finds distances but path reconstruction needs parent array. During relaxation, when updating neighbor distance, also record parent[neighbor] = current. After algorithm finishes, backtrack from destination using parent array until reaching source, then reverse. For example, shortest path A→E: parent[B]=A, parent[D]=B, parent[E]=D. Backtrack: E→D→B→A, reverse to A→B→D→E. Without parent array, we only know distance is 15 but not which nodes to visit. The parent array traces the actual path by recording how we reached each node. Alternative: during relaxation, store entire path, but this uses O(V²) space vs O(V) for parent array. Parent array is space-efficient way to reconstruct optimal path.',
          keyPoints: [
            'Parent array tracks how we reached each node',
            'Update parent during edge relaxation',
            'Backtrack from dest to source, then reverse',
            'O(V) space vs O(V²) for storing full paths',
            'Essential for path reconstruction, not just distances',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: "What is Dijkstra's algorithm used for?",
          options: [
            'MST',
            'Single-source shortest path with non-negative weights',
            'All-pairs shortest path',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra finds shortest path from one source to all vertices. Requires non-negative edge weights. Uses priority queue (min-heap). O(E log V) time.',
        },
        {
          id: 'mc2',
          question: 'Why does Dijkstra use a priority queue?',
          options: [
            'Random choice',
            'Always process closest unvisited vertex next - greedy choice ensures optimality',
            'Faster traversal',
            'Required',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra greedy: always process vertex with minimum distance. Priority queue extracts min in O(log V). Ensures when vertex processed, shortest path found. Without queue: O(V²).',
        },
        {
          id: 'mc3',
          question: 'What is the time complexity of Dijkstra?',
          options: [
            'O(V)',
            'O(E log V) with min-heap, O(V²) with array',
            'O(V³)',
            'O(E)',
          ],
          correctAnswer: 1,
          explanation:
            'With min-heap: V extractions O(V log V) + E updates O(E log V) = O(E log V). With array: find min O(V) for V vertices = O(V²). Dense graphs: array better.',
        },
        {
          id: 'mc4',
          question: 'Why does Dijkstra fail with negative weights?',
          options: [
            'Random',
            'Greedy assumes found path is shortest - negative edges can improve later, violating assumption',
            'Too slow',
            'No reason',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra greedy: once vertex processed, distance is final. Negative edges can make later path shorter, breaking assumption. Example: A→B=5, A→C=2, C→B=-10 gives A→C→B=-8 < 5.',
        },
        {
          id: 'mc5',
          question: 'How do you reconstruct the shortest path in Dijkstra?',
          options: [
            'Cannot reconstruct',
            'Track parent/previous vertex for each node, backtrack from target to source',
            'Store all paths',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'During Dijkstra, when updating dist[v], store parent[v] = u. After algorithm, backtrack: path = [target], while path[-1] != source: path.append(parent[path[-1]]). Reverse for source→target.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain Bellman-Ford algorithm. Why can it handle negative weights when Dijkstra cannot?',
          sampleAnswer:
            'Bellman-Ford relaxes all edges V-1 times. For each iteration, for each edge (u,v), if dist[u] + weight(u,v) < dist[v], update dist[v]. After V-1 iterations, all shortest paths found (shortest path has at most V-1 edges). Can handle negative weights because it considers all possible paths systematically, not greedily. Dijkstra assumes once a node is processed, its distance is final. Bellman-Ford reconsiders nodes multiple times, allowing negative edges to improve paths discovered earlier. For example, with negative edge, early iteration might find expensive path, later iteration finds cheaper path through negative edge. The V-1 iterations ensure even longest path (visiting V-1 edges) is discovered. Extra Vth iteration detects negative cycles.',
          keyPoints: [
            'Relax all edges V-1 times',
            'Considers all paths systematically, not greedy',
            'Reconsiders nodes multiple times',
            'Handles negative weights by re-evaluation',
            'V-1 iterations enough for longest path',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe negative cycle detection in Bellman-Ford. Why is this important?',
          sampleAnswer:
            'After V-1 iterations, run one more iteration. If any distance still improves, negative cycle exists. Why important? Negative cycles make shortest path undefined - you can loop infinitely reducing cost. For example, A→B cost 5, B→C cost -10, C→A cost 2. Loop A→B→C→A has cost 5-10+2=-3. Each loop reduces cost by 3, making shortest path -infinity. Applications: currency arbitrage (negative cycle means profit), detecting inconsistent constraints, game balancing. Detection algorithm: if Vth iteration still updates distances, trace back the updated node to find cycle. Some variations mark all nodes reachable from negative cycle as having distance -infinity. Critical for correctness - reporting finite distance when none exists is wrong.',
          keyPoints: [
            'Run Vth iteration, if updates exist → negative cycle',
            'Negative cycle makes shortest path undefined (-infinity)',
            'Example: arbitrage, inconsistent constraints',
            'Trace updated node to find actual cycle',
            'Critical: prevent reporting incorrect finite distances',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare Bellman-Ford O(VE) vs Dijkstra O((V+E) log V). When is each preferred despite complexity difference?',
          sampleAnswer:
            'Bellman-Ford is O(VE), slower than Dijkstra O((V+E) log V). For dense graph with V=1000, E=500K: Bellman-Ford is 500M operations, Dijkstra is 5M. Bellman-Ford preferred when: negative weights present (Dijkstra fails), need negative cycle detection, graph is small or sparse with few edges, simpler to implement (no heap needed), distributed systems (edge-based relaxation parallelizes well). Dijkstra preferred when: all weights non-negative (always use if possible), performance critical, large dense graphs. In practice: use Dijkstra if guaranteed non-negative (GPS, network delays), use Bellman-Ford only when negative weights necessary (currency exchange, potential functions). Never use Bellman-Ford if Dijkstra works - speed difference is significant.',
          keyPoints: [
            'Bellman-Ford: O(VE), Dijkstra: O((V+E) log V)',
            'Bellman-Ford when: negative weights, cycle detection',
            'Dijkstra when: non-negative, performance matters',
            'In practice: always prefer Dijkstra if possible',
            'Speed difference significant for large graphs',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does Bellman-Ford handle that Dijkstra cannot?',
          options: [
            'Large graphs',
            'Negative edge weights and negative cycle detection',
            'Directed graphs',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Bellman-Ford handles negative weights and detects negative cycles. Relaxes all edges V-1 times. Slower O(VE) than Dijkstra O(E log V), but more versatile.',
        },
        {
          id: 'mc2',
          question: 'How does Bellman-Ford detect negative cycles?',
          options: [
            'Cannot detect',
            'After V-1 iterations, if any edge can still relax, negative cycle exists',
            'Count edges',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Shortest path has at most V-1 edges. After V-1 iterations, distances should be final. If Vth iteration still improves distance, cycle with negative weight exists.',
        },
        {
          id: 'mc3',
          question: 'What is the time complexity of Bellman-Ford?',
          options: [
            'O(E log V)',
            'O(VE) - V-1 iterations, each relaxes E edges',
            'O(V²)',
            'O(V³)',
          ],
          correctAnswer: 1,
          explanation:
            'Bellman-Ford: V-1 iterations of relaxing all E edges = O(VE). Slower than Dijkstra O(E log V) but handles negative weights. Dense graph: O(V³).',
        },
        {
          id: 'mc4',
          question: 'Why does Bellman-Ford require V-1 iterations?',
          options: [
            'Random',
            'Shortest simple path has at most V-1 edges - each iteration extends path by 1 edge',
            'Optimization',
            'Historical',
          ],
          correctAnswer: 1,
          explanation:
            'Shortest simple path (no repeated vertices) has at most V-1 edges. Iteration i finds shortest paths with ≤i edges. After V-1 iterations, all shortest paths found.',
        },
        {
          id: 'mc5',
          question: 'When should you use Bellman-Ford?',
          options: [
            'Always',
            'When graph has negative weights, need negative cycle detection, or distributed/simple implementation',
            'Never',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Use Bellman-Ford when: 1) Negative weights present, 2) Need to detect negative cycles, 3) Distributed systems (simple to parallelize), 4) Small graphs where O(VE) acceptable. Otherwise Dijkstra faster.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain Floyd-Warshall algorithm. How does it compute all-pairs shortest paths?',
          sampleAnswer:
            'Floyd-Warshall uses dynamic programming with 3 nested loops. DP state: dist[i][j][k] = shortest path from i to j using only vertices 0..k as intermediates. Recurrence: dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1]). Either use k as intermediate or not. Base case: dist[i][j][0] = weight(i,j) if edge exists, infinity otherwise. Can optimize to 2D by updating in-place: for each k, for each i, for each j: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]). After considering all vertices as intermediates, dist[i][j] is shortest path from i to j. Works for negative weights (detects negative cycles if dist[i][i] < 0). Simple nested loops, no heap needed.',
          keyPoints: [
            'DP: shortest i→j using vertices 0..k',
            'Three nested loops: k, i, j',
            'Consider using k as intermediate or not',
            'Updates in-place: dist[i][j] = min(current, via k)',
            'O(V³) time, handles negative weights',
          ],
        },
        {
          id: 'q2',
          question:
            'When should you use Floyd-Warshall vs running Dijkstra V times?',
          sampleAnswer:
            'Floyd-Warshall is O(V³), Dijkstra V times is O(V × (V+E) log V) = O(V²E log V) for general, O(V² log V) for sparse. For dense graphs (E ≈ V²): Dijkstra V times is O(V³ log V), Floyd-Warshall is O(V³) - Floyd wins. For sparse graphs (E ≈ V): Dijkstra V times is O(V² log V), Floyd-Warshall is O(V³) - Dijkstra wins. Use Floyd-Warshall when: need full all-pairs matrix, graph is dense, simplicity matters (easy to code), negative weights but no negative cycles. Use Dijkstra V times when: sparse graph, only need some pairs, V is large. Example: 1000 vertices, 5000 edges: Floyd is 1B ops, Dijkstra×V is 50M ops. For small complete graphs (V < 500), Floyd-Warshall often preferred for simplicity.',
          keyPoints: [
            'Floyd-Warshall: O(V³), Dijkstra×V: O(V²E log V)',
            'Dense graphs: Floyd-Warshall wins',
            'Sparse graphs: Dijkstra×V wins',
            'Floyd when: dense, simple, negative weights',
            'Dijkstra when: sparse, large V',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe how to reconstruct paths in Floyd-Warshall. Why is it more complex than single-source algorithms?',
          sampleAnswer:
            'Need next array: next[i][j] = next vertex after i on shortest path to j. During relaxation, when updating dist[i][j] via k, set next[i][j] = next[i][k] (go through k). Path reconstruction: start at i, follow next[i][j] to get second vertex, then next[second][j] until reaching j. More complex because: storing all paths needs O(V²) next array vs O(V) parent array for single-source, reconstruction requires following chain vs simple backtracking. Alternative: store full paths O(V³) space but impractical. The next array is space-efficient compromise - O(V²) space, O(V) time per path reconstruction. For complete path matrix, would need O(V³) space to store all V² paths of average length V.',
          keyPoints: [
            'Next array: next[i][j] = next vertex after i to j',
            'Update next when updating distances via k',
            'Reconstruct: follow next pointers from i to j',
            'More complex: O(V²) array vs O(V) for single-source',
            'Alternative full paths: O(V³) space impractical',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does Floyd-Warshall compute?',
          options: [
            'Single-source shortest path',
            'All-pairs shortest paths - shortest between every pair of vertices',
            'MST',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd-Warshall: all-pairs shortest paths in O(V³). Dynamic programming approach. Computes dist[i][j] for every pair. Handles negative weights but not negative cycles.',
        },
        {
          id: 'mc2',
          question: 'What is the time complexity of Floyd-Warshall?',
          options: [
            'O(E log V)',
            'O(V³) - three nested loops over all vertices',
            'O(VE)',
            'O(V²)',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd-Warshall: O(V³) with three nested loops (k, i, j). For each pair (i,j), try all intermediate vertices k. Space O(V²) for distance matrix.',
        },
        {
          id: 'mc3',
          question: 'How does Floyd-Warshall work?',
          options: [
            'BFS from each vertex',
            'DP: try each vertex k as intermediate, update dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])',
            'Greedy',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Floyd-Warshall DP: for each intermediate vertex k, for all pairs (i,j), check if path through k is shorter. dist[i][j] = min(direct, via k). After k loops, all shortest paths found.',
        },
        {
          id: 'mc4',
          question: 'When should you use Floyd-Warshall?',
          options: [
            'Always',
            'Dense graph, need all-pairs, or graph is small (V³ acceptable)',
            'Large sparse graphs',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Use Floyd-Warshall when: 1) Need distances between all pairs, 2) Dense graph (E≈V²), 3) Small V (V³ acceptable), 4) Simple implementation. For sparse: run Dijkstra V times = O(VE log V) faster.',
        },
        {
          id: 'mc5',
          question: 'How do you detect negative cycles in Floyd-Warshall?',
          options: [
            'Cannot detect',
            'After algorithm, if dist[i][i] < 0 for any i, negative cycle exists',
            'Count edges',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'After Floyd-Warshall, check diagonal: if dist[i][i] < 0, vertex i is part of negative cycle. Normal shortest path from vertex to itself is 0. Negative indicates negative cycle.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain Union-Find data structure. How do find and union operations work?',
          sampleAnswer:
            'Union-Find (Disjoint Set Union) tracks connected components. Each element has parent pointer, root of tree represents component. Find(x): follow parent pointers until reaching root (element where parent[x] = x). Union(x, y): find roots of x and y, make one root point to other. Initially each element is own parent (separate components). For example, union(1,2) and union(3,4): creates two trees with roots 1 and 3. Then union(1,3) merges into one tree. Operations track which elements are connected without storing all edges. Used for: dynamic connectivity, Kruskal MST, network connectivity, image segmentation. Basic operations are O(n) worst case (linear tree), but optimizations make them nearly O(1).',
          keyPoints: [
            'Tracks connected components with parent pointers',
            'Find: follow parents to root',
            'Union: connect roots of two components',
            'Initially: each element separate',
            'Used: connectivity, Kruskal, segmentation',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe path compression and union by rank optimizations. Why do they make operations nearly constant time?',
          sampleAnswer:
            'Path compression: during find(x), make all nodes on path point directly to root. Flattens tree. Union by rank: track tree depth (rank), attach shorter tree under taller. Together achieve O(α(n)) per operation where α is inverse Ackermann - grows so slowly it is effectively O(1) (α(n) < 5 for any practical n). Without optimizations, trees can become linear chains O(n) tall. With optimizations, trees stay very flat. For example, path compression: find(5) in chain 1→2→3→4→5 flattens to all pointing to 1. Union by rank: prevents attaching big tree under small (which increases height). Mathematical proof shows amortized O(α(n)) - nearly constant for all practical purposes. This makes Union-Find incredibly efficient.',
          keyPoints: [
            'Path compression: flatten during find',
            'Union by rank: shorter under taller',
            'Achieves O(α(n)) ≈ O(1) practical',
            'Without: trees can be O(n) tall',
            'With: trees stay very flat (height < 5)',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through using Union-Find for Kruskal MST algorithm. Why is Union-Find perfect for this?',
          sampleAnswer:
            'Kruskal MST: sort edges by weight, iterate edges in order, add edge if it connects different components (no cycle). Union-Find checks connectivity efficiently. Algorithm: initialize Union-Find with n vertices (separate components), sort edges O(E log E), for each edge (u,v): if find(u) != find(v) (different components), add edge to MST and union(u,v). Why perfect? Need to: 1) check if edge creates cycle (different components?), 2) merge components. Union-Find does both in O(α(n)) ≈ O(1). Alternative: DFS to check cycle is O(V+E) per edge, total O(E(V+E)) - too slow. Union-Find makes Kruskal O(E log E) dominated by sorting. The "union" operation naturally represents merging components as we build MST.',
          keyPoints: [
            'Kruskal: sort edges, add if no cycle',
            'Union-Find checks: different components?',
            'Find: check connectivity, Union: merge components',
            'O(α(n)) per edge vs O(V+E) DFS',
            'Total: O(E log E) for sorting, Union-Find is fast',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is Union-Find used for?',
          options: [
            'Sorting',
            'Disjoint set operations - track connected components, detect cycles',
            'Shortest path',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            "Union-Find (Disjoint Set Union): efficiently track which vertices are in same connected component. Operations: find(x) finds root, union(x,y) merges sets. Used in Kruskal's MST, cycle detection.",
        },
        {
          id: 'mc2',
          question: 'What optimizations make Union-Find efficient?',
          options: [
            'None',
            'Path compression + union by rank - nearly O(1) amortized per operation',
            'Sorting',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Optimizations: 1) Path compression: during find, point all nodes to root (flattens tree), 2) Union by rank: attach smaller tree to larger (keeps tree shallow). Together: O(α(N)) ≈ O(1) amortized.',
        },
        {
          id: 'mc3',
          question: 'How does path compression work?',
          options: [
            'Deletes paths',
            'During find(x), make all nodes on path point directly to root',
            'Random',
            'Sorts nodes',
          ],
          correctAnswer: 1,
          explanation:
            'Path compression: when finding root of x, update parent of all nodes on path to point directly to root. Flattens tree structure. Future finds on same path become O(1).',
        },
        {
          id: 'mc4',
          question: 'What is union by rank?',
          options: [
            'Random union',
            'Always attach shorter tree under taller tree to keep balanced',
            'Sort first',
            'Union by size',
          ],
          correctAnswer: 1,
          explanation:
            'Union by rank: track tree height (rank). When merging, make root of shorter tree point to root of taller. Keeps tree shallow. Alternative: union by size (attach smaller to larger).',
        },
        {
          id: 'mc5',
          question:
            'What is the amortized time complexity of Union-Find with optimizations?',
          options: [
            'O(log N)',
            'O(α(N)) where α is inverse Ackermann - effectively O(1)',
            'O(N)',
            'O(1) exact',
          ],
          correctAnswer: 1,
          explanation:
            'With path compression + union by rank: O(α(N)) amortized per operation, where α is inverse Ackermann function. α(N) ≤ 5 for any practical N. Effectively constant time.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare Kruskal vs Prim for MST. When would you choose each?',
          sampleAnswer:
            'Kruskal: sort edges O(E log E), use Union-Find to add edges avoiding cycles. Total O(E log E). Prim: grow tree from vertex, use heap to pick minimum edge to unexplored vertex. Total O(E log V). Choose Kruskal when: sparse graph (E << V²), edges already sorted or can sort efficiently, easy to parallelize (edge-based), want simple implementation with Union-Find. Choose Prim when: dense graph (E ≈ V²), need to start from specific vertex, edges stored as adjacency list. For example, sparse graph 1000 vertices, 5000 edges: Kruskal O(5000 log 5000) ≈ 60K, Prim O(5000 log 1000) ≈ 50K - similar. Dense graph 1000 vertices, 500K edges: Kruskal O(500K log 500K) ≈ 9M, Prim O(500K log 1000) ≈ 5M - Prim better. In practice, both work well.',
          keyPoints: [
            'Kruskal: O(E log E) edge-based, sparse graphs',
            'Prim: O(E log V) vertex-based, dense graphs',
            'Kruskal: sort edges, union-find',
            'Prim: grow tree, heap for min edge',
            'Choice depends on graph density',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain why MST algorithms are greedy and why the greedy choice is optimal.',
          sampleAnswer:
            'MST greedy choice: always pick minimum weight edge that connects components (Kruskal) or minimum edge to unexplored vertex (Prim). Proof by cut property: for any cut of graph, minimum weight edge crossing cut is in some MST. Both algorithms respect this: Kruskal picks minimum globally avoiding cycles (implicit cut between connected components), Prim picks minimum locally (explicit cut between tree and non-tree vertices). The greedy choice is safe - adding minimum edge cannot be wrong because cut property guarantees it belongs to some MST. This is unlike general optimization where local optimum might not be global. MST has special structure making greedy optimal. Proof by contradiction: if greedy edge not in MST, can swap it for heavier edge, reducing total weight - contradiction.',
          keyPoints: [
            'Greedy: always pick minimum available edge',
            'Cut property: min edge crossing cut is in MST',
            'Kruskal respects cut between components',
            'Prim respects cut between tree and non-tree',
            'Proof: swapping greedy edge reduces weight',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe how MST is used for clustering. What is the connection?',
          sampleAnswer:
            'MST for clustering: build MST, remove k-1 heaviest edges to get k clusters. Intuition: MST connects all points with minimum total distance. Heavy edges represent large gaps between clusters. Removing heavy edges separates natural clusters. For example, customer segmentation: points are customers (features as coordinates), distance is dissimilarity. MST finds minimum connections. Remove 2 heaviest edges to get 3 clusters. Why works? MST captures global structure efficiently. Removing edge disconnects into components - natural clusters. This is single-linkage clustering. Alternative: use MST edge weights as threshold (edges > threshold separate clusters). MST gives hierarchical clustering by considering removal of edges in weight order. Simple, efficient O(E log E), but sensitive to outliers (single outlier edge can connect clusters).',
          keyPoints: [
            'Build MST, remove k-1 heaviest edges → k clusters',
            'Heavy edges = gaps between clusters',
            'MST captures global structure efficiently',
            'Single-linkage clustering',
            'Limitation: sensitive to outlier edges',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a Minimum Spanning Tree?',
          options: [
            'Shortest path tree',
            'Tree connecting all vertices with minimum total edge weight',
            'Largest tree',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'MST: subset of edges forming tree (connected, acyclic) that includes all vertices with minimum sum of edge weights. Unique for distinct weights. Used in network design.',
        },
        {
          id: 'mc2',
          question: "How does Kruskal's algorithm work?",
          options: [
            'BFS',
            "Sort edges by weight, greedily add edge if doesn't create cycle (use Union-Find)",
            'Random',
            'DFS',
          ],
          correctAnswer: 1,
          explanation:
            'Kruskal: 1) Sort all edges by weight O(E log E), 2) For each edge, if vertices in different components (Union-Find), add edge and union. O(E log E) for sort + O(E α(V)) for unions.',
        },
        {
          id: 'mc3',
          question: "How does Prim's algorithm work?",
          options: [
            'Sorting edges',
            'Start from vertex, grow tree by adding minimum weight edge connecting tree to new vertex (use priority queue)',
            'Random',
            'Union-Find',
          ],
          correctAnswer: 1,
          explanation:
            'Prim: 1) Start with any vertex, 2) Repeatedly add minimum weight edge connecting tree to new vertex (use min-heap), 3) Continue until all vertices included. O(E log V) with heap.',
        },
        {
          id: 'mc4',
          question: 'When should you use Kruskal vs Prim?',
          options: [
            'Same',
            'Kruskal: sparse graphs (edge-focused). Prim: dense graphs (vertex-focused)',
            'Random',
            'Always Prim',
          ],
          correctAnswer: 1,
          explanation:
            'Kruskal O(E log E): better for sparse graphs (fewer edges to sort). Uses Union-Find. Prim O(E log V): better for dense graphs (many edges per vertex). Uses priority queue. Both correct.',
        },
        {
          id: 'mc5',
          question: 'What applications use MST?',
          options: [
            'None',
            'Network design (min cable), clustering, approximation algorithms (TSP)',
            'Only theoretical',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'MST applications: 1) Network design (minimize cable/pipe length), 2) Clustering (cut MST edges), 3) TSP approximation (MST gives lower bound), 4) Image segmentation, 5) Handwriting recognition.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare all shortest path algorithms by complexity and use case. Which should you use when?',
          sampleAnswer:
            'BFS: O(V+E), unweighted graphs. Dijkstra: O((V+E) log V), weighted non-negative. Bellman-Ford: O(VE), weighted with negatives. Floyd-Warshall: O(V³), all-pairs any weights. Use BFS when: unweighted or unit weights (simplest, fastest). Use Dijkstra when: weighted non-negative, single-source (almost always use if possible). Use Bellman-Ford when: negative weights, need cycle detection, distributed system. Use Floyd-Warshall when: all-pairs needed, dense graph, negative weights OK. For example: social network distances (unweighted) → BFS. GPS navigation (positive weights) → Dijkstra. Currency exchange (negative edges) → Bellman-Ford. City distance matrix (all pairs, moderate size) → Floyd-Warshall. The choice hierarchy: try simpler first (BFS, Dijkstra), only use complex when necessary.',
          keyPoints: [
            'BFS: O(V+E) unweighted',
            'Dijkstra: O((V+E) log V) weighted non-negative',
            'Bellman-Ford: O(VE) negative weights',
            'Floyd-Warshall: O(V³) all-pairs',
            'Hierarchy: simpler first, complex when necessary',
          ],
        },
        {
          id: 'q2',
          question:
            'When would you need to implement Dijkstra from scratch vs using library? What are the tradeoffs?',
          sampleAnswer:
            'Implement from scratch when: custom graph representation, need to modify algorithm (bidirectional search, A* heuristic), educational purpose, performance-critical with specific optimizations, no suitable library available. Use library when: standard implementation suffices, development speed matters, well-tested code preferred, graph format matches library. Tradeoffs of custom: full control and optimization but time-consuming and bug-prone. Library: fast development but less flexibility. For interviews: implement from scratch (test understanding). For production: prefer library (NetworkX Python, Boost C++, JGraphT Java) unless special needs. Custom optimization examples: persistent data structures for online queries, approximation for huge graphs, parallel implementation for distributed systems. Modern libraries are well-optimized, prefer them unless compelling reason.',
          keyPoints: [
            'Scratch when: custom needs, modifications, learning',
            'Library when: standard case, speed, reliability',
            'Interviews: implement to show understanding',
            'Production: prefer library unless special needs',
            'Libraries: NetworkX, Boost, JGraphT',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe space-time tradeoffs in graph algorithms. When would you optimize for space vs time?',
          sampleAnswer:
            'Time-space tradeoffs: precompute all-pairs O(V²) space vs compute on-demand O(1) space. Store paths O(V³) vs reconstruct O(V²). Implicit graph representation (generate neighbors) vs explicit adjacency list. Optimize for time when: frequent queries, space available, performance critical (real-time systems). Optimize for space when: huge graphs (billions of edges), limited memory, infrequent queries. For example, Google Maps: precomputes many routes O(V²) space for fast query. Mobile app: compute on-demand to save phone memory. Streaming graphs: process edges online, no full storage. Modern trend: compress graphs, external memory algorithms, approximate answers. The choice depends on system constraints: embedded systems (tight memory), data centers (optimize time), mobile (balance both).',
          keyPoints: [
            'Precompute vs on-demand computation',
            'Store paths vs reconstruct',
            'Time when: frequent queries, space available',
            'Space when: huge graphs, limited memory',
            'Modern: compression, streaming, approximation',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'How do shortest path algorithms compare?',
          options: [
            'All same',
            'BFS: unweighted O(V+E). Dijkstra: non-negative O(E log V). Bellman-Ford: negative O(VE). Floyd-Warshall: all-pairs O(V³)',
            'Random',
            'All O(N²)',
          ],
          correctAnswer: 1,
          explanation:
            'BFS fastest for unweighted. Dijkstra fastest for non-negative weights. Bellman-Ford handles negatives. Floyd-Warshall for all-pairs. Choose based on graph properties and requirements.',
        },
        {
          id: 'mc2',
          question: 'When should you use BFS vs Dijkstra?',
          options: [
            'Always Dijkstra',
            'BFS: unweighted graphs O(V+E). Dijkstra: weighted non-negative O(E log V)',
            'Random',
            'Same thing',
          ],
          correctAnswer: 1,
          explanation:
            'BFS is special case of Dijkstra for unweighted (all weights = 1). BFS O(V+E) simpler and faster. Dijkstra O(E log V) generalizes to weighted. Use simplest algorithm that works.',
        },
        {
          id: 'mc3',
          question: 'How do MST algorithms compare?',
          options: [
            'Same algorithm',
            'Kruskal: edge-based O(E log E), good for sparse. Prim: vertex-based O(E log V), good for dense',
            'Random',
            'Both O(V³)',
          ],
          correctAnswer: 1,
          explanation:
            'Kruskal: sort edges, add if no cycle. O(E log E). Prim: grow tree, add min edge. O(E log V). Kruskal better for sparse (few edges), Prim for dense (many edges).',
        },
        {
          id: 'mc4',
          question: 'What is the space complexity trade-off?',
          options: [
            'All same',
            'BFS/Dijkstra/Bellman-Ford: O(V). Floyd-Warshall: O(V²) for all-pairs matrix',
            'All O(1)',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Single-source algorithms (BFS, Dijkstra, Bellman-Ford): O(V) for distances. All-pairs (Floyd-Warshall): O(V²) matrix. Trade-off: space vs complete information.',
        },
        {
          id: 'mc5',
          question: 'How do you choose the right algorithm?',
          options: [
            'Random',
            'Consider: weighted? negative? all-pairs? sparse/dense? Then match to algorithm constraints and complexity',
            'Always use Dijkstra',
            'No method',
          ],
          correctAnswer: 1,
          explanation:
            'Decision tree: 1) Unweighted → BFS, 2) Non-negative weighted → Dijkstra, 3) Negative weights → Bellman-Ford, 4) All-pairs → Floyd-Warshall, 5) MST → Kruskal (sparse) or Prim (dense).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize an advanced graph problem? What keywords signal these algorithms?',
          sampleAnswer:
            'Keywords: "shortest path", "minimum spanning tree", "weighted graph", "negative weights", "all pairs", "connectivity", "network design". Patterns: optimization on weighted graph, need distances between many pairs, building optimal network, detecting arbitrage/cycles. For example, "find shortest route with tolls" → weighted shortest path (Dijkstra). "Connect all cities with minimum cable" → MST. "Currency exchange profitability" → Bellman-Ford with cycle detection. "Distance table between all airports" → Floyd-Warshall. "Check if network components connected" → Union-Find. The signals: weights mentioned (not just connectivity), optimization problem (minimum/shortest), all-pairs requirement, special graph properties. Advanced graphs solve optimization, not just traversal.',
          keyPoints: [
            'Keywords: shortest, minimum, weighted, all pairs',
            'Optimization on weighted graphs',
            'Examples: routes with costs, network design',
            'vs Basic: optimization not just traversal',
            'Weights + optimization → advanced algorithms',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your advanced graph interview approach from recognition to implementation.',
          sampleAnswer:
            'First, recognize algorithm type from problem (weighted → shortest path, connect all → MST). Second, clarify: negative weights? Single-source or all-pairs? Need path or just distance? Third, choose algorithm: non-negative → Dijkstra, negative → Bellman-Ford, all-pairs small graph → Floyd-Warshall, connect all → MST (Kruskal/Prim). Fourth, state complexity and why that algorithm. Fifth, draw small example showing algorithm steps. Sixth, discuss data structures: heap for Dijkstra, Union-Find for Kruskal. Seventh, code clearly with proper initialization and relaxation logic. Eighth, test with edge cases: disconnected graph, negative cycles, self-loops. Finally, discuss optimizations: bidirectional search, A* heuristic. This shows: recognition, algorithm selection, implementation, optimization knowledge.',
          keyPoints: [
            'Recognize type: weights, single/all pairs, MST',
            'Clarify: negative weights, paths vs distances',
            'Choose algorithm with justification',
            'State complexity, draw example',
            'Code with proper structures, test edges',
            'Discuss optimizations',
          ],
        },
        {
          id: 'q3',
          question:
            'What are common mistakes in advanced graph problems and how do you avoid them?',
          sampleAnswer:
            'First: using Dijkstra with negative weights (fails silently). Second: forgetting to check negative cycles in Bellman-Ford. Third: initializing distances incorrectly (source != 0 or others != infinity). Fourth: heap contains duplicate entries for same vertex in Dijkstra (inefficient). Fifth: Union-Find without optimizations (O(n) instead of O(α(n))). Sixth: Floyd-Warshall loop order wrong (k must be outermost). Seventh: off-by-one in iterations (V-1 for Bellman-Ford). My strategy: verify weight constraints, always test with negative weights if allowed, initialize carefully, use decrease-key or visited set for Dijkstra, always implement Union-Find with optimizations, remember Floyd-Warshall loop order (alphabetical: k,i,j). Most mistakes from wrong algorithm choice or incorrect initialization.',
          keyPoints: [
            'Wrong algorithm for weight type',
            'Missing negative cycle check',
            'Incorrect initialization',
            'Heap duplicates in Dijkstra',
            'Union-Find without optimizations',
            'Test: negative weights, initialization, loop order',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What keywords signal an advanced graph problem?',
          options: [
            'Traversal',
            'Shortest path, minimum/maximum, optimal, flow, spanning tree, connectivity',
            'Random',
            'Simple graph',
          ],
          correctAnswer: 1,
          explanation:
            'Advanced graph keywords: "shortest path", "minimum cost", "spanning tree", "maximum flow", "strongly connected", "bridges/articulation". Suggests optimization algorithms beyond BFS/DFS.',
        },
        {
          id: 'mc2',
          question: 'How do you approach a shortest path interview question?',
          options: [
            'Always Dijkstra',
            'Clarify: single-source or all-pairs? weights? negative? Then choose: BFS/Dijkstra/Bellman-Ford/Floyd-Warshall',
            'Random',
            'BFS only',
          ],
          correctAnswer: 1,
          explanation:
            'Approach: 1) Clarify single-source vs all-pairs, 2) Check weights (none/positive/negative), 3) Choose algorithm: unweighted→BFS, non-negative→Dijkstra, negative→Bellman-Ford, all-pairs→Floyd-Warshall.',
        },
        {
          id: 'mc3',
          question: 'What should you clarify in a graph interview?',
          options: [
            'Nothing',
            'Directed/undirected? Weighted? Connected? Constraints on V,E? Dense/sparse?',
            'Random',
            'Language only',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: 1) Directed or undirected, 2) Weighted (and range), 3) Connected or multiple components, 4) V,E size (affects algorithm choice), 5) Dense (E≈V²) or sparse (E≈V).',
        },
        {
          id: 'mc4',
          question: 'What is a common graph algorithm mistake?',
          options: [
            'Using graphs',
            'Wrong algorithm (Dijkstra with negative weights), not handling disconnected components, off-by-one in adjacency',
            'Good naming',
            'Comments',
          ],
          correctAnswer: 1,
          explanation:
            'Common mistakes: 1) Dijkstra with negative weights (use Bellman-Ford), 2) Forgetting disconnected components (loop all vertices), 3) Wrong complexity analysis, 4) Not initializing distances correctly.',
        },
        {
          id: 'mc5',
          question: 'How should you communicate your graph solution?',
          options: [
            'Just code',
            'Explain algorithm choice (why Dijkstra vs BFS?), complexity, walk through example, discuss trade-offs',
            'No explanation',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Communication: 1) Why this algorithm (graph properties→algorithm), 2) How it works briefly, 3) Walk through small example, 4) Time O(?) and space O(?), 5) Trade-offs (Kruskal vs Prim).',
        },
      ],
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
