/**
 * Graph Algorithm Selection Guide Section
 */

export const algorithmselectionSection = {
  id: 'algorithm-selection',
  title: 'Graph Algorithm Selection Guide',
  content: `## 🎯 Which Graph Algorithm Should I Use?

\`\`\`
START: What is your goal?

├─ TRAVERSAL: Visit all nodes/edges
│  ├─ Need level-by-level order? → BFS
│  ├─ Need to explore deeply first? → DFS
│  ├─ Find connected components? → DFS or BFS (both work)
│  └─ Topological ordering (DAG)? → DFS + Stack or Kahn\'s Algorithm (BFS)
│
├─ SHORTEST PATH: Find minimum distance
│  ├─ Unweighted graph? → BFS
│  ├─ Weighted with non-negative weights? → Dijkstra's Algorithm
│  ├─ Weighted with negative weights? → Bellman-Ford
│  ├─ All pairs shortest paths? → Floyd-Warshall
│  └─ Path with heuristic (grid, known goal)? → A* Search
│
├─ CONNECTIVITY: Check if nodes are connected
│  ├─ Static queries? → Single DFS/BFS
│  ├─ Dynamic queries (many queries)? → Union-Find (Disjoint Set)
│  └─ Strongly connected components (directed)? → Tarjan's or Kosaraju\'s │
├─ CYCLE DETECTION: Find if cycle exists
│  ├─ Undirected graph? → DFS with parent tracking
│  ├─ Directed graph? → DFS with recursion stack
│  └─ Find all cycles? → DFS with backtracking
│
├─ MINIMUM SPANNING TREE: Connect all nodes with minimum cost
│  ├─ Need edge-by-edge building? → Kruskal's (with Union-Find)
│  ├─ Growing from a vertex? → Prim's Algorithm
│  └─ Already have Union-Find structure? → Kruskal\'s │
├─ NETWORK FLOW: Maximum flow from source to sink
│  └─ Use Ford-Fulkerson or Edmonds-Karp
│
└─ BIPARTITE CHECK: Can graph be 2-colored?
   └─ BFS or DFS with coloring
\`\`\`

---

## 📊 Graph Algorithm Comparison Table

| Algorithm | Use Case | Graph Type | Time | Space | Key Insight |
|-----------|----------|------------|------|-------|-------------|
| **BFS** | Shortest path (unweighted), level-order | Any | O(V+E) | O(V) | Queue, explores layer by layer |
| **DFS** | Cycle detection, topological sort, components | Any | O(V+E) | O(V) | Stack/recursion, explores deeply |
| **Dijkstra** | Shortest path (weighted, non-negative) | Weighted | O((V+E) log V) | O(V) | Priority queue, greedy approach |
| **Bellman-Ford** | Shortest path with negative weights | Weighted | O(V·E) | O(V) | Relax all edges V-1 times |
| **Floyd-Warshall** | All pairs shortest paths | Weighted | O(V³) | O(V²) | DP, tries all intermediate nodes |
| **A*** | Shortest path with heuristic | Weighted | O(E) best case | O(V) | Like Dijkstra + heuristic function |
| **Prim's** | Minimum spanning tree | Weighted, undirected | O((V+E) log V) | O(V) | Greedy, grow tree from vertex |
| **Kruskal's** | Minimum spanning tree | Weighted, undirected | O(E log E) | O(V) | Sort edges, use Union-Find |
| **Union-Find** | Dynamic connectivity | Any | O(α(n)) per op | O(V) | Near-constant amortized time |
| **Tarjan's** | Strongly connected components | Directed | O(V+E) | O(V) | One-pass DFS with low-link |
| **Kahn's** | Topological sort | DAG | O(V+E) | O(V) | BFS with in-degree tracking |

*V = vertices, E = edges, α(n) = inverse Ackermann function (practically constant)*

---

## 🔍 DFS vs BFS: When to Use Each

### Use BFS When:
✅ **Finding shortest path in unweighted graph**
\`\`\`python
# BFS guarantees shortest path in unweighted graphs
def shortestPath (graph, start, end):
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist  # Guaranteed shortest!
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append((neighbor, dist + 1))
    return -1
\`\`\`

✅ **Need to explore layer by layer**
- Level-order traversal
- Finding all nodes at distance k
- Spreading/infection problems

✅ **Graph is very deep or infinite**
- BFS uses queue, won't run into stack overflow

### Use DFS When:
✅ **Detecting cycles**
\`\`\`python
# DFS is natural for cycle detection
def hasCycle (graph):
    visited = set()
    rec_stack = set()  # Recursion stack for directed graph
    
    def dfs (node):
        visited.add (node)
        rec_stack.add (node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs (neighbor):
                    return True
            elif neighbor in rec_stack:  # Back edge = cycle
                return True
        
        rec_stack.remove (node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs (node):
                return True
    return False
\`\`\`

✅ **Topological sorting**
- DFS naturally gives reverse topological order

✅ **Path finding (not shortest)**
- Finding ANY path
- Backtracking problems
- Tree traversals (inorder, preorder, postorder)

✅ **Memory constrained**
- DFS uses O(h) space (height), BFS uses O(w) space (width)
- For wide graphs, DFS is more memory efficient

---

## 🚀 Dijkstra vs Bellman-Ford vs Floyd-Warshall

### Dijkstra\'s Algorithm
**When to use:**
- Single source shortest path
- **Non-negative edge weights only**
- Need best performance

**Time:** O((V + E) log V) with priority queue

\`\`\`python
import heapq

def dijkstra (graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]  # (distance, node)
    
    while pq:
        curr_dist, u = heapq.heappop (pq)
        
        if curr_dist > dist[u]:
            continue  # Already found better path
        
        for v, weight in graph[u]:
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush (pq, (new_dist, v))
    
    return dist
\`\`\`

**❌ Fails with negative weights!**

### Bellman-Ford Algorithm
**When to use:**
- Single source shortest path
- **Graph has negative edge weights**
- Need to detect negative cycles

**Time:** O(V · E) - slower than Dijkstra

\`\`\`python
def bellmanFord (graph, V, start):
    dist = [float('inf')] * V
    dist[start] = 0
    
    # Relax all edges V-1 times
    for _ in range(V - 1):
        for u, v, weight in edges:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
    
    # Check for negative cycles
    for u, v, weight in edges:
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            return None  # Negative cycle detected!
    
    return dist
\`\`\`

### Floyd-Warshall Algorithm
**When to use:**
- **All pairs** shortest paths
- Dense graph (many edges)
- Small graph (V ≤ 400)

**Time:** O(V³) - only feasible for small graphs

\`\`\`python
def floydWarshall (graph, V):
    # dist[i][j] = shortest distance from i to j
    dist = [[float('inf')] * V for _ in range(V)]
    
    # Initialize
    for i in range(V):
        dist[i][i] = 0
    
    for u, v, weight in edges:
        dist[u][v] = weight
    
    # Try all intermediate nodes k
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min (dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
\`\`\`

**Decision Matrix:**

| Need | Graph Size | Negative Weights? | Choose |
|------|-----------|-------------------|--------|
| Single source | Any | No | **Dijkstra** |
| Single source | Any | Yes | **Bellman-Ford** |
| All pairs | Small (V ≤ 400) | Either | **Floyd-Warshall** |
| All pairs | Large | No | Run **Dijkstra** V times |

---

## 🧭 A* Search: When and How

**A* = Dijkstra + Heuristic**

**When to use:**
✅ You know the goal location (target)
✅ You can estimate distance to goal (heuristic)
✅ Working with grids/maps
✅ Want faster than Dijkstra

**Key insight:** Prioritize nodes that seem closer to goal.

\`\`\`python
import heapq

def astar (graph, start, goal, heuristic):
    # heuristic (node) = estimated distance from node to goal
    # For grid: Manhattan distance or Euclidean distance
    
    g_score = {start: 0}  # Actual distance from start
    f_score = {start: heuristic (start)}  # g + h
    pq = [(f_score[start], start)]
    
    while pq:
        _, current = heapq.heappop (pq)
        
        if current == goal:
            return g_score[current]  # Found shortest path!
        
        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic (neighbor)
                heapq.heappush (pq, (f_score[neighbor], neighbor))
    
    return -1  # No path

# Example heuristic for grid (Manhattan distance)
def manhattan (node, goal):
    return abs (node[0] - goal[0]) + abs (node[1] - goal[1])
\`\`\`

**Heuristic Requirements:**
- **Admissible:** Never overestimate actual distance (h (n) ≤ actual)
- **Consistent:** h (n) ≤ cost (n, n') + h (n') for all neighbors n'

**Common heuristics:**
- **Grid (4-directional):** Manhattan distance
- **Grid (8-directional):** Chebyshev distance
- **Grid (any direction):** Euclidean distance
- **Graph:** Precomputed landmarks

---

## 🌲 Prim\'s vs Kruskal's for MST

Both find Minimum Spanning Tree, but different approaches:

### Prim's Algorithm
**Grows tree from a single vertex**

\`\`\`python
import heapq

def prim (graph, start):
    mst = []
    visited = {start}
    edges = [(weight, start, neighbor) for neighbor, weight in graph[start]]
    heapq.heapify (edges)
    
    while edges:
        weight, u, v = heapq.heappop (edges)
        if v in visited:
            continue
        
        visited.add (v)
        mst.append((u, v, weight))
        
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush (edges, (w, v, neighbor))
    
    return mst
\`\`\`

**Best when:** Dense graph, starting from specific vertex

### Kruskal\'s Algorithm
**Sorts edges and adds smallest that don't create cycles**

\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list (range (n))
        self.rank = [0] * n
    
    def find (self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find (self.parent[x])
        return self.parent[x]
    
    def union (self, x, y):
        px, py = self.find (x), self.find (y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal (edges, V):
    edges.sort()  # Sort by weight
    uf = UnionFind(V)
    mst = []
    
    for weight, u, v in edges:
        if uf.union (u, v):  # No cycle
            mst.append((u, v, weight))
            if len (mst) == V - 1:  # MST complete
                break
    
    return mst
\`\`\`

**Best when:** Sparse graph, already have Union-Find structure

**Comparison:**

| Aspect | Prim's | Kruskal\'s |
|--------|--------|-----------|
| **Approach** | Grow tree from vertex | Add edges by weight |
| **Data Structure** | Priority queue | Union-Find |
| **Time** | O((V+E) log V) | O(E log E) |
| **Best for** | Dense graphs | Sparse graphs |
| **When edges sorted** | No benefit | Can skip sorting: O(E·α(V)) |

---

## 🚨 Common Graph Algorithm Mistakes

### Mistake 1: No Visited Set
**Problem:** Infinite loop in graphs with cycles.

\`\`\`python
# ❌ BAD: Infinite loop possible
def dfs (graph, node):
    for neighbor in graph[node]:
        dfs (graph, neighbor)  # May visit same node again!

# ✅ GOOD: Track visited
def dfs (graph, node, visited):
    if node in visited:
        return
    visited.add (node)
    for neighbor in graph[node]:
        dfs (graph, neighbor, visited)
\`\`\`

### Mistake 2: Wrong Dijkstra with Negative Weights
**Problem:** Dijkstra assumes non-negative weights.

\`\`\`python
# ❌ BAD: Dijkstra with negative weights
edges = [(A, B, -5), (B, C, 3)]  # Has negative weight
dijkstra (graph, A)  # WRONG RESULTS!

# ✅ GOOD: Use Bellman-Ford
bellman_ford (graph, A)  # Handles negative weights
\`\`\`

### Mistake 3: Not Handling Disconnected Graphs
**Problem:** Only visiting one component.

\`\`\`python
# ❌ BAD: Only processes one component
def countNodes (graph):
    visited = set()
    count = 0
    
    def dfs (node):
        if node in visited:
            return 0
        visited.add (node)
        return 1 + sum (dfs (n) for n in graph[node])
    
    return dfs(0)  # Only counts component containing 0!

# ✅ GOOD: Process all components
def countNodes (graph):
    visited = set()
    count = 0
    
    def dfs (node):
        if node in visited:
            return 0
        visited.add (node)
        return 1 + sum (dfs (n) for n in graph[node])
    
    for node in graph:
        if node not in visited:
            count += dfs (node)
    
    return count
\`\`\`

### Mistake 4: Confusing Directed vs Undirected
**Problem:** Treating edges incorrectly.

\`\`\`python
# ❌ BAD: Treating directed as undirected
graph = {0: [1], 1: [2]}  # Directed: 0→1→2
# Assuming you can go 1→0 is WRONG!

# ✅ GOOD: Be explicit
def buildUndirectedGraph (edges):
    graph = defaultdict (list)
    for u, v in edges:
        graph[u].append (v)
        graph[v].append (u)  # Add both directions
    return graph

def buildDirectedGraph (edges):
    graph = defaultdict (list)
    for u, v in edges:
        graph[u].append (v)  # One direction only
    return graph
\`\`\`

### Mistake 5: Stack Overflow in DFS
**Problem:** Deep recursion in large graphs.

\`\`\`python
# ❌ BAD: Recursive DFS on large graph
def dfs (graph, node, visited):
    if node in visited:
        return
    visited.add (node)
    for neighbor in graph[node]:
        dfs (graph, neighbor, visited)  # May overflow!

# ✅ GOOD: Iterative DFS
def dfs (graph, start):
    stack = [start]
    visited = {start}
    
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                stack.append (neighbor)
\`\`\`

---

## 💡 Interview Tips for Graph Problems

### Tip 1: Clarify the Graph Properties
\`\`\`
Always ask:
✅ Directed or undirected?
✅ Weighted or unweighted?
✅ Can it have cycles?
✅ Is it connected?
✅ What's the representation? (adjacency list/matrix/edge list)
✅ Are there self-loops or multiple edges?
\`\`\`

### Tip 2: Start with the Right Algorithm
\`\`\`
Pattern Recognition:
"Shortest path in unweighted" → BFS
"Shortest path in weighted (non-negative)" → Dijkstra
"Detect cycle" → DFS
"All pairs shortest path" → Floyd-Warshall (if small)
"Connected components" → DFS or Union-Find
"Topological sort" → DFS or Kahn's
\`\`\`

### Tip 3: Build the Graph Representation
\`\`\`python
# Template for building adjacency list
from collections import defaultdict

def buildGraph (edges, directed=False):
    graph = defaultdict (list)
    for u, v in edges:
        graph[u].append (v)
        if not directed:
            graph[v].append (u)
    return graph

# For weighted graphs
def buildWeightedGraph (edges, directed=False):
    graph = defaultdict (list)
    for u, v, weight in edges:
        graph[u].append((v, weight))
        if not directed:
            graph[v].append((u, weight))
    return graph
\`\`\`

### Tip 4: Watch Out for Edge Cases
\`\`\`
Common edge cases:
✅ Empty graph
✅ Single node
✅ Disconnected components
✅ Cycles (especially self-loops)
✅ Negative weights (if applicable)
✅ Unreachable nodes
\`\`\`

### Tip 5: Explain Time and Space Complexity
\`\`\`
Standard complexities:
BFS/DFS: O(V + E) time, O(V) space
Dijkstra: O((V + E) log V) time, O(V) space
Bellman-Ford: O(V · E) time, O(V) space
Floyd-Warshall: O(V³) time, O(V²) space

Always express in terms of V (vertices) and E (edges)!
\`\`\``,
};
