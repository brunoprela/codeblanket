import { Module } from '@/lib/types';

export const graphsModule: Module = {
  id: 'graphs',
  title: 'Graphs',
  description:
    'Master graph traversal, pathfinding, and connectivity problems for complex network structures.',
  icon: '🕸️',
  timeComplexity: 'O(V + E) for traversals',
  spaceComplexity: 'O(V) for visited tracking',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to Graphs',
      content: `A **graph** is a data structure consisting of **vertices (nodes)** connected by **edges**. Graphs model relationships and networks: social networks, maps, dependencies, etc.

**Graph Terminology:**

- **Vertex/Node**: A point in the graph
- **Edge**: Connection between two vertices
- **Directed Graph**: Edges have direction (A → B)
- **Undirected Graph**: Edges are bidirectional (A ↔ B)
- **Weighted Graph**: Edges have values/costs
- **Degree**: Number of edges connected to a vertex
- **Path**: Sequence of vertices connected by edges
- **Cycle**: Path that starts and ends at same vertex
- **Connected Graph**: Path exists between any two vertices
- **DAG**: Directed Acyclic Graph (no cycles)

**Graph Representations:**

**1. Adjacency List** (Most Common)
\`\`\`python
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

# Visualized:
#   0---1
#   |   |
#   2---3
\`\`\`

**Pros**: Space efficient O(V + E), fast to iterate neighbors
**Cons**: Slow to check if edge exists

**2. Adjacency Matrix**
\`\`\`python
matrix = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
# matrix[i][j] = 1 if edge exists from i to j
\`\`\`

**Pros**: O(1) edge lookup, simple
**Cons**: O(V²) space, inefficient for sparse graphs

**3. Edge List**
\`\`\`python
edges = [(0,1), (0,2), (1,3), (2,3)]
\`\`\`

**Pros**: Simple, good for algorithms processing all edges
**Cons**: Slow to find neighbors

**When to Use Graphs:**
- Social networks (friends, followers)
- Maps and navigation (cities, roads)
- Dependencies (tasks, packages)
- Networks (computers, websites)
- State machines and game trees`,
    },
    {
      id: 'traversals',
      title: 'Graph Traversals: BFS and DFS',
      content: `**Two main traversal algorithms:**

**1. Breadth-First Search (BFS)**

Explore **level by level**, like ripples in water.

**Uses:**
- **Shortest path** in unweighted graphs
- **Level-order** traversal
- Finding **connected components**

**Algorithm:**
1. Start at source node
2. Visit all neighbors (1 edge away)
3. Then visit their neighbors (2 edges away)
4. Continue until all reachable nodes visited

**Implementation (Queue):**
\`\`\`python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)  # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
\`\`\`

**Visualization:**
\`\`\`
Graph:
    1
   / \\
  2   3
 /     \\
4       5

BFS from 1: 1 → 2,3 → 4,5
Level 0: [1]
Level 1: [2, 3]
Level 2: [4, 5]
\`\`\`

---

**2. Depth-First Search (DFS)**

Explore **as far as possible** before backtracking.

**Uses:**
- **Cycle detection**
- **Topological sort**
- **Finding paths**
- **Connected components**

**Algorithm:**
1. Start at source node
2. Go as deep as possible on one path
3. Backtrack when stuck
4. Try other paths

**Implementation (Recursive):**
\`\`\`python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(node)
    print(node)  # Process node
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    
    return visited
\`\`\`

**Implementation (Iterative with Stack):**
\`\`\`python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        print(node)  # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    
    return visited
\`\`\`

**Visualization:**
\`\`\`
Graph:
    1
   / \\
  2   3
 /     \\
4       5

DFS from 1: 1 → 2 → 4 (backtrack) → 3 → 5
Path: 1 → 2 → 4 → 3 → 5
\`\`\`

---

**BFS vs DFS Comparison:**

| Feature | BFS | DFS |
|---------|-----|-----|
| Data Structure | Queue | Stack/Recursion |
| Order | Level by level | Deep first |
| Shortest Path | ✅ Yes (unweighted) | ❌ No |
| Space | O(W) width | O(H) height |
| Cycle Detection | Harder | Easier |
| Complete | ✅ Yes | ✅ Yes |

**Choosing:**
- **BFS**: Shortest path, level info, closer nodes first
- **DFS**: Memory efficient, cycle detection, exploring all paths`,
      codeExample: `from collections import deque
from typing import Dict, List, Set


def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Breadth-First Search traversal.
    Returns nodes in BFS order.
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def dfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Depth-First Search traversal (recursive).
    Returns nodes in DFS order.
    """
    visited = set()
    result = []
    
    def dfs_helper(node):
        visited.add(node)
        result.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_helper(neighbor)
    
    dfs_helper(start)
    return result


def shortest_path_bfs(graph: Dict[int, List[int]], start: int, end: int) -> List[int]:
    """
    Find shortest path using BFS.
    Returns path from start to end, or empty list if no path.
    """
    if start == end:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                
                if neighbor == end:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    return []  # No path found


def has_cycle_dfs(graph: Dict[int, List[int]]) -> bool:
    """
    Detect cycle in directed graph using DFS.
    """
    visited = set()
    rec_stack = set()  # Recursion stack
    
    def dfs_cycle(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs_cycle(node):
                return True
    
    return False`,
    },
    {
      id: 'patterns',
      title: 'Common Graph Patterns',
      content: `**Pattern 1: Connected Components**

Find all connected groups in undirected graph.

**Approach**: Run BFS/DFS from each unvisited node.

\`\`\`python
def count_components(n, edges):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    visited = set()
    count = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    for node in range(n):
        if node not in visited:
            dfs(node)
            count += 1
    
    return count
\`\`\`

---

**Pattern 2: Cycle Detection**

**Undirected Graph:**
Use DFS with parent tracking. If we visit a node that's already visited and it's not the parent, there's a cycle.

**Directed Graph:**
Use DFS with recursion stack. If we visit a node currently in the stack, there's a cycle.

---

**Pattern 3: Topological Sort**

Order nodes in DAG such that for edge u → v, u comes before v.

**Applications**: Task scheduling, course prerequisites

**Algorithm (Kahn's - BFS):**
\`\`\`python
def topological_sort(graph):
    # Calculate in-degrees
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Start with nodes having in-degree 0
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all nodes processed (no cycle)
    return result if len(result) == len(graph) else []
\`\`\`

---

**Pattern 4: Shortest Path (Weighted)**

**Dijkstra's Algorithm**: Find shortest path in weighted graph with non-negative weights.

\`\`\`python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if curr_dist > distances[node]:
            continue
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
\`\`\`

---

**Pattern 5: Union-Find (Disjoint Set)**

Efficiently track and merge connected components.

**Operations:**
- **Find**: Which component does element belong to?
- **Union**: Merge two components

\`\`\`python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
\`\`\``,
    },
    {
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
    },
    {
      id: 'templates',
      title: 'Code Templates',
      content: `**Template 1: BFS**
\`\`\`python
from collections import deque

def bfs_template(graph, start):
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
\`\`\`

**Template 2: DFS (Recursive)**
\`\`\`python
def dfs_template(graph, node, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(node)
    # Process node
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_template(graph, neighbor, visited)
    
    return visited
\`\`\`

**Template 3: DFS (Iterative)**
\`\`\`python
def dfs_iterative_template(graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    
    return visited
\`\`\`

**Template 4: Shortest Path (BFS)**
\`\`\`python
def shortest_path(graph, start, end):
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                
                if neighbor == end:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    
    return []
\`\`\`

**Template 5: Cycle Detection (Undirected)**
\`\`\`python
def has_cycle_undirected(graph):
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Visited non-parent = cycle
        
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False
\`\`\`

**Template 6: Topological Sort**
\`\`\`python
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([n for n in graph if in_degree[n] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []
\`\`\``,
    },
    {
      id: 'interview',
      title: 'Interview Strategy',
      content: `**Recognition Signals:**

**Use Graph algorithms when you see:**
- "Network", "graph", "tree" (tree is special graph)
- "Nodes and edges", "vertices and connections"
- "Connected", "path", "cycle"
- "Dependencies", "prerequisites"
- "Social network", "friends", "followers"
- "Grid" problems (can model as graph)
- "Islands", "regions" (connected components)

---

**Problem-Solving Steps:**

**Step 1: Clarify Graph Type**
- Directed or undirected?
- Weighted or unweighted?
- Cyclic or acyclic?
- How is it represented (list, matrix, implicit)?

**Step 2: Choose Algorithm**
- **Shortest path (unweighted)?** → BFS
- **Any path?** → DFS or BFS
- **Cycle detection?** → DFS
- **All components?** → DFS/BFS from each unvisited
- **Topological order?** → Kahn's or DFS
- **Connectivity?** → Union-Find

**Step 3: Consider Representation**
- **Given**: Use as-is
- **Build**: Adjacency list (most common)
- **Grid**: Convert to implicit graph

**Step 4: Track State**
- Visited set (almost always needed)
- Parent/predecessor (for path reconstruction)
- Distance/level (for shortest path)
- Recursion stack (for cycle in directed)

---

**Interview Communication:**

**Example: Number of Islands**

1. **Clarify:**
   - "Is this a 2D grid of land and water?"
   - "Are diagonals considered connected?" (Usually no)
   - "Can I modify the input grid?"

2. **Explain approach:**
   - "This is a connected components problem."
   - "Each island is one component."
   - "I'll iterate through the grid and run DFS from each unvisited land cell."
   - "DFS will mark all connected land as visited."

3. **Walk through example:**
   \`\`\`
   Grid:
   1 1 0
   0 1 0
   0 0 1
   
   Start at (0,0): DFS marks (0,0), (0,1), (1,1) → Island 1
   Start at (2,2): DFS marks (2,2) → Island 2
   Total: 2 islands
   \`\`\`

4. **Complexity:**
   - "Time: O(M * N) - visit each cell once."
   - "Space: O(M * N) - worst case recursion depth."

---

**Common Pitfalls:**

**1. Forgetting Visited Set**
Leads to infinite loops!

**2. Wrong Parent Tracking**
For undirected graphs, track parent to avoid false cycle detection.

**3. Not Handling Disconnected Graphs**
Must try DFS/BFS from all nodes, not just one.

**4. Modifying Graph During Traversal**
Be careful with in-place modifications.

---

**Practice Plan:**

1. **Basics (Day 1-2):**
   - Number of Islands
   - Clone Graph
   - Course Schedule

2. **Traversals (Day 3-4):**
   - All Paths from Source to Target
   - Pacific Atlantic Water Flow
   - Surrounded Regions

3. **Advanced (Day 5-7):**
   - Network Delay Time (Dijkstra)
   - Alien Dictionary
   - Critical Connections

4. **Resources:**
   - LeetCode Graph tag (200+ problems)
   - Practice both DFS and BFS variants
   - Draw graphs for visualization`,
    },
  ],
  keyTakeaways: [
    'Graphs consist of vertices (nodes) connected by edges; can be directed/undirected, weighted/unweighted',
    'BFS explores level-by-level using queue; finds shortest path in unweighted graphs',
    'DFS explores deeply using stack/recursion; better for cycle detection and memory efficiency',
    'Adjacency list (dict of lists) is most common representation: O(V + E) space',
    'Most graph algorithms are O(V + E) time - linear in graph size',
    'Connected components: run DFS/BFS from each unvisited node',
    "Topological sort: order nodes in DAG using Kahn's algorithm (BFS with in-degrees)",
    'Union-Find provides near-constant time connectivity queries with path compression',
  ],
  relatedProblems: ['number-of-islands', 'course-schedule', 'clone-graph'],
};
