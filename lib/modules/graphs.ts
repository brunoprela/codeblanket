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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain what a graph is and how it differs from a tree. When would you use a graph over a tree?',
          sampleAnswer:
            'A graph is a collection of nodes (vertices) connected by edges. Unlike trees, graphs can have cycles, multiple paths between nodes, and disconnected components. Trees are special graphs: connected, acyclic, with exactly n-1 edges for n nodes. Use graphs when relationships are not strictly hierarchical - for example, social networks where friendships are bidirectional and can form cycles, or road maps where multiple routes exist between cities. Trees model hierarchical relationships like file systems or org charts. Graphs model peer relationships, networks, dependencies, and any scenario where cycles or multiple connections are natural. Graphs are more general; trees are constrained graphs.',
          keyPoints: [
            'Graph: nodes connected by edges',
            'Can have cycles, multiple paths, disconnected components',
            'Trees: special graphs (connected, acyclic, n-1 edges)',
            'Use graphs: non-hierarchical relationships, cycles',
            'Trees: hierarchical, Graphs: peer relationships',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare adjacency list and adjacency matrix representations. When would you choose each?',
          sampleAnswer:
            'Adjacency list stores for each node a list of its neighbors. Space is O(V + E) where V is vertices, E is edges. Good for sparse graphs where E << V^2. Checking if edge exists is O(degree), adding edge is O(1). Adjacency matrix is VxV grid where matrix[i][j] = 1 if edge exists. Space is O(V^2) regardless of edges. Good for dense graphs or when you frequently check if edge exists (O(1) lookup). For example, social network with 1M users but each connected to ~100: list uses 100M entries, matrix needs 1T entries - list wins. For complete graph where everyone connects to everyone: matrix is better. In practice, most real graphs are sparse, so adjacency list is more common.',
          keyPoints: [
            'List: O(V + E) space, good for sparse',
            'Matrix: O(V^2) space, good for dense',
            'List: O(degree) edge check, Matrix: O(1)',
            'Most real graphs sparse → list preferred',
            'Matrix: when frequent edge existence checks',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe directed vs undirected graphs. Give real-world examples where each is appropriate.',
          sampleAnswer:
            'Directed graphs have edges with direction: A→B does not imply B→A. Used for asymmetric relationships like Twitter follows (I follow you, you might not follow me), web page links, task dependencies (task A must complete before B). Undirected graphs have bidirectional edges: A-B means both A to B and B to A. Used for symmetric relationships like Facebook friendships (mutual), road connections (bidirectional travel), collaboration networks. In code, directed graphs store edges once, undirected store twice (both directions) or check both ways. Directed enables modeling one-way relationships and detecting cycles in dependencies. Undirected is simpler when relationships are naturally symmetric.',
          keyPoints: [
            'Directed: edges have direction (asymmetric)',
            'Examples: Twitter follows, web links, task dependencies',
            'Undirected: bidirectional edges (symmetric)',
            'Examples: friendships, roads, collaborations',
            'Choice depends on relationship symmetry',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is a graph?',
          options: [
            'A tree',
            'Vertices (nodes) connected by edges modeling relationships/networks',
            'An array',
            'A sorted list',
          ],
          correctAnswer: 1,
          explanation:
            'A graph consists of vertices (nodes) connected by edges. Graphs model networks like social connections, maps, dependencies, etc. Unlike trees, graphs can have cycles.',
        },
        {
          id: 'mc2',
          question: 'What is the most common graph representation and why?',
          options: [
            'Adjacency matrix',
            'Adjacency list - space efficient O(V+E), fast neighbor iteration',
            'Edge list',
            'Array',
          ],
          correctAnswer: 1,
          explanation:
            'Adjacency list is most common: space O(V+E) vs matrix O(V²), fast to iterate neighbors. Most real graphs are sparse, making list more efficient.',
        },
        {
          id: 'mc3',
          question: 'What is a DAG?',
          options: [
            'A weighted graph',
            'Directed Acyclic Graph - directed with no cycles',
            'A tree',
            'Dense graph',
          ],
          correctAnswer: 1,
          explanation:
            'DAG = Directed Acyclic Graph. Edges have direction and no cycles exist. Used for dependency graphs, scheduling, compilation order. Enables topological sorting.',
        },
        {
          id: 'mc4',
          question: 'When should you use adjacency matrix over adjacency list?',
          options: [
            'Always',
            'Dense graphs where checking edge existence is frequent (O(1) lookup)',
            'Sparse graphs',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Use matrix when graph is dense (many edges) and need fast O(1) edge lookup. Matrix uses O(V²) space - inefficient for sparse graphs where list is better.',
        },
        {
          id: 'mc5',
          question:
            'What is the difference between directed and undirected graphs?',
          options: [
            'No difference',
            'Directed: edges have direction (A→B), Undirected: edges are bidirectional (A↔B)',
            'Directed is faster',
            'Undirected has more edges',
          ],
          correctAnswer: 1,
          explanation:
            "Directed: edges go one way (A→B doesn't mean B→A). Undirected: edges go both ways (A-B means both A→B and B→A). Social networks often undirected, web pages directed.",
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Compare BFS and DFS for graphs. When would you choose one over the other?',
          sampleAnswer:
            'BFS explores level by level using a queue, visiting all neighbors before going deeper. DFS explores as deep as possible using a stack (or recursion), going down one path before backtracking. BFS finds shortest path in unweighted graphs - guaranteed to find closest node first. Use BFS for: shortest path, level-order problems, finding closest node. DFS is better for: detecting cycles, topological sort, exhaustive search (like finding all paths). BFS uses O(width) space for queue, DFS uses O(depth) for stack. In dense graphs, BFS might use more memory. For finding if path exists, either works. For optimization problems, choice depends on whether shortest or any path matters.',
          keyPoints: [
            'BFS: level by level, queue, finds shortest path',
            'DFS: deep first, stack/recursion, exhaustive search',
            'BFS: closest node, level problems',
            'DFS: cycles, topological sort, all paths',
            'Space: BFS O(width), DFS O(depth)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain cycle detection in graphs. How does the approach differ for directed vs undirected graphs?',
          sampleAnswer:
            'For undirected graphs, use DFS with visited set. If we reach a visited node that is not our immediate parent, there is a cycle - we found two different paths to same node. Track parent to avoid false positive from bidirectional edge. For directed graphs, we need three states: unvisited, visiting (in current DFS path), and visited (completely done). If we reach a node in "visiting" state, there is a cycle - we came back to an ancestor in current path. After exploring a node, mark it visited. The key difference: undirected needs parent tracking, directed needs recursion stack tracking. Both use DFS because we need to track current path.',
          keyPoints: [
            'Undirected: DFS, detect visited node (not parent)',
            'Track parent to avoid false positive',
            'Directed: three states (unvisited, visiting, visited)',
            'Visiting state = in current DFS path',
            'Reach visiting node = cycle found',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through topological sort. Why does it only work for DAGs (Directed Acyclic Graphs)?',
          sampleAnswer:
            'Topological sort produces linear ordering where for every edge u→v, u comes before v. Use DFS: visit all nodes, after finishing a node (all descendants explored), add it to result. Reverse the result for topological order. Why only DAGs? If there is a cycle A→B→C→A, we cannot order them linearly - each should come before the next, creating contradiction. For example, task dependencies: if A depends on B, B on C, C on A, we cannot determine start order. Acyclic ensures no circular dependencies. Applications: task scheduling, course prerequisites, build systems. Kahn algorithm alternative uses BFS and in-degrees - repeatedly remove nodes with zero incoming edges.',
          keyPoints: [
            'Linear ordering: for edge u→v, u before v',
            'DFS: add node after finishing, then reverse',
            'Only DAGs: cycles create ordering contradiction',
            'Example: circular task dependencies impossible',
            'Alternative: Kahn algorithm with BFS and in-degrees',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the key difference between BFS and DFS?',
          options: [
            'Speed',
            'BFS explores level-by-level (queue), DFS explores depth-first (stack/recursion)',
            'Space only',
            'They are the same',
          ],
          correctAnswer: 1,
          explanation:
            'BFS uses queue to explore level-by-level (closest first). DFS uses stack/recursion to explore depth-first (go deep before backtracking). Different exploration orders.',
        },
        {
          id: 'mc2',
          question: 'When should you use BFS over DFS?',
          options: [
            'Always',
            'Shortest path unweighted, level-order, closest nodes first',
            'Any traversal',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Use BFS for: shortest path in unweighted graph (finds closest), level-order traversal, minimum moves. BFS explores by distance from start.',
        },
        {
          id: 'mc3',
          question: 'When should you use DFS over BFS?',
          options: [
            'Never',
            'Pathfinding, cycle detection, topological sort, connected components',
            'Shortest path only',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Use DFS for: any path (not shortest), cycle detection, topological sort, connected components, backtracking. DFS explores deeply, better for path existence.',
        },
        {
          id: 'mc4',
          question: 'What data structure does BFS use?',
          options: [
            'Stack',
            'Queue (FIFO) - explores level-by-level',
            'Heap',
            'Array',
          ],
          correctAnswer: 1,
          explanation:
            'BFS uses queue (FIFO). Add neighbors to queue, process in order added. This ensures level-by-level exploration (all distance k before k+1).',
        },
        {
          id: 'mc5',
          question: 'Why track visited nodes in graph traversals?',
          options: [
            'For speed',
            'Prevents infinite loops in cycles, ensures O(V+E) time',
            'Random requirement',
            'Memory optimization',
          ],
          correctAnswer: 1,
          explanation:
            'Visited set prevents revisiting nodes. Without it, cycles cause infinite loops. With it, each node processed once, giving O(V+E) time complexity.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the connected components pattern. Why do we need to check all nodes, not just start from one?',
          sampleAnswer:
            'Connected components finds all separate groups in an undirected graph. We iterate through all nodes, running DFS or BFS from each unvisited node. Each DFS/BFS explores one complete component. We need to check all nodes because the graph might be disconnected - not all nodes reachable from one starting point. For example, social network might have isolated friend groups with no connections between them. Starting from one node only finds that person connected group. By checking all nodes and tracking visited, we ensure we discover all components. Each time we start a new DFS from unvisited node, we have found a new component. Count increases with each new starting point.',
          keyPoints: [
            'Find all separate groups in graph',
            'DFS/BFS from each unvisited node',
            'Graph might be disconnected',
            'One start point only finds one component',
            'New DFS start = new component found',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe Kahn algorithm for topological sort. How does tracking in-degrees help?',
          sampleAnswer:
            'Kahn algorithm uses BFS and tracks in-degrees (number of incoming edges) for each node. Start with nodes having in-degree 0 (no dependencies) - add them to queue. Process queue: remove node, add to result, decrease in-degree of all neighbors by 1. If a neighbor in-degree becomes 0, add it to queue. This ensures we only process a node after all its prerequisites are processed. The in-degree tracking tells us when all dependencies are satisfied. If we finish and some nodes remain unprocessed, there is a cycle. For task scheduling, in-degree 0 means task has no prerequisites and can start immediately. As we complete tasks, dependent tasks become ready. This is intuitive and easier to understand than DFS topological sort.',
          keyPoints: [
            'BFS with in-degree tracking',
            'Start with in-degree 0 nodes (no dependencies)',
            'Process node, decrease neighbor in-degrees',
            'In-degree 0 = all dependencies satisfied',
            'Unprocessed nodes at end = cycle exists',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through bipartite checking with coloring. Why does odd-length cycle prevent bipartiteness?',
          sampleAnswer:
            'Bipartite graph can be colored with two colors such that no adjacent nodes have same color. Use BFS/DFS: color starting node with color 0, color all neighbors with color 1, their neighbors with color 0, etc. If we ever try to color a node but it already has different color, graph is not bipartite. Odd-length cycle prevents bipartiteness because as we alternate colors around cycle, we end up trying to give same node two different colors. For example, triangle ABC: A is color 0, B is color 1, C is color 0, but C connects to A (also color 0) - conflict! Even cycles work: alternate colors around cycle ends correctly. Applications: matching problems, scheduling with conflicts.',
          keyPoints: [
            'Two-color the graph, no adjacent same color',
            'BFS/DFS: alternate colors',
            'Already-colored node with different color = not bipartite',
            'Odd cycle: colors conflict when cycle closes',
            'Even cycles: colors alternate correctly',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'How do you find connected components in an undirected graph?',
          options: [
            'Sort nodes',
            'Run BFS/DFS from each unvisited node, each traversal finds one component',
            'Use heap',
            'Random selection',
          ],
          correctAnswer: 1,
          explanation:
            'Connected components: iterate through nodes, for each unvisited node, run BFS/DFS (marks all reachable nodes). Each traversal finds one component. Count of traversals = number of components.',
        },
        {
          id: 'mc2',
          question: 'How do you detect a cycle in a directed graph?',
          options: [
            'BFS only',
            'DFS with recursion stack - if we visit a node already in current path, cycle exists',
            'Sort',
            'Cannot detect',
          ],
          correctAnswer: 1,
          explanation:
            'Directed cycle: DFS with recursion stack. Track nodes in current path. If we reach a node already in path (not just visited), found a cycle back to ancestor.',
        },
        {
          id: 'mc3',
          question:
            'What makes topological sort possible for DAGs but not cyclic graphs?',
          options: [
            'Speed',
            'DAGs have no cycles - can order linearly where all edges go forward',
            'Random',
            'Size',
          ],
          correctAnswer: 1,
          explanation:
            'Topological sort orders nodes so all edges go from earlier to later. Cycles make this impossible (circular dependency). DAGs (acyclic) guarantee such ordering exists.',
        },
        {
          id: 'mc4',
          question: 'How do you check if a graph is bipartite?',
          options: [
            'Count nodes',
            'Try 2-coloring with BFS/DFS - if adjacent nodes get same color, not bipartite',
            'Sort edges',
            'Random testing',
          ],
          correctAnswer: 1,
          explanation:
            'Bipartite: attempt to 2-color graph. BFS/DFS alternating colors. If we try to color a node but it already has different color, graph has odd cycle and is not bipartite.',
        },
        {
          id: 'mc5',
          question: 'What is Union-Find used for in graph problems?',
          options: [
            'Sorting',
            'Efficiently tracking connected components, detecting cycles in undirected graphs',
            'Shortest path',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Union-Find (Disjoint Set): efficiently merge components (union) and check if nodes connected (find). O(α(N)) amortized per operation. Used for Kruskal MST, cycle detection.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why BFS and DFS are both O(V + E) time. Walk me through what each part represents.',
          sampleAnswer:
            'Both visit every vertex once and explore every edge once, giving O(V + E). V represents visiting each vertex: we mark it visited, process it, add to queue or call recursion. E represents exploring edges: for each vertex, we check all its neighbors via edges. In adjacency list, summing all neighbor lists gives exactly E edges total. For example, graph with 5 vertices and 7 edges: we visit 5 vertices (V work) and check 7 edges (E work), so 5 + 7 = 12 operations. The + not × because we do V work (visiting) and separately E work (edge checking), not nested. This is why adjacency list is efficient - we only examine edges that exist, not all V^2 possible edges.',
          keyPoints: [
            'Visit each vertex once = O(V)',
            'Explore each edge once = O(E)',
            'Sum of all adjacency lists = E',
            'V + E not V × E: separate operations',
            'Adjacency list: only check existing edges',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare time complexity of Dijkstra vs Bellman-Ford. When would you choose the slower one?',
          sampleAnswer:
            'Dijkstra is O(E log V) with min-heap, Bellman-Ford is O(V × E). Dijkstra is faster but requires non-negative edge weights. Bellman-Ford handles negative weights and detects negative cycles. Choose Bellman-Ford when: graph has negative edges, need to detect negative cycles, or graph is small so cubic time acceptable. For example, currency exchange with fees (negative edges) or finding arbitrage (negative cycle detection). Dijkstra fails with negative edges because greedy approach assumes once a node is finalized, no better path exists. Negative edges can create better paths to already-finalized nodes. Most real graphs (roads, networks) have non-negative weights, so Dijkstra is usually preferred.',
          keyPoints: [
            'Dijkstra: O(E log V), needs non-negative weights',
            'Bellman-Ford: O(V × E), handles negative weights',
            'Bellman-Ford: detects negative cycles',
            'Use Bellman-Ford: negative edges, detect cycles',
            'Dijkstra fails with negative: greedy assumption broken',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain Union-Find complexity with path compression and union by rank. Why is it nearly constant?',
          sampleAnswer:
            'Union-Find with optimizations achieves O(α(n)) per operation where α is inverse Ackermann function - grows so slowly it is effectively constant for all practical n (less than 5 for n < 10^80). Path compression: during find, make all nodes point directly to root, flattening tree. Union by rank: attach smaller tree under larger, keeping trees shallow. Together, these prevent deep trees. Without optimizations, trees can be O(n) deep, making operations O(n). With both optimizations, trees stay very flat (height < 5 practically), giving nearly O(1) operations. This makes Union-Find extremely efficient for dynamic connectivity - can handle millions of operations in linear time.',
          keyPoints: [
            'With optimizations: O(α(n)) ≈ O(1) practical',
            'Path compression: flatten tree during find',
            'Union by rank: attach smaller tree under larger',
            'Prevents deep trees (height < 5 practically)',
            'Without optimizations: O(n) per operation',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the time complexity of BFS/DFS traversal?',
          options: [
            'O(V)',
            'O(V + E) - visit all vertices and edges once',
            'O(E)',
            'O(V²)',
          ],
          correctAnswer: 1,
          explanation:
            'BFS/DFS: O(V + E). Visit each vertex once (O(V)), explore each edge once (O(E)). Total: O(V + E). Space: O(V) for visited set.',
        },
        {
          id: 'mc2',
          question:
            "What is the time complexity of Dijkstra's algorithm with min heap?",
          options: [
            'O(V + E)',
            'O((V + E) log V) - heap operations for all edges',
            'O(V²)',
            'O(E log E)',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra with min heap: each edge causes heap operation O(log V). Process O(V + E) edges total. Time: O((V + E) log V). Better than O(V²) for sparse graphs.',
        },
        {
          id: 'mc3',
          question: 'When do you need Bellman-Ford instead of Dijkstra?',
          options: [
            'Always',
            'When graph has negative weight edges',
            'For faster execution',
            'Random choice',
          ],
          correctAnswer: 1,
          explanation:
            'Dijkstra fails with negative weights. Bellman-Ford handles negative edges and detects negative cycles. Time: O(V*E) vs Dijkstra O((V+E) log V).',
        },
        {
          id: 'mc4',
          question: 'What is the space complexity of graph traversals?',
          options: [
            'O(1)',
            'O(V) for visited set and queue/stack',
            'O(E)',
            'O(V²)',
          ],
          correctAnswer: 1,
          explanation:
            'Graph traversals: O(V) space for visited set + queue/stack can hold up to O(V) nodes. DFS recursion: O(H) where H is depth.',
        },
        {
          id: 'mc5',
          question:
            'Why is adjacency list better than adjacency matrix for sparse graphs?',
          options: [
            'Faster edge lookup',
            'Space: O(V+E) vs O(V²), and most real graphs are sparse',
            'Random',
            'Always better',
          ],
          correctAnswer: 1,
          explanation:
            'Sparse graphs have few edges (E << V²). List uses O(V+E) space, matrix O(V²). Social networks, web graphs are sparse, making list much more efficient.',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Walk me through the BFS template. Why do we use a queue and how does the visited set prevent issues?',
          sampleAnswer:
            'BFS uses queue for level-order traversal: process all nodes at distance d before distance d+1. Start by adding source to queue and marking visited. While queue not empty: dequeue node, process it, enqueue unvisited neighbors and mark them visited. Queue ensures FIFO - first discovered are first processed, guaranteeing level-order. Visited set prevents two issues: infinite loops (cycles cause revisiting) and redundant work (processing same node multiple times). Mark visited when enqueueing, not when dequeueing - prevents adding same node to queue multiple times. This template finds shortest path in unweighted graphs because we explore by distance from source.',
          keyPoints: [
            'Queue: FIFO for level-order traversal',
            'Mark visited when enqueueing',
            'Prevents: infinite loops and redundant work',
            'Process level d before d+1',
            'Finds shortest path in unweighted graphs',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the DFS recursive template. How does the visited set interact with recursion?',
          sampleAnswer:
            'DFS explores as deep as possible before backtracking. Mark current node visited, process it, then recursively visit each unvisited neighbor. The recursion stack implicitly tracks current path - when a recursive call returns, we backtrack to try other branches. Visited set ensures we do not revisit nodes, preventing infinite loops in cycles. Mark visited before recursing to neighbors - this is the "choose" step. Unlike backtracking where we unmark (unchoose), in graph traversal we keep nodes marked because we do not need to revisit. The combination of recursion (for path) and visited set (for seen nodes) enables complete graph exploration without redundancy.',
          keyPoints: [
            'Recursive: explore deep first',
            'Mark visited before recursing to neighbors',
            'Recursion stack tracks current path',
            'Visited set prevents revisiting (no unmark needed)',
            'Returns = backtrack to try other branches',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe the topological sort DFS template. Why do we add to result after exploring neighbors?',
          sampleAnswer:
            'Topological sort DFS visits all nodes, recursively exploring neighbors first, then adding current node to result after finishing. This post-order traversal ensures dependencies are added before dependents. Reverse the result to get topological order. Why add after exploring? Consider edge A→B: we must output A before B. When visiting A, we recurse to B first. B finishes and gets added. Then A finishes and gets added. Result has [B, A], reverse gives [A, B] - correct! Adding before recursing would give wrong order. The key insight: node is added when all descendants are processed - exactly when all dependencies are satisfied. This is why DFS naturally produces reverse topological order.',
          keyPoints: [
            'Post-order: add after exploring all neighbors',
            'Ensures dependencies added before dependents',
            'Reverse result for topological order',
            'Node added when all descendants processed',
            'DFS naturally produces reverse topological order',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What data structure does BFS template use?',
          options: [
            'Stack',
            'Queue (deque) for level-by-level exploration',
            'Heap',
            'Array',
          ],
          correctAnswer: 1,
          explanation:
            'BFS template uses queue (collections.deque). Add start, while queue: pop, process, add unvisited neighbors. Queue ensures level-by-level exploration.',
        },
        {
          id: 'mc2',
          question:
            'What is the key difference between iterative and recursive DFS templates?',
          options: [
            'Speed',
            'Iterative uses explicit stack, recursive uses call stack',
            'Complexity',
            'They are the same',
          ],
          correctAnswer: 1,
          explanation:
            'DFS: recursive uses call stack (cleaner code), iterative uses explicit stack (more control, avoids stack overflow). Both explore depth-first, same complexity.',
        },
        {
          id: 'mc3',
          question:
            'In Union-Find template, what do path compression and union by rank do?',
          options: [
            'Nothing',
            'Flatten trees during find and attach smaller under larger for O(α(N)) amortized',
            'Sort elements',
            'Random optimization',
          ],
          correctAnswer: 1,
          explanation:
            'Path compression: flatten tree during find (make nodes point to root). Union by rank: attach smaller tree under larger. Together give O(α(N)) ≈ O(1) per operation.',
        },
        {
          id: 'mc4',
          question: 'What is common to all graph traversal templates?',
          options: [
            'Sorting',
            'Visited set to track processed nodes and prevent cycles',
            'Heap usage',
            'Random selection',
          ],
          correctAnswer: 1,
          explanation:
            'All graph templates use visited set to: 1) prevent infinite loops in cycles, 2) ensure each node processed once, 3) achieve O(V+E) complexity.',
        },
        {
          id: 'mc5',
          question: 'When would you modify the basic BFS template?',
          options: [
            'Never',
            'Track distance/level, find shortest path, level-order specific logic',
            'Always',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Modify BFS for: tracking distance (distance array), shortest path (parent pointers), level-order processing (track level), multi-source BFS (start with multiple nodes).',
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize when a problem needs graph algorithms? What keywords or patterns signal this?',
          sampleAnswer:
            'Several signals indicate graph problems. Explicit mentions: "network", "graph", "connections", "dependencies", "path". Implicit: relationships between entities (social connections, prerequisites), grid traversal (islands, word search - grids are implicit graphs), scheduling with dependencies. Keywords like "reachable", "connected", "shortest path", "cycle", "order tasks" suggest graphs. For example, "find shortest path between cities" is clearly graph. "Can you finish all courses given prerequisites" is graph (topological sort). "Count number of islands in grid" is graph (connected components on grid). Ask: are there entities with relationships? Do I need to traverse connections? Is there a network structure?',
          keyPoints: [
            'Explicit: network, graph, connections, dependencies',
            'Implicit: entity relationships, grid traversal',
            'Keywords: reachable, connected, shortest, cycle, order',
            'Prerequisites → topological sort',
            'Grid problems → graph on grid',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your approach to a graph problem in an interview. What questions do you ask?',
          sampleAnswer:
            'First, clarify the graph structure: directed or undirected? Weighted or unweighted? Can it have cycles? How is it represented (adjacency list, matrix, edges list)? Then identify the problem type: shortest path, connectivity, cycle detection, topological sort? Based on type, choose algorithm: BFS for shortest path in unweighted, Dijkstra for weighted, DFS for cycles, Union-Find for connectivity. State complexity: O(V + E) for BFS/DFS. Discuss edge cases: empty graph, disconnected, self-loops. Draw small example: 4 nodes, show algorithm steps. Code with clear structure: build graph, run algorithm, return result. Mention optimizations: early termination, bidirectional search.',
          keyPoints: [
            'Clarify: directed/undirected, weighted, cycles, representation',
            'Identify problem type: path, connectivity, cycle, topological',
            'Choose algorithm based on type',
            'State complexity with reasoning',
            'Draw example, show steps',
            'Code clearly, discuss optimizations',
          ],
        },
        {
          id: 'q3',
          question:
            'What are common pitfalls in graph problems and how do you avoid them?',
          sampleAnswer:
            'First: forgetting to handle disconnected graphs - must iterate through all nodes, not just start from one. Second: not marking visited, causing infinite loops in cycles. Third: marking visited at wrong time in BFS (mark when enqueueing, not dequeueing). Fourth: for undirected graphs, adding edges both directions or checking both. Fifth: off-by-one in adjacency matrix vs list indexing. Sixth: not handling empty graph or single node. Seventh: in topological sort, not checking if all nodes processed (indicates cycle). My strategy: draw the graph, trace algorithm on paper, test with cycle and disconnected cases, verify visited logic, check directed vs undirected handling.',
          keyPoints: [
            'Handle disconnected graphs (check all nodes)',
            'Mark visited to prevent infinite loops',
            'BFS: mark when enqueueing',
            'Undirected: edges both ways',
            'Test: cycles, disconnected, empty graph',
            'Topological: verify all nodes processed',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What keywords signal a graph problem?',
          options: [
            'Array, list',
            'Network, graph, nodes/edges, dependencies, connections, paths',
            'Sorting',
            'Random',
          ],
          correctAnswer: 1,
          explanation:
            'Keywords: "network", "graph", "nodes/edges", "dependencies", "connections", "paths", "islands", "relationships". These indicate graph structure and traversal.',
        },
        {
          id: 'mc2',
          question: 'What should you clarify first in a graph interview?',
          options: [
            'Complexity only',
            'Directed/undirected? Weighted? Cycles? Connected? Representation?',
            'Language',
            'Nothing',
          ],
          correctAnswer: 1,
          explanation:
            'Clarify: 1) Directed vs undirected, 2) Weighted edges?, 3) Cycles allowed?, 4) Connected or multiple components?, 5) Representation (adjacency list/matrix). These determine algorithm choice.',
        },
        {
          id: 'mc3',
          question: 'What is a common mistake in graph problems?',
          options: [
            'Using traversal',
            'Forgetting visited set - causes infinite loops in cycles',
            'Good naming',
            'Complexity analysis',
          ],
          correctAnswer: 1,
          explanation:
            'Most common: forgetting visited set. Without it, cycles cause infinite loops. Always track visited nodes to prevent revisiting and ensure O(V+E) complexity.',
        },
        {
          id: 'mc4',
          question: 'When should you choose BFS over DFS in an interview?',
          options: [
            'Always',
            'Shortest path unweighted, level-order, minimum steps',
            'Any problem',
            'Never',
          ],
          correctAnswer: 1,
          explanation:
            'Choose BFS for: shortest path in unweighted graph, level-order traversal, minimum steps/moves. BFS explores by distance, finding closest first.',
        },
        {
          id: 'mc5',
          question: 'What is good practice for graph interview communication?',
          options: [
            'Just code',
            'Clarify, explain approach (BFS/DFS/pattern), walk through example, discuss complexity',
            'Write fast',
            'Skip explanation',
          ],
          correctAnswer: 1,
          explanation:
            'Structure: 1) Clarify graph properties, 2) Identify pattern (traversal, shortest path, connected components), 3) Explain algorithm choice, 4) Walk through example, 5) Complexity analysis.',
        },
      ],
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
