/**
 * Common Graph Patterns Section
 */

export const patternsSection = {
  id: 'patterns',
  title: 'Common Graph Patterns',
  content: `**Pattern 1: Connected Components**

Find all connected groups in undirected graph.

**Approach**: Run BFS/DFS from each unvisited node.

\`\`\`python
def count_components (n, edges):
    # Build adjacency list
    graph = {i: [] for i in range (n)}
    for a, b in edges:
        graph[a].append (b)
        graph[b].append (a)
    
    visited = set()
    count = 0
    
    def dfs (node):
        visited.add (node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs (neighbor)
    
    for node in range (n):
        if node not in visited:
            dfs (node)
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

**Algorithm (Kahn\'s - BFS):**
\`\`\`python
def topological_sort (graph):
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
        result.append (node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append (neighbor)
    
    # Check if all nodes processed (no cycle)
    return result if len (result) == len (graph) else []
\`\`\`

---

**Pattern 4: Shortest Path (Weighted)**

**Dijkstra's Algorithm**: Find shortest path in weighted graph with non-negative weights.

\`\`\`python
import heapq

def dijkstra (graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (distance, node)
    
    while pq:
        curr_dist, node = heapq.heappop (pq)
        
        if curr_dist > distances[node]:
            continue
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush (pq, (distance, neighbor))
    
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
        self.parent = list (range (n))
        self.rank = [0] * n
    
    def find (self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find (self.parent[x])  # Path compression
        return self.parent[x]
    
    def union (self, x, y):
        root_x, root_y = self.find (x), self.find (y)
        
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
};
