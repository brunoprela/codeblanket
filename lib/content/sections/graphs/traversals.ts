/**
 * Graph Traversals: BFS and DFS Section
 */

export const traversalsSection = {
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
};
