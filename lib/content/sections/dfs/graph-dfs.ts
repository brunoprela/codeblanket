/**
 * DFS on Graphs Section
 */

export const graphdfsSection = {
  id: 'graph-dfs',
  title: 'DFS on Graphs',
  content: `**Graph DFS** is more complex than tree DFS because graphs can have cycles. We need to track visited nodes.

**Key Differences from Tree DFS:**
- Must track **visited nodes** (avoid cycles)
- Can have multiple connected components
- May need to explore all nodes as starting points

**Adjacency List Representation:**
\`\`\`python
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}
\`\`\`

**Basic Graph DFS Template:**
\`\`\`python
def dfs(graph, start):
    visited = set()
    
    def explore(node):
        if node in visited:
            return
        
        visited.add(node)
        print(node)  # Process node
        
        for neighbor in graph[node]:
            explore(neighbor)
    
    explore(start)
\`\`\`

**Common Graph DFS Problems:**

**1. Connected Components**
\`\`\`python
def count_components(n, edges):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
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

**2. Cycle Detection**
\`\`\`python
def has_cycle(graph):
    visited = set()
    rec_stack = set()  # Nodes in current path
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True  # Back edge = cycle
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
\`\`\`

**3. Path Finding**
\`\`\`python
def find_path(graph, start, end):
    visited = set()
    path = []
    
    def dfs(node):
        if node in visited:
            return False
        
        visited.add(node)
        path.append(node)
        
        if node == end:
            return True
        
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        
        path.pop()  # Backtrack
        return False
    
    dfs(start)
    return path if path and path[-1] == end else []
\`\`\``,
};
