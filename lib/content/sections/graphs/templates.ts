/**
 * Code Templates Section
 */

export const templatesSection = {
  id: 'templates',
  title: 'Code Templates',
  content: `**Template 1: BFS**
\`\`\`python
from collections import deque

def bfs_template (graph, start):
    visited = set([start])
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append (neighbor)
    
    return visited
\`\`\`

**Template 2: DFS (Recursive)**
\`\`\`python
def dfs_template (graph, node, visited=None):
    if visited is None:
        visited = set()
    
    visited.add (node)
    # Process node
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_template (graph, neighbor, visited)
    
    return visited
\`\`\`

**Template 3: DFS (Iterative)**
\`\`\`python
def dfs_iterative_template (graph, start):
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add (node)
        # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append (neighbor)
    
    return visited
\`\`\`

**Template 4: Shortest Path (BFS)**
\`\`\`python
def shortest_path (graph, start, end):
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
                
                visited.add (neighbor)
                queue.append((neighbor, new_path))
    
    return []
\`\`\`

**Template 5: Cycle Detection (Undirected)**
\`\`\`python
def has_cycle_undirected (graph):
    visited = set()
    
    def dfs (node, parent):
        visited.add (node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs (neighbor, node):
                    return True
            elif neighbor != parent:
                return True  # Visited non-parent = cycle
        
        return False
    
    for node in graph:
        if node not in visited:
            if dfs (node, -1):
                return True
    
    return False
\`\`\`

**Template 6: Topological Sort**
\`\`\`python
def topological_sort (graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([n for n in graph if in_degree[n] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append (node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append (neighbor)
    
    return result if len (result) == len (graph) else []
\`\`\``,
};
