/**
 * BFS on Graphs Section
 */

export const graphbfsSection = {
  id: 'graph-bfs',
  title: 'BFS on Graphs',
  content: `**Graph BFS** finds the shortest path in unweighted graphs and explores connected components.

**Key Differences from Tree BFS:**
- Must track **visited nodes** (graphs have cycles)
- Can start from any node
- May need to handle disconnected components

**Basic Graph BFS Template:**
\`\`\`python
from collections import deque

def bfs (graph, start):
    visited = set()
    queue = deque([start])
    visited.add (start)
    
    while queue:
        node = queue.popleft()
        print(node)  # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append (neighbor)
\`\`\`

**Shortest Path in Unweighted Graph:**
\`\`\`python
def shortest_path (graph, start, end):
    if start == end:
        return 0
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return dist + 1
            
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # No path exists
\`\`\`

**Shortest Path with Parent Tracking:**
\`\`\`python
def shortest_path_with_route (graph, start, end):
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([start])
    parent = {start: None}
    
    while queue:
        node = queue.popleft()
        
        if node == end:
            # Reconstruct path
            path = []
            curr = end
            while curr is not None:
                path.append (curr)
                curr = parent[curr]
            return path[::-1]
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                parent[neighbor] = node
                queue.append (neighbor)
    
    return []  # No path
\`\`\`

**Multi-Source BFS:**
\`\`\`python
def multi_source_bfs (graph, sources):
    """
    BFS starting from multiple sources simultaneously.
    Useful for problems like "distance from nearest X"
    """
    visited = set (sources)
    queue = deque([(s, 0) for s in sources])
    distances = {}
    
    while queue:
        node, dist = queue.popleft()
        distances[node] = dist
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add (neighbor)
                queue.append((neighbor, dist + 1))
    
    return distances
\`\`\`

**0-1 BFS (Weighted with 0/1 weights):**
\`\`\`python
def bfs_01(graph, start, end):
    """
    Special BFS for graphs with edges of weight 0 or 1.
    Use deque: add 0-weight edges to front, 1-weight to back
    """
    from collections import deque
    
    dist = {start: 0}
    deque_bfs = deque([start])
    
    while deque_bfs:
        node = deque_bfs.popleft()
        
        if node == end:
            return dist[node]
        
        for neighbor, weight in graph[node]:
            new_dist = dist[node] + weight
            
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                
                if weight == 0:
                    deque_bfs.appendleft (neighbor)  # 0-weight: front
                else:
                    deque_bfs.append (neighbor)      # 1-weight: back
    
    return -1
\`\`\``,
};
