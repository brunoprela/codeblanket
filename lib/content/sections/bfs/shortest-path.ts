/**
 * BFS for Shortest Path Section
 */

export const shortestpathSection = {
  id: 'shortest-path',
  title: 'BFS for Shortest Path',
  content: `**BFS guarantees shortest path** in unweighted graphs because it explores nodes in order of increasing distance.

**Why BFS Finds Shortest Path:**
1. Explores all nodes at distance k before distance k+1
2. First time you reach a node = shortest path to that node
3. Works only for **unweighted** graphs (all edges cost 1)

**Standard Shortest Path Template:**
\`\`\`python
def shortest_path_bfs(graph, start, end):
    if start == end:
        return 0
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        node, distance = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # Unreachable
\`\`\`

**With Path Reconstruction:**
\`\`\`python
def shortest_path_with_reconstruction(graph, start, end):
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([start])
    parent = {start: None}
    
    # BFS to find end
    while queue:
        node = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)
                
                if neighbor == end:
                    # Reconstruct path
                    path = []
                    curr = end
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    return path[::-1]
    
    return []  # No path
\`\`\`

**Grid Shortest Path (4-directional):**
\`\`\`python
def shortest_path_grid(grid, start, end):
    """
    Find shortest path in 2D grid.
    0 = walkable, 1 = blocked
    """
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    queue = deque([(start[0], start[1], 0)])
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == end:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1  # Unreachable
\`\`\`

**Key Points:**
- **First visit = shortest path** (in unweighted graphs)
- Use **visited set** to avoid cycles
- Track **distance or parent** for path reconstruction
- For weighted graphs, use Dijkstra instead`,
};
