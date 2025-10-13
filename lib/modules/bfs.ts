import { Module } from '@/lib/types';

export const bfsModule: Module = {
  id: 'bfs',
  title: 'Breadth-First Search (BFS)',
  description:
    'Master breadth-first search for level-by-level traversal and finding shortest paths in unweighted graphs.',
  icon: '📊',
  timeComplexity: 'O(V + E) for graphs, O(N) for trees',
  spaceComplexity: 'O(W) where W is maximum width',
  sections: [
    {
      id: 'introduction',
      title: 'Introduction to BFS',
      content: `**Breadth-First Search (BFS)** explores a graph or tree level by level, visiting all neighbors of a node before moving to the next level.

**Core Concept:**
- Start at a node
- Visit all immediate neighbors first (go wide)
- Then visit neighbors of neighbors
- Continue level by level

**Key Characteristics:**
- Uses a **queue** (FIFO - First In, First Out)
- Explores **layer by layer**
- **Finds shortest path** in unweighted graphs
- Visits nodes in order of distance from start

**BFS vs DFS:**
- **BFS**: Queue → Goes wide → O(W) space → Shortest path
- **DFS**: Stack/Recursion → Goes deep → O(H) space → All paths

**When to Use BFS:**
- **Shortest path** in unweighted graphs
- **Level-order** traversal of trees
- **Nearest neighbor** problems
- **Minimum number of steps/moves**
- Finding nodes at specific distance
- Web crawling (breadth-first)`,
    },
    {
      id: 'tree-bfs',
      title: 'BFS on Trees (Level-Order Traversal)',
      content: `**Level-Order Traversal** visits nodes level by level, left to right.

**Example Tree:**
\`\`\`
      1
     / \\
    2   3
   / \\
  4   5
\`\`\`
**Level-order:** 1, 2, 3, 4, 5

**Basic Template:**
\`\`\`python
from collections import deque

def level_order(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
\`\`\`

**Level-by-Level Processing:**
\`\`\`python
def level_order_levels(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level = []
        level_size = len(queue)  # Important!
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
\`\`\`

**Common Variations:**

**Right Side View:**
\`\`\`python
def right_side_view(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Add rightmost node of each level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
\`\`\`

**Zigzag Level Order:**
\`\`\`python
def zigzag_level_order(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    left_to_right = True
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        if not left_to_right:
            level.reverse()
        
        result.append(level)
        left_to_right = not left_to_right
    
    return result
\`\`\``,
      codeExample: `from collections import deque
from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """Return level-order traversal as list of levels"""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


def min_depth(root: Optional[TreeNode]) -> int:
    """Find minimum depth using BFS - stops at first leaf"""
    if not root:
        return 0
    
    queue = deque([(root, 1)])
    
    while queue:
        node, depth = queue.popleft()
        
        # First leaf we encounter is at minimum depth
        if not node.left and not node.right:
            return depth
        
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return 0


def max_width(root: Optional[TreeNode]) -> int:
    """Find maximum width of tree using BFS"""
    if not root:
        return 0
    
    max_width = 0
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        max_width = max(max_width, level_size)
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return max_width`,
    },
    {
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

**Shortest Path in Unweighted Graph:**
\`\`\`python
def shortest_path(graph, start, end):
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
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # No path exists
\`\`\`

**Shortest Path with Parent Tracking:**
\`\`\`python
def shortest_path_with_route(graph, start, end):
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
                path.append(curr)
                curr = parent[curr]
            return path[::-1]
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)
    
    return []  # No path
\`\`\`

**Multi-Source BFS:**
\`\`\`python
def multi_source_bfs(graph, sources):
    """
    BFS starting from multiple sources simultaneously.
    Useful for problems like "distance from nearest X"
    """
    visited = set(sources)
    queue = deque([(s, 0) for s in sources])
    distances = {}
    
    while queue:
        node, dist = queue.popleft()
        distances[node] = dist
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
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
                    deque_bfs.appendleft(neighbor)  # 0-weight: front
                else:
                    deque_bfs.append(neighbor)      # 1-weight: back
    
    return -1
\`\`\``,
      codeExample: `from collections import deque
from typing import List, Dict, Set


def bfs_shortest_distance(graph: Dict, start: int, end: int) -> int:
    """Find shortest path distance using BFS"""
    if start == end:
        return 0
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor == end:
                return dist + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # No path exists


def bfs_all_distances(graph: Dict, start: int) -> Dict[int, int]:
    """Find shortest distance from start to all reachable nodes"""
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances


def bfs_grid(grid: List[List[int]], start: tuple, end: tuple) -> int:
    """BFS on 2D grid - find shortest path from start to end"""
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
                grid[nr][nc] != 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1  # No path


def bfs_connected_components(graph: Dict, n: int) -> int:
    """Count connected components using BFS"""
    visited = set()
    count = 0
    
    def bfs(start):
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    for node in range(n):
        if node not in visited:
            bfs(node)
            count += 1
    
    return count`,
    },
    {
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
    },
    {
      id: 'complexity',
      title: 'Time and Space Complexity',
      content: `**Time Complexity:**

**Tree BFS:**
- **O(N)** where N = number of nodes
- Visit each node exactly once
- Process each node in constant time (queue operations)

**Graph BFS:**
- **O(V + E)** where V = vertices, E = edges
- Visit each vertex once: O(V)
- Explore each edge once (or twice for undirected): O(E)

**Grid BFS:**
- **O(rows × cols)** 
- Each cell visited at most once
- Check 4 neighbors per cell

**Space Complexity:**

**Queue Space:**
- **O(W)** where W = maximum width
- Worst case: complete binary tree last level = N/2 nodes
- Can be O(V) for graphs

**Visited Set:**
- **O(V)** for graphs or O(N) for trees
- Must track all visited nodes

**Total Space:**
- Trees: O(W) for queue + O(N) for result
- Graphs: O(V) for queue + O(V) for visited
- Generally O(N) or O(V)

**BFS vs DFS Space:**
- **BFS**: O(W) - proportional to width
- **DFS**: O(H) - proportional to height
- For balanced trees: W > H, so DFS uses less space
- For skewed trees: H ≈ N, so BFS might be better`,
    },
    {
      id: 'patterns',
      title: 'Common BFS Patterns',
      content: `**Pattern 1: Level-by-Level Processing**
\`\`\`python
def level_order(root):
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)  # Key: capture size
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            # Add children
        
        result.append(level)
\`\`\`

**Pattern 2: Shortest Path / Minimum Steps**
\`\`\`python
def min_steps(start, end):
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        state, steps = queue.popleft()
        if state == end:
            return steps
        
        for next_state in get_neighbors(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, steps + 1))
\`\`\`

**Pattern 3: Multi-Source BFS**
\`\`\`python
def multi_source(grid, sources):
    queue = deque(sources)  # Start from all sources
    visited = set(sources)
    
    while queue:
        node = queue.popleft()
        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
\`\`\`

**Pattern 4: Bidirectional BFS**
\`\`\`python
def bidirectional_bfs(start, end):
    """Meet in the middle - faster for long paths"""
    if start == end:
        return 0
    
    front = {start}
    back = {end}
    distance = 0
    
    while front and back:
        distance += 1
        
        # Expand smaller frontier
        if len(front) > len(back):
            front, back = back, front
        
        next_front = set()
        for node in front:
            for neighbor in get_neighbors(node):
                if neighbor in back:
                    return distance
                if neighbor not in visited:
                    next_front.add(neighbor)
        
        front = next_front
\`\`\`

**Pattern 5: State Space BFS**
\`\`\`python
def min_moves(start_state):
    """BFS on implicit graph of states"""
    queue = deque([(start_state, 0)])
    visited = {start_state}
    
    while queue:
        state, moves = queue.popleft()
        
        if is_goal(state):
            return moves
        
        for next_state in generate_next_states(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, moves + 1))
\`\`\``,
    },
    {
      id: 'interview-strategy',
      title: 'Interview Strategy',
      content: `**Recognizing BFS Problems:**

**Keywords to watch for:**
- "Shortest path/distance"
- "Minimum number of steps/moves"
- "Level by level"
- "Nearest/closest"
- "Layer by layer"
- "Each turn/round"

**BFS vs DFS Decision:**

**Use BFS when:**
- Need **shortest path** (unweighted)
- **Level-order** traversal required
- Find **nearest/closest** element
- **Minimum steps** to reach goal
- Width likely < height

**Use DFS when:**
- Need to explore **all paths**
- **Tree traversal** (preorder/inorder/postorder)
- **Backtracking** problems
- Path finding (not shortest)
- Height likely < width

**Problem-Solving Framework:**

**1. Identify the Graph:**
- Explicit graph (adjacency list/matrix)?
- Implicit graph (grid, state space)?
- Tree or general graph?

**2. Determine What to Track:**
- Just visit nodes? → Simple BFS
- Count distance/steps? → Track distance
- Reconstruct path? → Track parent
- Process levels? → Track level size

**3. Choose Data Structures:**
- Queue: Standard BFS
- Deque: 0-1 BFS
- Set: Visited nodes
- Dict: Parent tracking / distances

**4. Handle Edge Cases:**
- Empty input
- Start = end
- No path exists
- Disconnected components

**Common Mistakes:**
- Forgetting visited set (infinite loop!)
- Not marking visited before adding to queue
- Incorrect level-by-level processing
- Using DFS when shortest path needed
- Not handling disconnected components

**Interview Tips:**
- Always mention time/space complexity
- Discuss BFS vs DFS trade-offs
- Draw small example and trace through
- Mention optimizations (bidirectional, 0-1)
- Handle edge cases explicitly`,
    },
  ],
  keyTakeaways: [
    'BFS explores level by level using a queue (FIFO), visiting nearest nodes first',
    'BFS finds shortest path in unweighted graphs - first visit guarantees shortest',
    'Time: O(V+E) for graphs, O(N) for trees; Space: O(W) for queue width',
    'Level-by-level processing: capture queue size before inner loop',
    'Multi-source BFS starts from multiple nodes simultaneously',
    'Use BFS for: shortest path, level-order traversal, minimum steps problems',
    'Always use visited set to avoid cycles in graphs',
    'Bidirectional BFS can reduce search space from O(b^d) to O(b^(d/2))',
  ],
  relatedProblems: [
    'binary-tree-level-order',
    'shortest-path-binary-matrix',
    'rotting-oranges',
  ],
};
