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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain BFS (Breadth-First Search). How does it work and what makes it "breadth-first"?',
          sampleAnswer:
            'BFS explores all nodes at current level before moving to next level. Uses queue (FIFO - First In First Out). Start at root/source, add to queue, loop: dequeue node, process, enqueue unvisited children. "Breadth-first" means explore wide (all neighbors) before deep. For tree [1, [2, [4, 5]], [3]]: BFS visits level 0: 1, level 1: 2,3, level 2: 4,5. Compare DFS: 1→2→4→5→3 (deep first). BFS guarantees shortest path in unweighted graphs - first time you reach node is via shortest path. Time O(V+E) for graphs, O(n) for trees. Space O(w) where w is maximum width. Natural for: level-order traversal, shortest path, nearest neighbors.',
          keyPoints: [
            'Explore level by level, uses queue (FIFO)',
            'Process all neighbors before going deeper',
            'Guarantees shortest path in unweighted',
            'O(V+E) time, O(w) space (width)',
            'Uses: level order, shortest path, nearest',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare BFS vs DFS fundamentally. What is the key algorithmic difference?',
          sampleAnswer:
            'Key difference: data structure. BFS uses queue (FIFO), DFS uses stack (LIFO). This determines exploration order. BFS: dequeue explores oldest added (level by level). DFS: pop explores newest added (deep first). For tree [1, [2, [4, 5]], [3]]: BFS queue [1]→[2,3]→[3,4,5]→[4,5]→[5]→[], visits 1,2,3,4,5. DFS stack [1]→[2,3]→[2,4,5]→[2,4]→[2]→[], visits 1,3,2,5,4 (preorder with right-first). BFS finds shortest path (first arrival), DFS explores deeply (may find longer paths first). Space: BFS O(width), DFS O(height). Implementation: BFS always iterative (queue), DFS recursive or iterative (stack).',
          keyPoints: [
            'BFS: queue (FIFO), DFS: stack (LIFO)',
            'BFS: level by level, DFS: deep first',
            'BFS: shortest path, DFS: explores deeply',
            'BFS space: O(width), DFS: O(height)',
            'BFS always iterative, DFS can be recursive',
          ],
        },
        {
          id: 'q3',
          question:
            'Why does BFS guarantee shortest path in unweighted graphs? Prove or explain intuition.',
          sampleAnswer:
            'BFS explores nodes in order of increasing distance from source. Queue ensures nodes at distance d are all processed before nodes at distance d+1. When node first discovered, it is via shortest path because all shorter paths explored already. Proof by contradiction: suppose node X reached at distance d+1, but shorter path of length d exists. Then node Y on that path at distance d-1 would have enqueued X at distance d, contradiction (X already discovered). For example, graph A→B, A→C, B→D, C→D: BFS from A discovers B,C at distance 1, then D at distance 2 via both paths. First discovery is shortest. DFS may find D via A→C→D before A→B→D, no guarantee. This is why BFS is standard for shortest path in unweighted graphs.',
          keyPoints: [
            'Explores in order of increasing distance',
            'Queue processes distance d before d+1',
            'First discovery = shortest path',
            'Proof: shorter path would discover earlier',
            'DFS no guarantee: may explore long paths first',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain level-order traversal using BFS. How does it differ from DFS traversals?',
          sampleAnswer:
            'Level-order traversal visits nodes level by level (all at depth 0, then depth 1, etc.). BFS natural fit: queue ensures level-by-level processing. Algorithm: queue with root, loop: process all nodes at current level (track level size), enqueue children for next level. For tree [1, [2, [4, 5]], [3]]: level 0: [1], level 1: [2, 3], level 2: [4, 5]. DFS traversals (preorder, inorder, postorder) visit in depth-first order, no level separation. For example, preorder 1,2,4,5,3 mixes levels. Level-order applications: print tree by levels, find level with maximum sum, zigzag traversal. The key: BFS queue naturally groups by level, DFS stack/recursion does not.',
          keyPoints: [
            'Visit nodes level by level',
            'BFS with queue: natural level grouping',
            'Track level size to process level at a time',
            'vs DFS: no level separation',
            'Uses: print levels, level properties, zigzag',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through finding right-side view of binary tree using BFS. What is the pattern?',
          sampleAnswer:
            'Right-side view: values visible from right side (rightmost node at each level). BFS pattern: level-order, at each level take last node. Algorithm: queue with root, loop: process level (level_size = len(queue)), for each node in level add children, after level add last node value to result. For tree [1, [2, [4, 5]], [3]]: level 0 last is 1, level 1 last is 3, level 2 last is 5. Result: [1, 3, 5]. The pattern: level-order + track last node per level. Can also do DFS: preorder, recurse right before left, track depth, add first node seen at each depth (first from right is rightmost). BFS more intuitive for level-based problems.',
          keyPoints: [
            'Right-side view: rightmost at each level',
            'BFS: process level, take last node',
            'Track level size to know when level ends',
            'Alternative DFS: right-first, depth tracking',
            'BFS more intuitive for level problems',
          ],
        },
        {
          id: 'q3',
          question:
            'Describe maximum width of binary tree using BFS. What is the challenge?',
          sampleAnswer:
            'Maximum width: max number of nodes at any level (including nulls between leftmost and rightmost). Challenge: need to track node positions to account for nulls. Algorithm: BFS with (node, position), left child position 2×pos, right child 2×pos+1. At each level, width = last_pos - first_pos + 1. For example, tree [1, [2, 4], 3]: level 0 width 1, level 1: node 2 at pos 0, node 3 at pos 1, width 2. Level 2: node 4 at pos 0 (2×0), width 1. Without position tracking, cannot account for missing nodes. Max width 2. The key: position indexing like array representation of binary tree, allows calculating width with gaps.',
          keyPoints: [
            'Width: includes nulls between leftmost and rightmost',
            'Track: (node, position) in queue',
            'Left: 2×pos, right: 2×pos+1',
            'Width per level: last_pos - first_pos + 1',
            'Accounts for missing nodes via positions',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain BFS on graphs. What must you track that trees do not require?',
          sampleAnswer:
            'Graph BFS requires visited set because graphs have cycles. Without visited, infinite loop: A→B→C→A→B... Tree BFS does not need visited (no cycles, clear parent-child). Graph BFS: queue with start, visited set, loop: dequeue node, for each unvisited neighbor: mark visited, enqueue. For example, graph 0→1, 1→2, 2→0, 0→3: start 0, queue=[0], visit 0, enqueue 1,3, queue=[1,3], visit 1, enqueue 2, queue=[3,2], visit 3, queue=[2], visit 2, try 0 (visited, skip). Mark visited WHEN ENQUEUING not when dequeuing (avoid duplicates in queue). Time O(V+E), space O(V) for visited and queue.',
          keyPoints: [
            'Graph needs visited set (cycles exist)',
            'Tree: no cycles, no visited needed',
            'Mark visited when enqueuing',
            'Prevents: infinite loops, duplicate queue entries',
            'O(V+E) time, O(V) space',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe connected components using BFS. How does it compare to DFS for this problem?',
          sampleAnswer:
            'Connected components: count disconnected groups. Algorithm: for each unvisited node, BFS to mark entire component, count++. For example, graph nodes {0,1,2,3,4,5}, edges 0-1, 2-3-4: BFS from 0 marks 0,1 (component 1), BFS from 2 marks 2,3,4 (component 2), node 5 alone (component 3). Total 3 components. BFS vs DFS for components: both O(V+E), both work equally well. BFS uses queue, iterative only. DFS uses stack or recursion, simpler code. Choice does not matter for correctness or complexity. Preference: DFS slightly simpler (recursive), but BFS fine too. This is one problem where BFS and DFS are equivalent.',
          keyPoints: [
            'Count: BFS from each unvisited node',
            'Each BFS marks one component',
            'BFS vs DFS: both O(V+E), equivalent',
            'BFS: queue, iterative. DFS: recursive',
            'Either works, DFS slightly simpler',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through BFS on undirected vs directed graphs. What changes?',
          sampleAnswer:
            'Undirected: edge A-B means A→B and B→A. BFS explores both directions. For cycle detection, need to track parent to avoid revisiting immediate predecessor. Directed: edge A→B only allows A to B. BFS explores one direction only. Example undirected: 0-1-2, BFS from 0: visit 0, enqueue 1, visit 1, try 0 (parent, skip) enqueue 2, visit 2, try 1 (visited, skip). Example directed: 0→1→2, same but 2 cannot reach 1 (no back edge). For shortest path, both work same way. For cycle detection, directed needs three states (visiting/visited), undirected needs parent tracking. BFS complexity same for both: O(V+E). The key: undirected treats each edge as bidirectional.',
          keyPoints: [
            'Undirected: edge A-B is both directions',
            'BFS explores both ways, track parent',
            'Directed: edge A→B is one direction only',
            'Undirected cycle: needs parent tracking',
            'Both: O(V+E), BFS works same',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Explain why BFS finds shortest path in unweighted graphs. Walk through an example.',
          sampleAnswer:
            'BFS guarantees shortest path because it explores nodes in order of increasing distance. First time a node is reached is via shortest path (all shorter paths already explored). Example graph: A→B, A→C, B→D, C→D, D→E. Shortest A to E? BFS: start A (dist 0), visit B,C (dist 1), visit D (dist 2, first reach via B and C both distance 2), visit E (dist 3). Path A→B→D→E or A→C→D→E both length 3. Cannot be shorter because all paths of length < 3 already explored. DFS might find A→C→D→E first, but could also explore A→B→C→... longer path. BFS level-by-level ensures optimality.',
          keyPoints: [
            'Explores in order of increasing distance',
            'First arrival = shortest path',
            'All shorter paths explored already',
            'Example: finds distance 3 before exploring deeper',
            'DFS no guarantee: may explore long first',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe path reconstruction in BFS. How do you track the actual path, not just distance?',
          sampleAnswer:
            'Track parent array: parent[v] = u means we reached v from u. During BFS, when enqueuing neighbor v from u, set parent[v] = u. After BFS finishes, backtrack from destination to source using parent array, then reverse. Example A→B→D→E: parent[B]=A, parent[D]=B, parent[E]=D. Backtrack from E: E→D→B→A, reverse to A→B→D→E. Without parent array, only know distance 3, not which nodes. Alternative: store paths in queue (each node stores full path to reach it), but uses O(V²) space vs O(V) for parent. Parent array is standard, space-efficient way. Can also track distance alongside parent for both metrics.',
          keyPoints: [
            'Parent array: parent[v] = u (reached v from u)',
            'Update parent when enqueuing neighbor',
            'Backtrack from dest to source, then reverse',
            'O(V) space vs O(V²) for full paths',
            'Standard pattern for BFS path reconstruction',
          ],
        },
        {
          id: 'q3',
          question:
            'Compare shortest path: BFS vs Dijkstra vs Bellman-Ford. When to use each?',
          sampleAnswer:
            'BFS: O(V+E), unweighted graphs only. Finds shortest path by counting edges. Use when: all edges weight 1 or equal, simplest and fastest. Dijkstra: O((V+E) log V), weighted non-negative graphs. Finds shortest path by summing weights. Use when: edges have different positive weights, need actual distance. Bellman-Ford: O(VE), weighted with possible negative weights. Use when: negative weights exist, need negative cycle detection. For unweighted: always BFS. For weighted positive: always Dijkstra. For negative: Bellman-Ford. Never use Dijkstra with negative weights (fails). Example: social network distance (unweighted) → BFS. Road network with distances (positive weights) → Dijkstra. Currency exchange (negative edges possible) → Bellman-Ford.',
          keyPoints: [
            'BFS: O(V+E) unweighted',
            'Dijkstra: O((V+E) log V) weighted non-negative',
            'Bellman-Ford: O(VE) negative weights',
            'BFS fastest for unweighted',
            'Hierarchy: BFS → Dijkstra → Bellman-Ford',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'Analyze BFS time and space complexity for trees vs graphs.',
          sampleAnswer:
            'Trees: Time O(n) visit each node once. Space O(w) where w is maximum width. Balanced binary tree: width at bottom level is n/2, so O(n) space worst case. Graphs: Time O(V+E) visit each vertex once, explore each edge once (or twice for undirected). Space O(V) for queue and visited set. For example, complete binary tree 1M nodes: width at level log(1M) ≈ 500K, space O(500K). Graph 1000 vertices, 5000 edges: time O(6000), space O(1000). BFS space often larger than DFS O(height). For balanced tree height log n << width n/2. For skewed tree (linked list): height n, width 1, BFS better than DFS.',
          keyPoints: [
            'Trees: O(n) time, O(w) space (width)',
            'Graphs: O(V+E) time, O(V) space',
            'Balanced tree: width n/2, space O(n)',
            'Skewed tree: width 1, space O(1)',
            'BFS space often > DFS for balanced trees',
          ],
        },
        {
          id: 'q2',
          question:
            'Compare BFS vs DFS space for different tree shapes. Which is more space-efficient?',
          sampleAnswer:
            'Balanced tree: BFS O(w) = O(n/2) = O(n), DFS O(h) = O(log n). DFS more space-efficient. Skewed tree (linked list): BFS O(w) = O(1), DFS O(h) = O(n). BFS more space-efficient. Complete binary tree: width doubles each level, bottom has n/2 nodes. BFS O(n), DFS O(log n). DFS much better. For example, 1M node balanced tree: BFS uses 500K space, DFS uses 20 space (log 1M ≈ 20). For 1M node linked list: BFS uses 1 space, DFS uses 1M space. General rule: balanced/wide trees favor DFS, skewed/narrow trees favor BFS. Most real trees are balanced, so DFS usually more space-efficient.',
          keyPoints: [
            'Balanced: BFS O(n), DFS O(log n) - DFS better',
            'Skewed: BFS O(1), DFS O(n) - BFS better',
            'Complete: BFS O(n/2), DFS O(log n) - DFS much better',
            'Wide trees: DFS better, narrow: BFS better',
            'Most real trees balanced: DFS more efficient',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the queue space in BFS. Why does it grow to width of tree?',
          sampleAnswer:
            'BFS queue holds all nodes at current level before processing next level. Maximum size is width of widest level. For complete binary tree, widest level is bottom (n/2 nodes). Process level k: queue has all 2^k nodes at level k. After processing, queue has all 2^(k+1) nodes at level k+1. Peak queue size occurs at widest level. For example, tree with 4 levels (1+2+4+8=15 nodes): level 0 queue size 1, level 1 size 2, level 2 size 4, level 3 size 8 (peak). The nodes must all wait in queue before being processed. This is why BFS space is O(width) not O(depth). DFS uses stack, only one path active at a time, so O(depth).',
          keyPoints: [
            'Queue holds all nodes at current level',
            'Max size = widest level width',
            'Complete tree: widest is bottom (n/2)',
            'All nodes wait before processing',
            'vs DFS: only one path (depth)',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'What are common BFS patterns? How do you recognize when to use each?',
          sampleAnswer:
            'Pattern 1: Level-order traversal - recognize by "level", "depth", "row by row". Pattern 2: Shortest path - recognize by "minimum steps", "shortest", unweighted graph. Pattern 3: Nearest/closest - recognize by "nearest", "closest", "minimum distance from point". Pattern 4: Multi-source BFS - recognize by multiple start points, "distance from any". Pattern 5: State-space search - recognize by transformations, puzzle, "minimum moves". For example, "print tree level by level" → level-order pattern. "Shortest path from A to B" → shortest path pattern. "01 Matrix: distance to nearest 0" → multi-source BFS from all 0s. "Minimum moves to solve sliding puzzle" → state-space BFS.',
          keyPoints: [
            'Level-order: by level traversal',
            'Shortest path: minimum steps, unweighted',
            'Nearest: closest node/cell',
            'Multi-source: multiple starts',
            'State-space: transformations, puzzles',
          ],
        },
        {
          id: 'q2',
          question:
            'Describe multi-source BFS. How does it differ from single-source?',
          sampleAnswer:
            'Multi-source BFS: start from multiple sources simultaneously, find distances to all. Initialize queue with all sources (distance 0), mark all as visited, run standard BFS. All sources explored at same "level". Example: "01 Matrix - distance to nearest 0": instead of BFS from each cell to find nearest 0 (O(n² × n²)), start BFS from all 0s at once. Queue initially has all 0 cells (distance 0), then explores all cells distance 1 from any 0, then distance 2, etc. Each cell reached first time is at correct minimum distance. Time O(rows×cols) once vs O((rows×cols)²) for single-source from each cell. Pattern: "distance to nearest X" where X is multiple cells.',
          keyPoints: [
            'Start BFS from multiple sources simultaneously',
            'Initialize queue with all sources',
            'All sources at distance 0',
            'Example: distance to nearest 0 in grid',
            'O(V+E) once vs O(V×(V+E)) for V single-source',
          ],
        },
        {
          id: 'q3',
          question:
            'Walk me through solving "Minimum Knight Moves" with BFS. Why is BFS the right choice?',
          sampleAnswer:
            'Minimum knight moves: chess knight from (0,0) to (x,y), minimum moves? BFS because: unweighted graph (each move costs 1), need shortest path. State space: positions (row, col). Algorithm: queue with (0, 0, 0) (start position, moves=0), visited set, BFS: dequeue (r, c, moves), if target return moves, for each of 8 knight moves: new position (r+dr, c+dc), if not visited: mark visited, enqueue (new_r, new_c, moves+1). First reach of target is minimum moves. For example, (0,0) to (2,1): (0,0)→(1,2) or (2,1) both 1 move, BFS finds immediately. vs DFS: might explore (0,0)→(1,2)→(3,1)→... longer path first. BFS guarantees minimum.',
          keyPoints: [
            'State space: board positions',
            'BFS: unweighted, finds shortest',
            'Queue: (position, moves)',
            '8 knight moves from each position',
            'First reach = minimum moves',
          ],
        },
      ],
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
      quiz: [
        {
          id: 'q1',
          question:
            'How do you recognize a BFS problem? What keywords and patterns signal BFS?',
          sampleAnswer:
            'Keywords: "shortest path", "minimum steps", "level", "nearest", "closest", "minimum distance", "level-order". Patterns: 1) Shortest path in unweighted graph. 2) Level-by-level traversal. 3) Minimum moves/steps. 4) Nearest neighbor. 5) Multi-source exploration. Signals: unweighted graph + shortest, tree + level-order, minimum steps in state space. For example, "shortest path maze" → BFS (each move costs 1). "Binary tree level order" → BFS natural fit. "Minimum moves to solve puzzle" → BFS on state graph. If "shortest" with weighted → Dijkstra not BFS. If exploring all paths → DFS. If shortest unweighted → always BFS.',
          keyPoints: [
            'Keywords: shortest, minimum, level, nearest',
            'Patterns: shortest path, level order, min steps',
            'Unweighted + shortest → BFS',
            'Weighted + shortest → Dijkstra',
            'All paths → DFS',
          ],
        },
        {
          id: 'q2',
          question:
            'Walk me through your BFS interview approach from recognition to implementation.',
          sampleAnswer:
            'First, recognize BFS from keywords (shortest, level, minimum steps). Second, identify if graph/tree/state-space (positions, configurations). Third, define state representation (node, position, configuration). Fourth, determine what to track (distance, parent, level). Fifth, initialize: queue with start, visited set, mark start visited. Sixth, BFS loop: dequeue, check if target, enqueue unvisited neighbors with updated distance. Seventh, test with examples and edges. Finally, analyze complexity O(V+E) or O(states). For example, "word ladder": recognize minimum steps (BFS), state is word, neighbors are one-letter changes, queue with start word, BFS until reaching end word, O(words × word_length).',
          keyPoints: [
            'Recognize: shortest, level, minimum',
            'Identify: graph/tree/state-space',
            'Define: state representation',
            'Track: distance, parent, level',
            'Initialize: queue, visited, mark start',
            'Test and analyze complexity',
          ],
        },
        {
          id: 'q3',
          question:
            'What are the most common mistakes in BFS problems? How do you avoid them?',
          sampleAnswer:
            'First: using stack instead of queue (becomes DFS). Second: marking visited when dequeuing not enqueuing (duplicates in queue, higher space/time). Third: forgetting to mark start visited. Fourth: not handling disconnected graphs (only explore from one component). Fifth: not tracking level when problem needs it. Sixth: using BFS for weighted graphs (need Dijkstra). My strategy: 1) Always use queue (collections.deque in Python). 2) Mark visited when enqueuing neighbor. 3) Mark start before loop. 4) For components, loop through all unvisited nodes. 5) Track level: count nodes per level or store (node, level) in queue. 6) Unweighted only. Test: disconnected, single node, already visited start.',
          keyPoints: [
            'Mistakes: stack (DFS), visited timing, no start mark',
            'Mark visited when enqueuing not dequeuing',
            'Use queue (deque), not stack',
            'Disconnected: loop all unvisited',
            'Track level if needed',
            'Unweighted only (else Dijkstra)',
          ],
        },
      ],
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
