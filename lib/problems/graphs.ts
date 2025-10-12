import { Problem } from '../types';

export const graphsProblems: Problem[] = [
  {
    id: 'number-of-islands',
    title: 'Number of Islands',
    difficulty: 'Easy',
    description: `Given an \`m x n\` 2D binary grid \`grid\` which represents a map of \`'1'\`s (land) and \`'0'\`s (water), return **the number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**LeetCode:** [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
**YouTube:** [NeetCode - Number of Islands](https://www.youtube.com/watch?v=pV2kpPD66nE)

**Approach:**
This is a **connected components** problem. Use DFS or BFS to explore each island, marking visited cells. Each time we start a new DFS/BFS, we've found a new island.

**Key Insight:**
Think of the grid as an implicit graph where each land cell is connected to its adjacent land cells (up, down, left, right).`,
    examples: [
      {
        input:
          'grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]',
        output: '1',
        explanation: 'All the 1s are connected, forming one island.',
      },
      {
        input:
          'grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]',
        output: '3',
        explanation: 'There are three separate islands.',
      },
    ],
    constraints: [
      'm == grid.length',
      'n == grid[i].length',
      '1 <= m, n <= 300',
      'grid[i][j] is "0" or "1"',
    ],
    hints: [
      'Iterate through each cell in the grid',
      'When you find a "1" (unvisited land), increment island count',
      'Run DFS/BFS from that cell to mark all connected land as visited',
      'DFS will explore all 4 directions (up, down, left, right)',
      'Mark visited cells by changing "1" to "0" or use a visited set',
    ],
    starterCode: `from typing import List

def num_islands(grid: List[List[str]]) -> int:
    """
    Count the number of islands in a 2D grid.
    
    Args:
        grid: 2D grid of "1" (land) and "0" (water)
        
    Returns:
        Number of islands
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            ['1', '1', '1', '1', '0'],
            ['1', '1', '0', '1', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '0', '0', '0'],
          ],
        ],
        expected: 1,
      },
      {
        input: [
          [
            ['1', '1', '0', '0', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '1', '0', '0'],
            ['0', '0', '0', '1', '1'],
          ],
        ],
        expected: 3,
      },
      {
        input: [[['1']]],
        expected: 1,
      },
      {
        input: [[['0']]],
        expected: 0,
      },
    ],
    solution: `from typing import List


def num_islands(grid: List[List[str]]) -> int:
    """
    DFS approach (modifies grid in-place).
    Time: O(M * N), Space: O(M * N) for recursion
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        # Boundary checks
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0'):
            return
        
        # Mark as visited
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left
    
    # Iterate through all cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)  # Mark entire island
    
    return count


# Alternative: BFS approach
def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    BFS approach (doesn't modify input).
    Time: O(M * N), Space: O(min(M, N)) for queue
    """
    if not grid:
        return 0
    
    from collections import deque
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    count = 0
    
    def bfs(r, c):
        queue = deque([(r, c)])
        visited.add((r, c))
        
        while queue:
            row, col = queue.popleft()
            
            # Check all 4 directions
            for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                nr, nc = row + dr, col + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and
                    grid[nr][nc] == '1' and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                count += 1
                bfs(r, c)
    
    return count


# Alternative: Union-Find approach
def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    Union-Find approach.
    Time: O(M * N * α(M*N)), Space: O(M * N)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    class UnionFind:
        def __init__(self, grid):
            self.count = 0
            self.parent = {}
            self.rank = {}
            
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == '1':
                        key = r * cols + c
                        self.parent[key] = key
                        self.rank[key] = 0
                        self.count += 1
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            root_x, root_y = self.find(x), self.find(y)
            
            if root_x != root_y:
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                elif self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1
                self.count -= 1
    
    uf = UnionFind(grid)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                # Union with right and down neighbors
                if c + 1 < cols and grid[r][c + 1] == '1':
                    uf.union(r * cols + c, r * cols + c + 1)
                if r + 1 < rows and grid[r + 1][c] == '1':
                    uf.union(r * cols + c, (r + 1) * cols + c)
    
    return uf.count`,
    timeComplexity: 'O(M * N)',
    spaceComplexity: 'O(M * N) for recursion/visited set',
    order: 1,
    topic: 'Graphs',
    leetcodeUrl: 'https://leetcode.com/problems/number-of-islands/',
    youtubeUrl: 'https://www.youtube.com/watch?v=pV2kpPD66nE',
  },
  {
    id: 'course-schedule',
    title: 'Course Schedule',
    difficulty: 'Medium',
    description: `There are a total of \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you **must** take course \`bi\` first if you want to take course \`ai\`.

Return \`true\` if you can finish all courses. Otherwise, return \`false\`.

**LeetCode:** [207. Course Schedule](https://leetcode.com/problems/course-schedule/)
**YouTube:** [NeetCode - Course Schedule](https://www.youtube.com/watch?v=EgI5nU9etnU)

**Approach:**
This is a **cycle detection** problem in a directed graph. If there's a cycle in the dependency graph, it's impossible to complete all courses. Use DFS with a recursion stack or BFS with topological sort (Kahn's algorithm).

**Key Insight:**
- No cycle → Can complete all courses (DAG)
- Has cycle → Cannot complete all courses`,
    examples: [
      {
        input: 'numCourses = 2, prerequisites = [[1,0]]',
        output: 'true',
        explanation:
          'There are 2 courses. To take course 1 you should have finished course 0. So it is possible.',
      },
      {
        input: 'numCourses = 2, prerequisites = [[1,0],[0,1]]',
        output: 'false',
        explanation:
          'There are 2 courses. To take course 1 you need course 0, and to take course 0 you need course 1. Circular dependency!',
      },
    ],
    constraints: [
      '1 <= numCourses <= 2000',
      '0 <= prerequisites.length <= 5000',
      'prerequisites[i].length == 2',
      '0 <= ai, bi < numCourses',
      'All the pairs prerequisites[i] are unique',
    ],
    hints: [
      'Model this as a directed graph where edge a → b means "must take b before a"',
      'The problem becomes: does the graph have a cycle?',
      'DFS approach: use recursion stack to detect back edges (cycles)',
      "BFS approach: use topological sort (Kahn's) - if we process all nodes, no cycle",
      'Track three states: unvisited, visiting (in current path), visited',
    ],
    starterCode: `from typing import List

def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Determine if it's possible to finish all courses.
    
    Args:
        num_courses: Total number of courses
        prerequisites: List of [course, prerequisite] pairs
        
    Returns:
        True if possible to complete all courses, False otherwise
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [2, [[1, 0]]],
        expected: true,
      },
      {
        input: [
          2,
          [
            [1, 0],
            [0, 1],
          ],
        ],
        expected: false,
      },
      {
        input: [
          3,
          [
            [1, 0],
            [2, 1],
          ],
        ],
        expected: true,
      },
      {
        input: [
          4,
          [
            [1, 0],
            [2, 1],
            [3, 2],
            [1, 3],
          ],
        ],
        expected: false,
      },
    ],
    solution: `from typing import List
from collections import defaultdict, deque


def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    DFS with cycle detection.
    Time: O(V + E), Space: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Three states: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses
    
    def has_cycle(course):
        if state[course] == 1:  # Currently visiting = cycle
            return True
        if state[course] == 2:  # Already visited
            return False
        
        # Mark as visiting
        state[course] = 1
        
        # Check all prerequisites
        for prereq in graph[course]:
            if has_cycle(prereq):
                return True
        
        # Mark as visited
        state[course] = 2
        return False
    
    # Check each course
    for course in range(num_courses):
        if has_cycle(course):
            return False
    
    return True


# Alternative: Topological Sort (Kahn's Algorithm)
def can_finish_bfs(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    BFS with topological sort.
    Time: O(V + E), Space: O(V + E)
    """
    # Build graph and calculate in-degrees
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq → course
        in_degree[course] += 1
    
    # Start with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    processed = 0
    
    while queue:
        course = queue.popleft()
        processed += 1
        
        # Reduce in-degree of dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    # If we processed all courses, no cycle exists
    return processed == num_courses


# Alternative: DFS with visited set (simpler)
def can_finish_simple(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Simple DFS approach.
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    visited = set()
    rec_stack = set()
    
    def dfs(course):
        if course in rec_stack:
            return False  # Cycle detected
        if course in visited:
            return True  # Already checked
        
        rec_stack.add(course)
        
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        
        rec_stack.remove(course)
        visited.add(course)
        return True
    
    for course in range(num_courses):
        if course not in visited:
            if not dfs(course):
                return False
    
    return True`,
    timeComplexity: 'O(V + E) where V = courses, E = prerequisites',
    spaceComplexity: 'O(V + E)',
    order: 2,
    topic: 'Graphs',
    leetcodeUrl: 'https://leetcode.com/problems/course-schedule/',
    youtubeUrl: 'https://www.youtube.com/watch?v=EgI5nU9etnU',
  },
  {
    id: 'clone-graph',
    title: 'Clone Graph',
    difficulty: 'Hard',
    description: `Given a reference of a node in a **connected** undirected graph, return a **deep copy** (clone) of the graph.

Each node in the graph contains a value (\`int\`) and a list (\`List[Node]\`) of its neighbors.

\`\`\`
class Node {
    public int val;
    public List<Node> neighbors;
}
\`\`\`

**LeetCode:** [133. Clone Graph](https://leetcode.com/problems/clone-graph/)
**YouTube:** [NeetCode - Clone Graph](https://www.youtube.com/watch?v=mQeF6bN8hMk)

**Approach:**
Use DFS or BFS to traverse the graph. Maintain a hash map to track original → clone mappings to avoid creating duplicate clones and to handle cycles.

**Key Challenge:**
Handling the graph's connections correctly - must clone both nodes AND their relationships.`,
    examples: [
      {
        input: 'adjList = [[2,4],[1,3],[2,4],[1,3]]',
        output: '[[2,4],[1,3],[2,4],[1,3]]',
        explanation:
          'There are 4 nodes in the graph. Node 1 connects to nodes 2 and 4. Node 2 connects to nodes 1 and 3. Node 3 connects to nodes 2 and 4. Node 4 connects to nodes 1 and 3.',
      },
      {
        input: 'adjList = [[]]',
        output: '[[]]',
        explanation: 'The input contains a single node with no neighbors.',
      },
      {
        input: 'adjList = []',
        output: '[]',
        explanation: 'Empty graph.',
      },
    ],
    constraints: [
      'The number of nodes in the graph is in the range [0, 100]',
      '1 <= Node.val <= 100',
      'Node.val is unique for each node',
      'There are no repeated edges and no self-loops',
      'The Graph is connected and all nodes can be visited starting from the given node',
    ],
    hints: [
      'Use a hash map to store original node → cloned node mappings',
      'This prevents creating duplicate clones and handles cycles',
      'DFS/BFS through the graph, cloning each node and its neighbors',
      'When you encounter a node already in the map, return the clone',
      'Time: O(N + E) to visit all nodes and edges',
    ],
    starterCode: `from typing import Optional

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """
    Clone a graph using deep copy.
    
    Args:
        node: Reference node in the original graph
        
    Returns:
        Reference to the cloned graph
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [
              [2, 4],
              [1, 3],
              [2, 4],
              [1, 3],
            ],
          ],
        ],
        expected: [
          [2, 4],
          [1, 3],
          [2, 4],
          [1, 3],
        ],
      },
      {
        input: [[[[]]]],
        expected: [[]],
      },
      {
        input: [[[]]],
        expected: [],
      },
    ],
    solution: `from typing import Optional

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node: Optional[Node]) -> Optional[Node]:
    """
    DFS approach with hash map.
    Time: O(N + E), Space: O(N)
    """
    if not node:
        return None
    
    # Map original node -> cloned node
    clones = {}
    
    def dfs(node):
        # If already cloned, return the clone
        if node in clones:
            return clones[node]
        
        # Create clone
        clone = Node(node.val)
        clones[node] = clone
        
        # Clone all neighbors
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)


# Alternative: BFS approach
def clone_graph_bfs(node: Optional[Node]) -> Optional[Node]:
    """
    BFS approach with hash map.
    Time: O(N + E), Space: O(N)
    """
    if not node:
        return None
    
    from collections import deque
    
    # Map original -> clone
    clones = {node: Node(node.val)}
    queue = deque([node])
    
    while queue:
        curr = queue.popleft()
        
        # Process all neighbors
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                # Create clone
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            
            # Add neighbor to current node's clone
            clones[curr].neighbors.append(clones[neighbor])
    
    return clones[node]


# Alternative: One-liner using dictionary
def clone_graph_compact(node: Optional[Node]) -> Optional[Node]:
    """
    Compact recursive solution.
    """
    if not node:
        return None
    
    clones = {}
    
    def dfs(n):
        if n not in clones:
            clones[n] = Node(n.val)
            clones[n].neighbors = [dfs(nb) for nb in n.neighbors]
        return clones[n]
    
    return dfs(node)`,
    timeComplexity: 'O(N + E) where N = nodes, E = edges',
    spaceComplexity: 'O(N) for the hash map',
    order: 3,
    topic: 'Graphs',
    leetcodeUrl: 'https://leetcode.com/problems/clone-graph/',
    youtubeUrl: 'https://www.youtube.com/watch?v=mQeF6bN8hMk',
  },
];
