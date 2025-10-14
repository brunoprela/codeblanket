import { Problem } from '../types';

export const graphsProblems: Problem[] = [
  {
    id: 'number-of-islands',
    title: 'Number of Islands',
    difficulty: 'Easy',
    description: `Given an \`m x n\` 2D binary grid \`grid\` which represents a map of \`'1'\`s (land) and \`'0'\`s (water), return **the number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.


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

    leetcodeUrl: 'https://leetcode.com/problems/number-of-islands/',
    youtubeUrl: 'https://www.youtube.com/watch?v=pV2kpPD66nE',
    order: 1,
    topic: 'Graphs',
  },
  {
    id: 'course-schedule',
    title: 'Course Schedule',
    difficulty: 'Medium',
    description: `There are a total of \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you **must** take course \`bi\` first if you want to take course \`ai\`.

Return \`true\` if you can finish all courses. Otherwise, return \`false\`.


**Approach:**
This is a **cycle detection** problem in a directed graph. If there is a cycle in the dependency graph, it is impossible to complete all courses. Use DFS with a recursion stack or BFS with topological sort (Kahn algorithm).

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

    leetcodeUrl: 'https://leetcode.com/problems/course-schedule/',
    youtubeUrl: 'https://www.youtube.com/watch?v=EgI5nU9etnU',
    order: 2,
    topic: 'Graphs',
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

    leetcodeUrl: 'https://leetcode.com/problems/clone-graph/',
    youtubeUrl: 'https://www.youtube.com/watch?v=mQeF6bN8hMk',
    order: 3,
    topic: 'Graphs',
  },

  // EASY - Find if Path Exists in Graph
  {
    id: 'find-path-exists-graph',
    title: 'Find if Path Exists in Graph',
    difficulty: 'Easy',
    topic: 'Graphs',
    description: `There is a **bi-directional** graph with \`n\` vertices, where each vertex is labeled from \`0\` to \`n - 1\` (**inclusive**). The edges in the graph are represented as a 2D integer array \`edges\`, where each \`edges[i] = [ui, vi]\` denotes a bi-directional edge between vertex \`ui\` and vertex \`vi\`. Every vertex pair is connected by **at most one** edge, and no vertex has an edge to itself.

You want to determine if there is a **valid path** that exists from vertex \`source\` to vertex \`destination\`.

Given \`edges\` and the integers \`n\`, \`source\`, and \`destination\`, return \`true\` if there is a **valid path** from \`source\` to \`destination\`, or \`false\` otherwise.`,
    examples: [
      {
        input:
          'n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2',
        output: 'true',
        explanation:
          'There are two paths from vertex 0 to vertex 2: 0 → 1 → 2 and 0 → 2',
      },
      {
        input:
          'n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5',
        output: 'false',
        explanation: 'There is no path from vertex 0 to vertex 5.',
      },
    ],
    constraints: [
      '1 <= n <= 2 * 10^5',
      '0 <= edges.length <= 2 * 10^5',
      'edges[i].length == 2',
      '0 <= ui, vi <= n - 1',
      'ui != vi',
      '0 <= source, destination <= n - 1',
      'There are no duplicate edges',
      'There are no self edges',
    ],
    hints: ['Build adjacency list from edges', 'Use BFS or DFS to find path'],
    starterCode: `from typing import List
from collections import deque, defaultdict

def valid_path(n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    """
    Check if path exists from source to destination.
    
    Args:
        n: Number of vertices
        edges: List of edges
        source: Starting vertex
        destination: Target vertex
        
    Returns:
        True if path exists
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          3,
          [
            [0, 1],
            [1, 2],
            [2, 0],
          ],
          0,
          2,
        ],
        expected: true,
      },
      {
        input: [
          6,
          [
            [0, 1],
            [0, 2],
            [3, 5],
            [5, 4],
            [4, 3],
          ],
          0,
          5,
        ],
        expected: false,
      },
      {
        input: [1, [], 0, 0],
        expected: true,
      },
    ],
    timeComplexity: 'O(V + E)',
    spaceComplexity: 'O(V + E)',
    leetcodeUrl: 'https://leetcode.com/problems/find-if-path-exists-in-graph/',
    youtubeUrl: 'https://www.youtube.com/watch?v=muncqlKJrH0',
  },

  // EASY - Find Center of Star Graph
  {
    id: 'find-center-star-graph',
    title: 'Find Center of Star Graph',
    difficulty: 'Easy',
    topic: 'Graphs',
    description: `There is an undirected **star** graph consisting of \`n\` nodes labeled from \`1\` to \`n\`. A star graph is a graph where there is one **center** node and **exactly** \`n - 1\` edges that connect the center node with every other node.

You are given a 2D integer array \`edges\` where each \`edges[i] = [ui, vi]\` indicates that there is an edge between the nodes \`ui\` and \`vi\`. Return the center of the given star graph.`,
    examples: [
      {
        input: 'edges = [[1,2],[2,3],[4,2]]',
        output: '2',
        explanation:
          'Node 2 is connected to every other node, so 2 is the center.',
      },
      {
        input: 'edges = [[1,2],[5,1],[1,3],[1,4]]',
        output: '1',
      },
    ],
    constraints: [
      '3 <= n <= 10^5',
      'edges.length == n - 1',
      'edges[i].length == 2',
      '1 <= ui, vi <= n',
      'ui != vi',
      'The given edges represent a valid star graph',
    ],
    hints: ['The center appears in every edge', 'Check first two edges'],
    starterCode: `from typing import List

def find_center(edges: List[List[int]]) -> int:
    """
    Find center node of star graph.
    
    Args:
        edges: List of edges in star graph
        
    Returns:
        Center node number
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2],
            [2, 3],
            [4, 2],
          ],
        ],
        expected: 2,
      },
      {
        input: [
          [
            [1, 2],
            [5, 1],
            [1, 3],
            [1, 4],
          ],
        ],
        expected: 1,
      },
    ],
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/find-center-of-star-graph/',
    youtubeUrl: 'https://www.youtube.com/watch?v=KFVweOZyY0I',
  },

  // EASY - Find the Town Judge
  {
    id: 'find-town-judge',
    title: 'Find the Town Judge',
    difficulty: 'Easy',
    topic: 'Graphs',
    description: `In a town, there are \`n\` people labeled from \`1\` to \`n\`. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

1. The town judge trusts nobody.
2. Everybody (except for the town judge) trusts the town judge.
3. There is exactly one person that satisfies properties 1 and 2.

You are given an array \`trust\` where \`trust[i] = [ai, bi]\` representing that the person labeled \`ai\` trusts the person labeled \`bi\`. If a trust relationship does not exist in \`trust\` array, then such a trust relationship does not exist.

Return the label of the town judge if the town judge exists and can be identified, or return \`-1\` otherwise.`,
    examples: [
      {
        input: 'n = 2, trust = [[1,2]]',
        output: '2',
      },
      {
        input: 'n = 3, trust = [[1,3],[2,3]]',
        output: '3',
      },
      {
        input: 'n = 3, trust = [[1,3],[2,3],[3,1]]',
        output: '-1',
      },
    ],
    constraints: [
      '1 <= n <= 1000',
      '0 <= trust.length <= 10^4',
      'trust[i].length == 2',
      'All the pairs of trust are unique',
      'ai != bi',
      '1 <= ai, bi <= n',
    ],
    hints: [
      'Count in-degree and out-degree for each person',
      'Judge has in-degree = n-1, out-degree = 0',
    ],
    starterCode: `from typing import List

def find_judge(n: int, trust: List[List[int]]) -> int:
    """
    Find the town judge.
    
    Args:
        n: Number of people
        trust: Trust relationships
        
    Returns:
        Judge label or -1
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [2, [[1, 2]]],
        expected: 2,
      },
      {
        input: [
          3,
          [
            [1, 3],
            [2, 3],
          ],
        ],
        expected: 3,
      },
      {
        input: [
          3,
          [
            [1, 3],
            [2, 3],
            [3, 1],
          ],
        ],
        expected: -1,
      },
    ],
    timeComplexity: 'O(E)',
    spaceComplexity: 'O(N)',
    leetcodeUrl: 'https://leetcode.com/problems/find-the-town-judge/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ZwFjogT5HuQ',
  },

  // MEDIUM - All Paths From Source to Target
  {
    id: 'all-paths-source-target',
    title: 'All Paths From Source to Target',
    difficulty: 'Medium',
    topic: 'Graphs',
    description: `Given a directed acyclic graph (**DAG**) of \`n\` nodes labeled from \`0\` to \`n - 1\`, find all possible paths from node \`0\` to node \`n - 1\` and return them in **any order**.

The graph is given as follows: \`graph[i]\` is a list of all nodes you can visit from node \`i\` (i.e., there is a directed edge from node \`i\` to node \`graph[i][j]\`).`,
    examples: [
      {
        input: 'graph = [[1,2],[3],[3],[]]',
        output: '[[0,1,3],[0,2,3]]',
        explanation: 'There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.',
      },
      {
        input: 'graph = [[4,3,1],[3,2,4],[3],[4],[]]',
        output: '[[0,4],[0,3,4],[0,1,3,4],[0,1,2,3,4],[0,1,4]]',
      },
    ],
    constraints: [
      'n == graph.length',
      '2 <= n <= 15',
      '0 <= graph[i][j] < n',
      'graph[i][j] != i (no self-loops)',
      'All the elements of graph[i] are unique',
      'The input graph is guaranteed to be a DAG',
    ],
    hints: [
      'Use DFS backtracking',
      'Track current path',
      'Add path to result when reaching target',
    ],
    starterCode: `from typing import List

def all_paths_source_target(graph: List[List[int]]) -> List[List[int]]:
    """
    Find all paths from source (0) to target (n-1).
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        List of all paths
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[1, 2], [3], [3], []]],
        expected: [
          [0, 1, 3],
          [0, 2, 3],
        ],
      },
      {
        input: [[[4, 3, 1], [3, 2, 4], [3], [4], []]],
        expected: [
          [0, 4],
          [0, 3, 4],
          [0, 1, 3, 4],
          [0, 1, 2, 3, 4],
          [0, 1, 4],
        ],
      },
    ],
    timeComplexity: 'O(2^N * N)',
    spaceComplexity: 'O(N)',
    leetcodeUrl:
      'https://leetcode.com/problems/all-paths-from-source-to-target/',
    youtubeUrl: 'https://www.youtube.com/watch?v=bSfxLRBXQPU',
  },

  // MEDIUM - Keys and Rooms
  {
    id: 'keys-and-rooms',
    title: 'Keys and Rooms',
    difficulty: 'Medium',
    topic: 'Graphs',
    description: `There are \`n\` rooms labeled from \`0\` to \`n - 1\` and all the rooms are locked except for room \`0\`. Your goal is to visit all the rooms. However, you cannot enter a locked room without having its key.

When you visit a room, you may find a set of **distinct keys** in it. Each key has a number on it, denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

Given an array \`rooms\` where \`rooms[i]\` is the set of keys that you can obtain if you visited room \`i\`, return \`true\` if you can visit **all** the rooms, or \`false\` otherwise.`,
    examples: [
      {
        input: 'rooms = [[1],[2],[3],[]]',
        output: 'true',
        explanation:
          'Start in room 0, pick up key 1. Go to room 1, pick up key 2. Go to room 2, pick up key 3. Go to room 3. All rooms visited.',
      },
      {
        input: 'rooms = [[1,3],[3,0,1],[2],[0]]',
        output: 'false',
        explanation:
          'Cannot enter room 2 since the only key that unlocks it is in that room.',
      },
    ],
    constraints: [
      'n == rooms.length',
      '2 <= n <= 1000',
      '0 <= rooms[i].length <= 1000',
      '1 <= sum(rooms[i].length) <= 3000',
      '0 <= rooms[i][j] < n',
      'All the values of rooms[i] are unique',
    ],
    hints: [
      'Use DFS or BFS starting from room 0',
      'Track visited rooms',
      'Check if all rooms were visited',
    ],
    starterCode: `from typing import List

def can_visit_all_rooms(rooms: List[List[int]]) -> bool:
    """
    Check if all rooms can be visited.
    
    Args:
        rooms: List of keys in each room
        
    Returns:
        True if all rooms can be visited
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[[1], [2], [3], []]],
        expected: true,
      },
      {
        input: [[[1, 3], [3, 0, 1], [2], [0]]],
        expected: false,
      },
    ],
    timeComplexity: 'O(N + E)',
    spaceComplexity: 'O(N)',
    leetcodeUrl: 'https://leetcode.com/problems/keys-and-rooms/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ye_7c2K0Ark',
  },

  // MEDIUM - Course Schedule II (Topological Sort)
  {
    id: 'course-schedule-ii',
    title: 'Course Schedule II',
    difficulty: 'Medium',
    topic: 'Graphs',
    description: `There are a total of \`numCourses\` courses you have to take, labeled from \`0\` to \`numCourses - 1\`. You are given an array \`prerequisites\` where \`prerequisites[i] = [ai, bi]\` indicates that you **must** take course \`bi\` first if you want to take course \`ai\`.

For example, the pair \`[0, 1]\`, indicates that to take course \`0\` you have to first take course \`1\`.

Return **the ordering of courses you should take to finish all courses**. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.`,
    examples: [
      {
        input: 'numCourses = 2, prerequisites = [[1,0]]',
        output: '[0,1]',
        explanation:
          'There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].',
      },
      {
        input: 'numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]',
        output: '[0,2,1,3]',
        explanation:
          'There are a total of 4 courses. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after course 0. So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].',
      },
      {
        input: 'numCourses = 1, prerequisites = []',
        output: '[0]',
        explanation: 'There is only one course, so the answer is [0].',
      },
    ],
    constraints: [
      '1 <= numCourses <= 2000',
      '0 <= prerequisites.length <= numCourses * (numCourses - 1)',
      'prerequisites[i].length == 2',
      '0 <= ai, bi < numCourses',
      'ai != bi',
      'All the pairs [ai, bi] are distinct',
    ],
    hints: [
      "Use topological sort - Kahn's algorithm with BFS",
      'Build adjacency list and track in-degrees',
      'Start with nodes that have in-degree 0',
      'If ordering includes all nodes, return it; otherwise return []',
    ],
    starterCode: `from typing import List
from collections import deque, defaultdict

def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Return the ordering of courses using topological sort.
    
    Args:
        num_courses: Total number of courses
        prerequisites: List of [course, prerequisite] pairs
        
    Returns:
        Valid course ordering, or [] if impossible
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [2, [[1, 0]]],
        expected: [0, 1],
      },
      {
        input: [
          4,
          [
            [1, 0],
            [2, 0],
            [3, 1],
            [3, 2],
          ],
        ],
        expected: [0, 2, 1, 3], // or [0, 1, 2, 3]
      },
      {
        input: [1, []],
        expected: [0],
      },
    ],
    timeComplexity: 'O(V + E)',
    spaceComplexity: 'O(V + E)',
    solution: `from typing import List
from collections import deque, defaultdict

def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Course Schedule II - Topological Sort using Kahn's Algorithm (BFS).
    
    Time: O(V + E) where V = numCourses, E = prerequisites
    Space: O(V + E) for adjacency list and in-degree array
    
    Key Insight:
    - This is topological sort - order nodes so all edges point forward
    - Use Kahn's algorithm: start with nodes that have no dependencies (in-degree 0)
    - Process each node, reduce in-degrees of neighbors
    - If we process all nodes, return ordering; otherwise cycle exists
    
    Difference from Course Schedule I:
    - Course Schedule I: just check if possible (return True/False)
    - Course Schedule II: return the actual ordering
    """
    # Build adjacency list and in-degree array
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)  # prereq -> course
        in_degree[course] += 1
    
    # Queue: all courses with no prerequisites (in-degree 0)
    queue = deque()
    for course in range(num_courses):
        if in_degree[course] == 0:
            queue.append(course)
    
    # Topological sort
    ordering = []
    
    while queue:
        # Take a course with no prerequisites
        course = queue.popleft()
        ordering.append(course)
        
        # Reduce in-degree for courses that depend on this one
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            
            # If all prerequisites satisfied, add to queue
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    # Check if we processed all courses
    if len(ordering) == num_courses:
        return ordering
    else:
        return []  # Cycle detected


# Alternative: DFS-based topological sort
def find_order_dfs(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    DFS-based topological sort.
    
    Time: O(V + E)
    Space: O(V + E)
    
    Strategy:
    - DFS from each unvisited node
    - Add to result in post-order (after visiting all descendants)
    - Reverse at end to get correct ordering
    - Detect cycles during DFS
    """
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0 = unvisited, 1 = visiting (in current path), 2 = visited
    state = [0] * num_courses
    ordering = []
    has_cycle = False
    
    def dfs(course):
        nonlocal has_cycle
        
        if state[course] == 1:
            # Cycle detected (found node in current path)
            has_cycle = True
            return
        
        if state[course] == 2:
            # Already processed
            return
        
        # Mark as visiting
        state[course] = 1
        
        # Visit all neighbors
        for next_course in graph[course]:
            dfs(next_course)
            if has_cycle:
                return
        
        # Mark as visited (done processing)
        state[course] = 2
        
        # Add to ordering in post-order
        ordering.append(course)
    
    # DFS from all nodes
    for course in range(num_courses):
        if state[course] == 0:
            dfs(course)
            if has_cycle:
                return []
    
    # Reverse to get correct topological order
    return ordering[::-1]


# Example walkthrough:
"""
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]

Graph:
0 → 1
0 → 2
1 → 3
2 → 3

In-degrees:
0: 0 (no prerequisites)
1: 1 (needs 0)
2: 1 (needs 0)
3: 2 (needs 1 and 2)

Process:
1. Start with in-degree 0: queue = [0]
2. Process 0: ordering = [0]
   - Reduce in-degree of 1 and 2
   - in-degree[1] = 0, in-degree[2] = 0
   - queue = [1, 2]

3. Process 1: ordering = [0, 1]
   - Reduce in-degree of 3
   - in-degree[3] = 1
   - queue = [2]

4. Process 2: ordering = [0, 1, 2]
   - Reduce in-degree of 3
   - in-degree[3] = 0
   - queue = [3]

5. Process 3: ordering = [0, 1, 2, 3]
   - queue = []

Result: [0, 1, 2, 3] (or [0, 2, 1, 3] depending on queue order)
"""


# Comparison: BFS (Kahn's) vs DFS
"""
BFS (Kahn's Algorithm):
+ More intuitive for beginners
+ Natural for "level" processing
+ Easier to explain in interviews
- Requires in-degree array
- More space

DFS:
+ No need for in-degree tracking
+ More elegant code
+ Natural recursion
- Harder to explain cycle detection
- State management needed

Both have same complexity: O(V + E) time and space
"""`,
    leetcodeUrl: 'https://leetcode.com/problems/course-schedule-ii/',
    youtubeUrl: 'https://www.youtube.com/watch?v=Akt3glAwyfY',
  },

  // HARD - Alien Dictionary (Topological Sort)
  {
    id: 'alien-dictionary',
    title: 'Alien Dictionary',
    difficulty: 'Hard',
    topic: 'Graphs',
    description: `There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings \`words\` from the alien language's dictionary, where the strings in \`words\` are **sorted lexicographically** by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in **lexicographically increasing order** by the new language's rules. If there is no solution, return \`""\`. If there are multiple solutions, return **any of them**.`,
    examples: [
      {
        input: 'words = ["wrt","wrf","er","ett","rftt"]',
        output: '"wertf"',
        explanation:
          'From "wrt" and "wrf", we know \'t\' < \'f\'. From "wrt" and "er", we know \'w\' < \'e\'. From "er" and "ett", we know \'r\' < \'t\'. From "ett" and "rftt", we know \'e\' < \'r\'. Combining these gives one valid order: "wertf".',
      },
      {
        input: 'words = ["z","x"]',
        output: '"zx"',
        explanation: 'From "z" and "x", we know \'z\' < \'x\'.',
      },
      {
        input: 'words = ["z","x","z"]',
        output: '""',
        explanation: 'The order is invalid, so return "".',
      },
    ],
    constraints: [
      '1 <= words.length <= 100',
      '1 <= words[i].length <= 100',
      'words[i] consists of only lowercase English letters',
    ],
    hints: [
      'Build a graph from the word ordering',
      'Compare adjacent words to find character ordering',
      'Use topological sort to find the total ordering',
      'Handle invalid cases: cycle or words not in lexicographic order',
    ],
    starterCode: `from typing import List
from collections import defaultdict, deque

def alien_order(words: List[str]) -> str:
    """
    Find the order of characters in alien language.
    
    Args:
        words: List of words in sorted order by alien language
        
    Returns:
        String of unique characters in alien order, or "" if invalid
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [['wrt', 'wrf', 'er', 'ett', 'rftt']],
        expected: 'wertf',
      },
      {
        input: [['z', 'x']],
        expected: 'zx',
      },
      {
        input: [['z', 'x', 'z']],
        expected: '',
      },
    ],
    timeComplexity: 'O(C)',
    spaceComplexity: 'O(1)',
    solution: `from typing import List
from collections import defaultdict, deque, Counter

def alien_order(words: List[str]) -> str:
    """
    Alien Dictionary - Graph + Topological Sort.
    
    Time: O(C) where C = total characters across all words
    Space: O(1) - at most 26 letters
    
    Key Insight:
    - Build a directed graph from character ordering
    - Compare adjacent words to find edges (char1 -> char2)
    - Use topological sort to find total ordering
    - Invalid if: cycle exists or words not in lexicographic order
    
    Steps:
    1. Initialize graph with all characters
    2. Compare adjacent words to build edges
    3. Topological sort using Kahn's algorithm
    4. Return result if all characters processed
    """
    # Step 1: Initialize graph with all characters
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in graph}
    
    # Step 2: Build graph by comparing adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check for invalid case: word1 is prefix of word2 but word1 is longer
        # e.g., ["abc", "ab"] is invalid
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        # Find first differing character
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j] in alien order
                c1, c2 = word1[j], word2[j]
                
                # Add edge if not already present
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                
                # Only first difference matters
                break
    
    # Step 3: Topological sort using Kahn's algorithm
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        c = queue.popleft()
        result.append(c)
        
        # Reduce in-degree for neighbors
        for next_c in graph[c]:
            in_degree[next_c] -= 1
            if in_degree[next_c] == 0:
                queue.append(next_c)
    
    # Step 4: Check if all characters processed
    if len(result) < len(in_degree):
        return ""  # Cycle detected
    
    return "".join(result)


# Example walkthrough:
"""
Input: words = ["wrt","wrf","er","ett","rftt"]

Step 1: Extract all characters
{w, r, t, f, e}

Step 2: Build graph by comparing adjacent words

"wrt" vs "wrf":
- w == w, r == r, t != f
- Edge: t → f

"wrf" vs "er":
- w != e
- Edge: w → e

"er" vs "ett":
- e == e, r != t
- Edge: r → t

"ett" vs "rftt":
- e != r
- Edge: e → r

Final graph:
w → e
e → r
r → t
t → f

In-degrees:
w: 0
e: 1
r: 1
t: 1
f: 1

Step 3: Topological sort
queue = [w]
Process w: result = [w], queue = [e]
Process e: result = [w, e], queue = [r]
Process r: result = [w, e, r], queue = [t]
Process t: result = [w, e, r, t], queue = [f]
Process f: result = [w, e, r, t, f]

Result: "wertf"
"""


# Common mistakes:
"""
Mistake 1: Not handling prefix case
- ["abc", "ab"] is invalid (longer word comes first)
- Must check this before comparing characters

Mistake 2: Adding duplicate edges
- Can create wrong in-degrees
- Use set for adjacency list

Mistake 3: Not initializing all characters
- Some characters might not appear in comparisons
- Initialize graph with all characters from all words

Mistake 4: Comparing more than first difference
- Only first differing character matters
- Must break after finding difference
"""`,
    leetcodeUrl: 'https://leetcode.com/problems/alien-dictionary/',
    youtubeUrl: 'https://www.youtube.com/watch?v=6kTZYvNNyps',
  },
];
