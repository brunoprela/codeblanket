/**
 * Number of Connected Components in Undirected Graph
 * Problem ID: number-of-connected-components
 * Order: 11
 */

import { Problem } from '../../../types';

export const number_of_connected_componentsProblem: Problem = {
  id: 'number-of-connected-components',
  title: 'Number of Connected Components in Undirected Graph',
  difficulty: 'Medium',
  description: `You have a graph of \`n\` nodes labeled from \`0\` to \`n - 1\`. You are given an integer \`n\` and an array \`edges\` where \`edges[i] = [ai, bi]\` indicates that there is an **undirected** edge between nodes \`ai\` and \`bi\` in the graph.

Return the number of connected components in the graph.

A **connected component** is a set of nodes where there is a path between every pair of nodes in the set, and no path to any node outside the set.


**Why This Problem Matters:**

This is a **fundamental Union-Find problem** that tests:
- Understanding of graph connectivity
- Union-Find implementation from scratch
- Counting disjoint sets
- Common pattern: "group items into components"

**Key Applications:**
- Social networks (friend groups)
- Network topology (connected subnets)
- Image processing (connected regions)
- Clustering algorithms

**Approaches:**

**Approach 1: Union-Find (Optimal)**
- Initialize n separate components
- For each edge, union the two nodes
- Count remaining distinct components
- Time: O(N + E·α(N)) ≈ O(N + E)
- Space: O(N)
- **Best for interviews** - clean, efficient

**Approach 2: DFS/BFS**
- Build adjacency list
- For each unvisited node, run DFS/BFS and mark component
- Count number of DFS/BFS calls needed
- Time: O(N + E)
- Space: O(N + E)
- Works but more code

**Approach 3: Count Parents in Union-Find**
After all unions, count nodes where \`parent[i] == i\` (roots)`,
  examples: [
    {
      input: 'n = 5, edges = [[0,1],[1,2],[3,4]]',
      output: '2',
      explanation:
        'Component 1: {0,1,2} connected via edges [0,1] and [1,2]. Component 2: {3,4} connected via edge [3,4]. Total: 2 components.',
    },
    {
      input: 'n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]',
      output: '1',
      explanation: 'All nodes connected in one component: 0→1→2→3→4.',
    },
    {
      input: 'n = 4, edges = []',
      output: '4',
      explanation: 'No edges, so each node is its own component.',
    },
    {
      input: 'n = 1, edges = []',
      output: '1',
      explanation: 'Single node is one component.',
    },
  ],
  constraints: [
    '1 <= n <= 2000',
    '0 <= edges.length <= n * (n - 1) / 2',
    'edges[i].length == 2',
    '0 <= ai, bi < n',
    'ai != bi',
    'There are no repeated edges',
  ],
  hints: [
    'Start with n components (each node separate)',
    'Each successful union reduces component count by 1',
    'Use Union-Find to efficiently merge components',
    'Track component count during unions OR count roots at end',
    'Alternative: Use DFS/BFS to visit and mark components',
  ],
  starterCode: `def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count the number of connected components.
    
    Args:
        n: Number of nodes (0-indexed: 0 to n-1)
        edges: List of undirected edges [u, v]
        
    Returns:
        Number of connected components
        
    Example:
        n = 5, edges = [[0,1], [1,2], [3,4]]
        Initial: 5 components {0}, {1}, {2}, {3}, {4}
        After [0,1]: 4 components {0,1}, {2}, {3}, {4}
        After [1,2]: 3 components {0,1,2}, {3}, {4}
        After [3,4]: 2 components {0,1,2}, {3,4}
        Return 2
    """
    pass`,
  testCases: [
    {
      input: [
        5,
        [
          [0, 1],
          [1, 2],
          [3, 4],
        ],
      ],
      expected: 2,
    },
    {
      input: [
        5,
        [
          [0, 1],
          [1, 2],
          [2, 3],
          [3, 4],
        ],
      ],
      expected: 1,
    },
    {
      input: [4, []],
      expected: 4,
    },
    {
      input: [1, []],
      expected: 1,
    },
  ],
  solution: `class UnionFind:
    """
    Union-Find with component counting.
    
    Tracks number of distinct components as unions are performed.
    """
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.components = n  # Start with n separate components
    
    def find(self, x):
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union two components.
        Returns True if union performed (decrements component count).
        Returns False if already in same component.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Union by rank
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        # Successfully merged two components → decrease count
        self.components -= 1
        return True
    
    def get_count(self):
        """Return current number of components."""
        return self.components


def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components using Union-Find.
    
    Algorithm:
    1. Start with n components (each node separate)
    2. For each edge, union the two nodes
    3. Each successful union reduces component count by 1
    4. Return final component count
    
    Time: O(N + E·α(N)) where E is number of edges
    Space: O(N)
    
    Walkthrough: n=5, edges=[[0,1], [1,2], [3,4]]
    
    Initial: components = 5
    parent = [0,1,2,3,4]
    
    Edge [0,1]:
    - union(0,1): parent[1]=0, components=4
    - Components: {0,1}, {2}, {3}, {4}
    
    Edge [1,2]:
    - find(1)=0, find(2)=2
    - union: parent[2]=0, components=3
    - Components: {0,1,2}, {3}, {4}
    
    Edge [3,4]:
    - union(3,4): parent[4]=3, components=2
    - Components: {0,1,2}, {3,4}
    
    Return 2
    """
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.get_count()


# Alternative: Count roots at the end (without tracking during unions)
def countComponentsAlternative(n: int, edges: list[list[int]]) -> int:
    """
    Alternative approach: Count distinct roots after all unions.
    
    After processing all edges, count how many nodes are their own parent
    (i.e., roots of trees in the forest).
    
    Time: O(N + E·α(N))
    Space: O(N)
    """
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y
    
    # Process all edges
    for u, v in edges:
        union(u, v)
    
    # Count distinct roots
    # A node is a root if parent[i] == i
    return sum(find(i) == i for i in range(n))


# Alternative: DFS approach (for comparison)
def countComponentsDFS(n: int, edges: list[list[int]]) -> int:
    """
    DFS approach: Build graph and count number of DFS calls needed.
    
    Time: O(N + E)
    Space: O(N + E) for adjacency list
    
    Good alternative if Union-Find not allowed or graph is already built.
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    
    def dfs(node):
        """Visit all nodes in this component."""
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    components = 0
    for i in range(n):
        if i not in visited:
            dfs(i)
            components += 1
    
    return components


"""
Which Approach to Use in Interviews?

Union-Find (Recommended):
✅ Optimal time complexity
✅ Clean, concise code
✅ Shows advanced data structure knowledge
✅ Easy to track components during process
✅ No need to build adjacency list

DFS/BFS:
✅ Works well, also O(N+E)
✅ More intuitive for some people
✅ Useful if graph already built
⚠️ More code (build graph, track visited)
⚠️ Requires more space for adjacency list

Interview Strategy:
1. Start with Union-Find - it's optimal and cleaner
2. Explain time complexity: O(N + E·α(N)) ≈ O(N+E)
3. Mention DFS as alternative if asked
4. For this specific problem, Union-Find is preferred

Common Mistakes:
❌ Forgetting to count nodes with no edges (isolated nodes)
❌ Not using path compression (becomes O(log N))
❌ Building adjacency list unnecessarily (wastes space)
❌ Counting components incorrectly after unions

Key Insight:
Start with N components, each union merges two → decreases count by 1.
This is simpler than counting roots at the end!
"""`,
  timeComplexity: 'O(N + E·α(N)) ≈ O(N + E)',
  spaceComplexity: 'O(N)',
  order: 11,
  topic: 'Advanced Graphs',
  leetcodeUrl:
    'https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/',
  youtubeUrl: 'https://www.youtube.com/watch?v=8f1XPm4WOUc',
};
