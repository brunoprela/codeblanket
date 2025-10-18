/**
 * Redundant Connection
 * Problem ID: redundant-connection
 * Order: 10
 */

import { Problem } from '../../../types';

export const redundant_connectionProblem: Problem = {
  id: 'redundant-connection',
  title: 'Redundant Connection',
  difficulty: 'Medium',
  description: `In this problem, a tree is an **undirected graph** that is connected and has no cycles.

You are given a graph that started as a tree with \`n\` nodes labeled from \`1\` to \`n\`, with one additional edge added. The added edge has two **different** vertices chosen from \`1\` to \`n\`, and was not an edge that already existed. The graph is represented as an array \`edges\` of length \`n\` where \`edges[i] = [ai, bi]\` indicates that there is an edge between nodes \`ai\` and \`bi\` in the graph.

Return an edge that can be removed so that the resulting graph is a tree of \`n\` nodes. If there are multiple answers, return the answer that occurs last in the input.


**Why This Problem Matters:**

This is the **classic Union-Find cycle detection** problem that appears frequently at FAANG companies. It tests:
- Understanding of Union-Find (Disjoint Set Union)
- Cycle detection in undirected graphs
- Path compression and union by rank optimization
- Graph theory fundamentals

**Key Insight:**

A tree with \`n\` nodes has exactly \`n-1\` edges. When we add one more edge, it creates exactly one cycle. We need to find which edge creates the cycle.

**Approach: Union-Find**

Process edges one by one:
- If both vertices are already connected (same parent), this edge creates a cycle → return it
- Otherwise, union the two vertices

**Why Union-Find is Ideal:**
- Efficiently checks if two nodes are in the same component
- Nearly O(1) operations with path compression + union by rank
- Perfect for incremental connectivity queries

**Time Complexity:** O(N · α(N)) ≈ O(N) where α is inverse Ackermann function
**Space Complexity:** O(N) for parent and rank arrays`,
  examples: [
    {
      input: 'edges = [[1,2],[1,3],[2,3]]',
      output: '[2,3]',
      explanation:
        'The graph forms a triangle: 1-2, 1-3, 2-3. Removing [2,3] (last edge) breaks the cycle.',
    },
    {
      input: 'edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]',
      output: '[1,4]',
      explanation:
        'After adding edges [1,2], [2,3], [3,4], adding [1,4] creates cycle 1→2→3→4→1. [1,4] is the redundant edge.',
    },
    {
      input: 'edges = [[1,2],[1,3],[2,4]]',
      output: '[]',
      explanation:
        "This forms a valid tree with no cycles (this case won't actually occur in the problem).",
    },
  ],
  constraints: [
    'n == edges.length',
    '3 <= n <= 1000',
    'edges[i].length == 2',
    '1 <= ai < bi <= n',
    'ai != bi',
    'There are no repeated edges',
    'The given graph is connected',
  ],
  hints: [
    'Use Union-Find (Disjoint Set Union) data structure',
    'For each edge, check if both nodes already have the same parent',
    'If they do, this edge creates a cycle - return it',
    'Otherwise, union the two nodes and continue',
    'Implement path compression for find() to optimize',
    'Implement union by rank for union() to keep tree flat',
  ],
  starterCode: `def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """
    Find the redundant edge that creates a cycle.
    
    Args:
        edges: List of edges [u, v] where nodes are 1-indexed
        
    Returns:
        The last edge that creates a cycle
        
    Example:
        edges = [[1,2],[1,3],[2,3]]
        After [1,2]: 1-2 connected
        After [1,3]: 1-3 connected (1,2,3 in one component)
        After [2,3]: 2 and 3 already connected via 1 → CYCLE!
        Return [2,3]
    """
    pass`,
  testCases: [
    {
      input: [
        [
          [1, 2],
          [1, 3],
          [2, 3],
        ],
      ],
      expected: [2, 3],
    },
    {
      input: [
        [
          [1, 2],
          [2, 3],
          [3, 4],
          [1, 4],
          [1, 5],
        ],
      ],
      expected: [1, 4],
    },
    {
      input: [
        [
          [1, 2],
          [2, 3],
          [3, 1],
        ],
      ],
      expected: [3, 1],
    },
  ],
  solution: `class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure.
    
    Key Operations:
    - find(x): Find the root/parent of x (with path compression)
    - union(x, y): Merge the sets containing x and y (with union by rank)
    - connected(x, y): Check if x and y are in the same set
    
    Optimizations:
    1. Path Compression: Make all nodes on path point directly to root
    2. Union by Rank: Attach smaller tree under larger tree
    
    Time Complexity: O(α(n)) per operation (α is inverse Ackermann, practically constant)
    """
    
    def __init__(self, n):
        # Initialize: each node is its own parent (separate component)
        self.parent = list(range(n + 1))  # 1-indexed, so size n+1
        self.rank = [1] * (n + 1)         # Initial rank is 1
    
    def find(self, x):
        """
        Find root of x with path compression.
        
        Path compression: Make every node on path point directly to root.
        This flattens the tree structure for future operations.
        
        Example: Before path compression
            1 (root)
            |
            2
            |
            3
            |
            4
        
        find(4) → After path compression:
            1 (root)
           /|\\
          2 3 4  (all point to root)
        """
        if self.parent[x] != x:
            # Recursively find root and compress path
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """
        Union two sets containing x and y.
        
        Returns True if they were in different sets (union performed)
        Returns False if already in same set (creates cycle)
        
        Union by rank: Attach smaller tree under root of larger tree
        to keep overall tree height small.
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        # Already in same set → would create cycle
        if root_x == root_y:
            return False
        
        # Union by rank: attach smaller under larger
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            # Equal rank: arbitrarily choose, increment rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True


def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """
    Find the redundant edge using Union-Find.
    
    Algorithm:
    1. Initialize Union-Find with n nodes
    2. Process each edge [u, v]:
       - If u and v already connected → this edge creates cycle, return it
       - Otherwise, union u and v
    3. Return last edge that created cycle
    
    Time: O(N · α(N)) ≈ O(N) where N is number of edges
    Space: O(N) for parent and rank arrays
    
    Walkthrough: edges = [[1,2], [1,3], [2,3]]
    
    Initial: parent = [0,1,2,3], rank = [1,1,1,1]
    
    Edge [1,2]:
    - find(1) = 1, find(2) = 2 (different)
    - union(1,2): parent[2] = 1
    - parent = [0,1,1,3]
    
    Edge [1,3]:
    - find(1) = 1, find(3) = 3 (different)
    - union(1,3): parent[3] = 1
    - parent = [0,1,1,1]
    
    Edge [2,3]:
    - find(2) = 1 (2→1), find(3) = 1 (3→1)
    - Both have same root 1 → CYCLE!
    - Return [2,3]
    """
    n = len(edges)
    uf = UnionFind(n)
    
    for u, v in edges:
        # Try to union u and v
        # If they're already connected, this edge creates a cycle
        if not uf.union(u, v):
            return [u, v]
    
    return []  # No cycle found (shouldn't happen per problem constraints)


# Alternative: Without separate class (inline implementation)
def findRedundantConnectionInline(edges: list[list[int]]) -> list[int]:
    """
    Same algorithm, but with inline Union-Find implementation.
    More compact for interviews if time is limited.
    """
    n = len(edges)
    parent = list(range(n + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return False  # Already connected → cycle
        parent[root_x] = root_y
        return True
    
    for u, v in edges:
        if not union(u, v):
            return [u, v]
    
    return []


"""
Why Union-Find is Perfect Here:

1. **Incremental Connectivity:**
   - We add edges one at a time
   - Need to check if adding edge creates cycle
   - Union-Find excels at dynamic connectivity

2. **Efficiency:**
   - find(): O(α(n)) ≈ O(1) with path compression
   - union(): O(α(n)) ≈ O(1) with union by rank
   - Total: O(N) for N edges

3. **Simplicity:**
   - Clean, short implementation
   - Easy to explain in interview
   - No need for DFS/BFS

Comparison with DFS:
- DFS: O(N²) - would need to check cycle after each edge
- Union-Find: O(N) - constant time per edge

Interview Tips:
1. Explain path compression clearly
2. Draw example to show cycle detection
3. Mention that tree with n nodes has n-1 edges
4. Note that we return the LAST edge creating cycle

Common Mistakes:
❌ Using 0-indexed when problem is 1-indexed
❌ Forgetting path compression (makes it O(log N))
❌ Not checking if union succeeded before continuing
❌ Returning first cycle edge instead of last
"""`,
  timeComplexity: 'O(N · α(N)) ≈ O(N)',
  spaceComplexity: 'O(N)',
  order: 10,
  topic: 'Advanced Graphs',
  leetcodeUrl: 'https://leetcode.com/problems/redundant-connection/',
  youtubeUrl: 'https://www.youtube.com/watch?v=FXWRE67PLL0',
};
