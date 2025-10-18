/**
 * Time and Space Complexity Section
 */

export const complexitySection = {
  id: 'complexity',
  title: 'Time and Space Complexity',
  content: `**Time Complexity:**

**Tree DFS:**
- **O(N)** where N = number of nodes
- Visit each node exactly once
- Each edge traversed twice (down and back up)

**Graph DFS:**
- **O(V + E)** where V = vertices, E = edges
- Visit each vertex once: O(V)
- Explore each edge once (or twice for undirected): O(E)

**Space Complexity:**

**Recursive DFS:**
- **O(H)** where H = height/depth
- Call stack space
- Best case (balanced tree): O(log N)
- Worst case (skewed tree/graph): O(N)

**Iterative DFS:**
- **O(H)** for explicit stack
- Plus O(V) for visited set in graphs
- Total: O(V) for graphs

**Additional Space:**
- Visited set for graphs: O(V)
- Result array: O(N) or O(V)
- Path tracking: O(H) to O(V)

**Optimization Tips:**
- Use iterative for very deep structures
- Use visited set to avoid revisiting
- Clear visited set if running multiple DFS
- Consider BFS if you need shortest path`,
};
