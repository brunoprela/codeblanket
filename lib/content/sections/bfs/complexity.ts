/**
 * Time and Space Complexity Section
 */

export const complexitySection = {
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
};
