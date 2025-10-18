/**
 * Introduction to BFS Section
 */

export const introductionSection = {
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
};
