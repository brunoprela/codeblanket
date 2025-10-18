/**
 * Introduction to DFS Section
 */

export const introductionSection = {
  id: 'introduction',
  title: 'Introduction to DFS',
  content: `**Depth-First Search (DFS)** is a fundamental algorithm for exploring trees and graphs. It explores as far as possible along each branch before backtracking.

**Core Concept:**
- Start at a node
- Explore one neighbor completely (go deep)
- Backtrack when you hit a dead end
- Continue until all nodes are visited

**Key Characteristics:**
- Uses **recursion** or an explicit **stack**
- Goes **deep** before going wide
- Natural fit for trees and recursive problems
- Can find paths but NOT shortest paths

**DFS vs BFS:**
- **DFS**: Stack/Recursion → Goes deep → O(H) space
- **BFS**: Queue → Goes wide → O(W) space (W = width)

**When to Use DFS:**
- Tree traversals (preorder, inorder, postorder)
- Finding paths in a graph/maze
- Topological sorting
- Detecting cycles
- Finding connected components
- Generating permutations/combinations
- Backtracking problems`,
};
