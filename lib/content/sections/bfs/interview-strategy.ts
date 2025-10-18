/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
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
};
