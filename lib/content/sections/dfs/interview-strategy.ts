/**
 * Interview Strategy Section
 */

export const interviewstrategySection = {
  id: 'interview-strategy',
  title: 'Interview Strategy',
  content: `**Recognizing DFS Problems:**

**Keywords to watch for:**
- "Explore all paths"
- "Find all combinations/permutations"
- "Validate tree"
- "Find connected components"
- "Detect cycle"
- "Path sum"
- "Tree traversal"

**DFS vs BFS Decision:**

**Use DFS when:**
- Need to explore ALL paths
- Tree traversal problems
- Backtracking problems
- Path finding (not shortest)
- Cycle detection
- Topological sort
- Memory is limited (O(H) vs O(W))

**Use BFS when:**
- Need SHORTEST path (unweighted)
- Level-order traversal
- Nearest neighbor problems
- Width could be less than height

**Problem-Solving Framework:**

**1. Identify the Structure:**
- Tree or graph?
- Directed or undirected?
- Cycles possible?

**2. Choose DFS Variant:**
- Preorder? Inorder? Postorder?
- Recursive or iterative?
- Need to track visited?

**3. Define the State:**
- What information to pass down?
- What to return up?
- What to track globally?

**4. Handle Base Cases:**
- Null node?
- Leaf node?
- Target found?

**5. Backtrack if Needed:**
- Path problems
- Combination problems
- Restore state after recursion

**Common Pitfalls:**
- Forgetting visited set in graphs
- Not handling null nodes
- Forgetting to backtrack
- Incorrect base cases
- Stack overflow (use iterative)
- Modifying shared state incorrectly

**Interview Tips:**
- Start with recursive (simpler)
- Mention iterative alternative
- Discuss space complexity
- Draw small example
- Trace through recursion
- Handle edge cases (empty, single node)`,
};
