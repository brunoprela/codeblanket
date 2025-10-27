/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Graph algorithms when you see:**
- "Network", "graph", "tree" (tree is special graph)
- "Nodes and edges", "vertices and connections"
- "Connected", "path", "cycle"
- "Dependencies", "prerequisites"
- "Social network", "friends", "followers"
- "Grid" problems (can model as graph)
- "Islands", "regions" (connected components)

---

**Problem-Solving Steps:**

**Step 1: Clarify Graph Type**
- Directed or undirected?
- Weighted or unweighted?
- Cyclic or acyclic?
- How is it represented (list, matrix, implicit)?

**Step 2: Choose Algorithm**
- **Shortest path (unweighted)?** → BFS
- **Any path?** → DFS or BFS
- **Cycle detection?** → DFS
- **All components?** → DFS/BFS from each unvisited
- **Topological order?** → Kahn\'s or DFS
- **Connectivity?** → Union-Find

**Step 3: Consider Representation**
- **Given**: Use as-is
- **Build**: Adjacency list (most common)
- **Grid**: Convert to implicit graph

**Step 4: Track State**
- Visited set (almost always needed)
- Parent/predecessor (for path reconstruction)
- Distance/level (for shortest path)
- Recursion stack (for cycle in directed)

---

**Interview Communication:**

**Example: Number of Islands**1. **Clarify:**
   - "Is this a 2D grid of land and water?"
   - "Are diagonals considered connected?" (Usually no)
   - "Can I modify the input grid?"

2. **Explain approach:**
   - "This is a connected components problem."
   - "Each island is one component."
   - "I'll iterate through the grid and run DFS from each unvisited land cell."
   - "DFS will mark all connected land as visited."

3. **Walk through example:**
   \`\`\`
   Grid:
   1 1 0
   0 1 0
   0 0 1
   
   Start at (0,0): DFS marks (0,0), (0,1), (1,1) → Island 1
   Start at (2,2): DFS marks (2,2) → Island 2
   Total: 2 islands
   \`\`\`

4. **Complexity:**
   - "Time: O(M * N) - visit each cell once."
   - "Space: O(M * N) - worst case recursion depth."

---

**Common Pitfalls:**

**1. Forgetting Visited Set**
Leads to infinite loops!

**2. Wrong Parent Tracking**
For undirected graphs, track parent to avoid false cycle detection.

**3. Not Handling Disconnected Graphs**
Must try DFS/BFS from all nodes, not just one.

**4. Modifying Graph During Traversal**
Be careful with in-place modifications.

---

**Practice Plan:**1. **Basics (Day 1-2):**
   - Number of Islands
   - Clone Graph
   - Course Schedule

2. **Traversals (Day 3-4):**
   - All Paths from Source to Target
   - Pacific Atlantic Water Flow
   - Surrounded Regions

3. **Advanced (Day 5-7):**
   - Network Delay Time (Dijkstra)
   - Alien Dictionary
   - Critical Connections

4. **Resources:**
   - LeetCode Graph tag (200+ problems)
   - Practice both DFS and BFS variants
   - Draw graphs for visualization`,
};
