/**
 * Interview Strategy Section
 */

export const interviewSection = {
  id: 'interview',
  title: 'Interview Strategy',
  content: `**Recognition Signals:**

**Use Advanced Graphs when you see:**
- "Shortest path", "minimum cost", "cheapest route"
- "Weighted graph", "edge costs"
- "Network", "cities connected", "flights"
- "Negative weights", "negative cycle"
- "All pairs shortest path"

---

**Problem-Solving Steps:**

**Step 1: Identify Problem Type (2 min)**
- Single source or all pairs?
- Weighted or unweighted?
- Negative weights possible?
- Need to detect cycles?

**Step 2: Choose Algorithm (2 min)**
- Follow decision tree above
- Consider time/space constraints

**Step 3: Implementation (15 min)**
- Dijkstra: Min-heap + relaxation
- Bellman-Ford: Relax V-1 times
- Floyd-Warshall: Triple nested loop

**Step 4: Edge Cases (2 min)**
- Disconnected graph
- Negative cycles
- Source unreachable
- Self-loops

---

**Common Mistakes:**

**1. Using Dijkstra with Negative Weights**
Will give wrong answer! Use Bellman-Ford.

**2. Not Checking for Negative Cycles**
Bellman-Ford can detect them - use it!

**3. Wrong Heap Priority**
Dijkstra: always pop minimum distance.

**4. Forgetting Visited Set**
Dijkstra: mark as visited after popping.

---

**Interview Communication:**

*Interviewer: Find shortest path in weighted graph.*

**You:**1. "Are all weights non-negative?" → Use Dijkstra
2. "Can weights be negative?" → Use Bellman-Ford
3. "Need all pairs?" → Consider Floyd-Warshall

4. **Dijkstra Explanation:**
   - "Use min-heap with (distance, node)."
   - "Always extend shortest known path (greedy)."
   - "Relax neighbors: update if shorter path found."
   - "O((V+E)logV) time, O(V) space."`,
};
