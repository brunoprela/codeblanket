/**
 * Dijkstra Section
 */

export const dijkstraSection = {
  id: 'dijkstra',
  title: 'Dijkstra',
  content: `**Dijkstra's Algorithm** finds the shortest path from a source to all other vertices in a **weighted graph with non-negative weights**.

**Algorithm:**
1. Initialize distances: source = 0, others = ∞
2. Use min-heap to always process closest unvisited vertex
3. For each neighbor, try to relax (improve) distance
4. Mark vertex as visited once processed

**Key Insight:**
Greedy approach - always extend the shortest known path.

**Complexity:**
- Time: O((V + E) log V) with min-heap
- Space: O(V)

**When to Use:**
- Non-negative edge weights only!
- Single-source shortest path
- GPS, network routing

**Example:**
\`\`\`
Graph:
A --1--> B
|        |
2        3
|        |
v        v
C --1--> D

Shortest paths from A:
A → A: 0
A → B: 1
A → C: 2
A → D: 3 (via A→B→D or A→C→D)
\`\`\`

**Why Non-Negative Weights:**
Negative weights can create shorter paths after a node is "finalized", breaking the greedy assumption.`,
};
