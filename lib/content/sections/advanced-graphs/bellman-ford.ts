/**
 * Bellman-Ford Algorithm Section
 */

export const bellmanfordSection = {
  id: 'bellman-ford',
  title: 'Bellman-Ford Algorithm',
  content: `**Bellman-Ford** finds shortest paths even with **negative edge weights**, and detects **negative cycles**.

**Algorithm:**1. Initialize distances: source = 0, others = ∞
2. Relax all edges V-1 times
3. Check for negative cycles (one more relaxation)

**Key Insight:**
In a graph with V vertices, shortest path has at most V-1 edges. Relax all edges V-1 times guarantees finding shortest paths.

**Complexity:**
- Time: O(V * E)
- Space: O(V)

**Advantages over Dijkstra:**
- ✅ Handles negative weights
- ✅ Detects negative cycles
- ❌ Slower than Dijkstra

**Example with Negative Weight:**
\`\`\`
A --2--> B
|        |
1       -3
|        |
v        v
C <--1-- D

Shortest A → D:
Via A→B→D: 2 + (-3) = -1 (shortest!)
Via A→C→D: 1 + 1 = 2
\`\`\`

**Negative Cycle:**
If relaxation happens in Vth iteration, negative cycle exists (distances keep decreasing infinitely).`,
};
