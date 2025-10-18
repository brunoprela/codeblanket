/**
 * Floyd-Warshall Algorithm Section
 */

export const floydwarshallSection = {
  id: 'floyd-warshall',
  title: 'Floyd-Warshall Algorithm',
  content: `**Floyd-Warshall** finds shortest paths between **all pairs** of vertices using **dynamic programming**.

**Algorithm:**
For each intermediate vertex k:
  For each pair (i, j):
    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

**Key Insight:**
Try using each vertex as intermediate point. If path through k is shorter, use it.

**Complexity:**
- Time: O(V³)
- Space: O(V²)

**When to Use:**
- Need all-pairs shortest paths
- Dense graph (many edges)
- Small number of vertices (≤ 400)

**Example:**
\`\`\`
Initial:
  A  B  C
A 0  1  ∞
B ∞  0  1
C 1  ∞  0

After considering each vertex:
  A  B  C
A 0  1  2
B 2  0  1
C 1  2  0
\`\`\`

**Advantages:**
- Simple to implement
- Finds all pairs at once
- Handles negative weights
- Can detect negative cycles (dist[i][i] < 0)

**Disadvantages:**
- O(V³) time (slow for large graphs)
- O(V²) space (stores all pairs)`,
};
