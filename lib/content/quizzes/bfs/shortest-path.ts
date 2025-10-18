/**
 * Quiz questions for BFS for Shortest Path section
 */

export const shortestpathQuiz = [
  {
    id: 'q1',
    question:
      'Explain why BFS finds shortest path in unweighted graphs. Walk through an example.',
    sampleAnswer:
      'BFS guarantees shortest path because it explores nodes in order of increasing distance. First time a node is reached is via shortest path (all shorter paths already explored). Example graph: A→B, A→C, B→D, C→D, D→E. Shortest A to E? BFS: start A (dist 0), visit B,C (dist 1), visit D (dist 2, first reach via B and C both distance 2), visit E (dist 3). Path A→B→D→E or A→C→D→E both length 3. Cannot be shorter because all paths of length < 3 already explored. DFS might find A→C→D→E first, but could also explore A→B→C→... longer path. BFS level-by-level ensures optimality.',
    keyPoints: [
      'Explores in order of increasing distance',
      'First arrival = shortest path',
      'All shorter paths explored already',
      'Example: finds distance 3 before exploring deeper',
      'DFS no guarantee: may explore long first',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe path reconstruction in BFS. How do you track the actual path, not just distance?',
    sampleAnswer:
      'Track parent array: parent[v] = u means we reached v from u. During BFS, when enqueuing neighbor v from u, set parent[v] = u. After BFS finishes, backtrack from destination to source using parent array, then reverse. Example A→B→D→E: parent[B]=A, parent[D]=B, parent[E]=D. Backtrack from E: E→D→B→A, reverse to A→B→D→E. Without parent array, only know distance 3, not which nodes. Alternative: store paths in queue (each node stores full path to reach it), but uses O(V²) space vs O(V) for parent. Parent array is standard, space-efficient way. Can also track distance alongside parent for both metrics.',
    keyPoints: [
      'Parent array: parent[v] = u (reached v from u)',
      'Update parent when enqueuing neighbor',
      'Backtrack from dest to source, then reverse',
      'O(V) space vs O(V²) for full paths',
      'Standard pattern for BFS path reconstruction',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare shortest path: BFS vs Dijkstra vs Bellman-Ford. When to use each?',
    sampleAnswer:
      'BFS: O(V+E), unweighted graphs only. Finds shortest path by counting edges. Use when: all edges weight 1 or equal, simplest and fastest. Dijkstra: O((V+E) log V), weighted non-negative graphs. Finds shortest path by summing weights. Use when: edges have different positive weights, need actual distance. Bellman-Ford: O(VE), weighted with possible negative weights. Use when: negative weights exist, need negative cycle detection. For unweighted: always BFS. For weighted positive: always Dijkstra. For negative: Bellman-Ford. Never use Dijkstra with negative weights (fails). Example: social network distance (unweighted) → BFS. Road network with distances (positive weights) → Dijkstra. Currency exchange (negative edges possible) → Bellman-Ford.',
    keyPoints: [
      'BFS: O(V+E) unweighted',
      'Dijkstra: O((V+E) log V) weighted non-negative',
      'Bellman-Ford: O(VE) negative weights',
      'BFS fastest for unweighted',
      'Hierarchy: BFS → Dijkstra → Bellman-Ford',
    ],
  },
];
