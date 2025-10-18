/**
 * Quiz questions for Floyd-Warshall Algorithm section
 */

export const floydwarshallQuiz = [
  {
    id: 'q1',
    question:
      'Explain Floyd-Warshall algorithm. How does it compute all-pairs shortest paths?',
    sampleAnswer:
      'Floyd-Warshall uses dynamic programming with 3 nested loops. DP state: dist[i][j][k] = shortest path from i to j using only vertices 0..k as intermediates. Recurrence: dist[i][j][k] = min(dist[i][j][k-1], dist[i][k][k-1] + dist[k][j][k-1]). Either use k as intermediate or not. Base case: dist[i][j][0] = weight(i,j) if edge exists, infinity otherwise. Can optimize to 2D by updating in-place: for each k, for each i, for each j: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]). After considering all vertices as intermediates, dist[i][j] is shortest path from i to j. Works for negative weights (detects negative cycles if dist[i][i] < 0). Simple nested loops, no heap needed.',
    keyPoints: [
      'DP: shortest i→j using vertices 0..k',
      'Three nested loops: k, i, j',
      'Consider using k as intermediate or not',
      'Updates in-place: dist[i][j] = min(current, via k)',
      'O(V³) time, handles negative weights',
    ],
  },
  {
    id: 'q2',
    question: 'When should you use Floyd-Warshall vs running Dijkstra V times?',
    sampleAnswer:
      'Floyd-Warshall is O(V³), Dijkstra V times is O(V × (V+E) log V) = O(V²E log V) for general, O(V² log V) for sparse. For dense graphs (E ≈ V²): Dijkstra V times is O(V³ log V), Floyd-Warshall is O(V³) - Floyd wins. For sparse graphs (E ≈ V): Dijkstra V times is O(V² log V), Floyd-Warshall is O(V³) - Dijkstra wins. Use Floyd-Warshall when: need full all-pairs matrix, graph is dense, simplicity matters (easy to code), negative weights but no negative cycles. Use Dijkstra V times when: sparse graph, only need some pairs, V is large. Example: 1000 vertices, 5000 edges: Floyd is 1B ops, Dijkstra×V is 50M ops. For small complete graphs (V < 500), Floyd-Warshall often preferred for simplicity.',
    keyPoints: [
      'Floyd-Warshall: O(V³), Dijkstra×V: O(V²E log V)',
      'Dense graphs: Floyd-Warshall wins',
      'Sparse graphs: Dijkstra×V wins',
      'Floyd when: dense, simple, negative weights',
      'Dijkstra when: sparse, large V',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how to reconstruct paths in Floyd-Warshall. Why is it more complex than single-source algorithms?',
    sampleAnswer:
      'Need next array: next[i][j] = next vertex after i on shortest path to j. During relaxation, when updating dist[i][j] via k, set next[i][j] = next[i][k] (go through k). Path reconstruction: start at i, follow next[i][j] to get second vertex, then next[second][j] until reaching j. More complex because: storing all paths needs O(V²) next array vs O(V) parent array for single-source, reconstruction requires following chain vs simple backtracking. Alternative: store full paths O(V³) space but impractical. The next array is space-efficient compromise - O(V²) space, O(V) time per path reconstruction. For complete path matrix, would need O(V³) space to store all V² paths of average length V.',
    keyPoints: [
      'Next array: next[i][j] = next vertex after i to j',
      'Update next when updating distances via k',
      'Reconstruct: follow next pointers from i to j',
      'More complex: O(V²) array vs O(V) for single-source',
      'Alternative full paths: O(V³) space impractical',
    ],
  },
];
