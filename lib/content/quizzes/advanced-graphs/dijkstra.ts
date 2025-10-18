/**
 * Quiz questions for Dijkstra section
 */

export const dijkstraQuiz = [
  {
    id: 'q1',
    question:
      'Explain Dijkstra algorithm step by step. Why does it need non-negative weights?',
    sampleAnswer:
      'Dijkstra finds shortest paths from source using greedy approach. Steps: 1) Initialize distances (source=0, others=infinity), 2) Use min-heap with distances, 3) Pop node with minimum distance, 4) For each neighbor, if source→current→neighbor is shorter than current best to neighbor, update neighbor distance and add to heap, 5) Repeat until heap empty. Needs non-negative weights because greedy assumption: once a node is popped from heap (minimum distance found), that distance is final - no future path can be shorter. With negative weights, later path through negative edge could be shorter, breaking the guarantee. For example, A→B cost 5, B→C cost -10, A→C cost 3. Greedy picks A→C=3 as final, but A→B→C=-5 is actually shorter. Non-negative ensures processed nodes are finalized.',
    keyPoints: [
      'Greedy: always extend shortest known path',
      'Min-heap for next node with minimum distance',
      'Relax edges: update if shorter path found',
      'Needs non-negative: greedy assumption breaks otherwise',
      'Negative weights can create shorter paths later',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Dijkstra implementation with priority queue vs simple array. What are the tradeoffs?',
    sampleAnswer:
      'Priority queue (min-heap) implementation: O((V+E) log V) time, O(V) space. Each vertex added/removed from heap O(log V), each edge relaxation potentially updates heap O(log V). Simple array: O(V²) time, O(V) space. Each iteration scans all vertices O(V) to find minimum, V iterations gives O(V²). Tradeoffs: heap is better for sparse graphs (E << V²), array is better for dense graphs (E ≈ V²) or small graphs. For example, 1000 vertices, 5000 edges (sparse): heap is O(5000 log 1000) ≈ 50K, array is O(1M). For complete graph with 1000 vertices, 500K edges (dense): heap is O(500K log 1000) ≈ 5M, array is O(1M). In practice, heap almost always preferred due to modern sparse graphs.',
    keyPoints: [
      'Heap: O((V+E) log V), better for sparse',
      'Array: O(V²), better for dense',
      'Heap: log V per operation, array: V per iteration',
      'Sparse graphs: heap wins',
      'Modern graphs usually sparse, prefer heap',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through reconstructing the shortest path from Dijkstra. Why do we need a parent/predecessor array?',
    sampleAnswer:
      'Dijkstra finds distances but path reconstruction needs parent array. During relaxation, when updating neighbor distance, also record parent[neighbor] = current. After algorithm finishes, backtrack from destination using parent array until reaching source, then reverse. For example, shortest path A→E: parent[B]=A, parent[D]=B, parent[E]=D. Backtrack: E→D→B→A, reverse to A→B→D→E. Without parent array, we only know distance is 15 but not which nodes to visit. The parent array traces the actual path by recording how we reached each node. Alternative: during relaxation, store entire path, but this uses O(V²) space vs O(V) for parent array. Parent array is space-efficient way to reconstruct optimal path.',
    keyPoints: [
      'Parent array tracks how we reached each node',
      'Update parent during edge relaxation',
      'Backtrack from dest to source, then reverse',
      'O(V) space vs O(V²) for storing full paths',
      'Essential for path reconstruction, not just distances',
    ],
  },
];
