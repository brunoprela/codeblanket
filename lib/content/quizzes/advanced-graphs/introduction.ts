/**
 * Quiz questions for Advanced Graph Algorithms section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between basic graph algorithms and advanced graph algorithms. What makes an algorithm "advanced"?',
    sampleAnswer:
      'Basic algorithms (BFS, DFS, topological sort) work on unweighted graphs or simple problems, run in O(V+E). Advanced algorithms handle weighted graphs, find optimal paths, detect special structures. "Advanced" means: dealing with edge weights (Dijkstra, Bellman-Ford), all-pairs problems (Floyd-Warshall), graph connectivity (Union-Find, MST), optimization (shortest path, minimum spanning tree). For example, BFS finds if path exists (basic), Dijkstra finds shortest weighted path (advanced). DFS visits nodes (basic), Tarjan finds strongly connected components (advanced). Advanced algorithms are more complex, often involve greedy choices or dynamic programming, have higher complexity O(E log V) or O(V³). They solve optimization and structural problems that basic algorithms cannot.',
    keyPoints: [
      'Basic: unweighted, simple, O(V+E)',
      'Advanced: weighted, optimization, O(E log V) or O(V³)',
      'Handle: edge weights, all-pairs, connectivity',
      'Examples: Dijkstra, Floyd-Warshall, Union-Find',
      'Solve: shortest paths, MST, special structures',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare single-source vs all-pairs shortest path problems. When would you use each?',
    sampleAnswer:
      'Single-source finds shortest paths from one node to all others. Use Dijkstra O(E log V) or Bellman-Ford O(VE). All-pairs finds shortest paths between every pair of nodes. Use Floyd-Warshall O(V³). Choose single-source when: start node known (GPS navigation from current location), run multiple times for different sources (still faster than all-pairs if sources < V). Choose all-pairs when: need distances between all pairs (distance matrix), running single-source V times anyway, V is small (few hundred nodes). For example, GPS routing is single-source (from your location). City distance table is all-pairs (all city pairs). Time comparison: Dijkstra V times is O(VE log V), Floyd-Warshall is O(V³). For sparse graphs (E << V²), V × Dijkstra faster.',
    keyPoints: [
      'Single-source: one node to all, O(E log V)',
      'All-pairs: every pair, O(V³)',
      'Single when: known start, sparse graph',
      'All-pairs when: need full matrix, small V',
      'V × Dijkstra vs Floyd-Warshall depends on graph density',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe MST (Minimum Spanning Tree). What real-world problems does it solve?',
    sampleAnswer:
      'MST is tree connecting all vertices with minimum total edge weight. No cycles, n-1 edges for n vertices. Algorithms: Kruskal (sort edges, union-find) O(E log E), Prim (greedy with heap) O(E log V). Real-world: network design (minimum cable to connect all buildings), circuit design (minimum wire to connect pins), clustering (threshold on MST edges), approximation for TSP. For example, connecting cities with roads: MST gives minimum total road length while ensuring all cities connected. Utility network design: minimum pipe/cable to reach all customers. Key property: locally optimal edge choices (greedy) lead to globally optimal tree. MST is unique if all edge weights distinct. Multiple MSTs possible with duplicate weights.',
    keyPoints: [
      'Tree connecting all vertices, minimum total weight',
      'n-1 edges, no cycles',
      'Algorithms: Kruskal O(E log E), Prim O(E log V)',
      'Real-world: network design, clustering, TSP approx',
      'Greedy choices lead to global optimum',
    ],
  },
];
