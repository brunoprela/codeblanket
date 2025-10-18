/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize an advanced graph problem? What keywords signal these algorithms?',
    sampleAnswer:
      'Keywords: "shortest path", "minimum spanning tree", "weighted graph", "negative weights", "all pairs", "connectivity", "network design". Patterns: optimization on weighted graph, need distances between many pairs, building optimal network, detecting arbitrage/cycles. For example, "find shortest route with tolls" → weighted shortest path (Dijkstra). "Connect all cities with minimum cable" → MST. "Currency exchange profitability" → Bellman-Ford with cycle detection. "Distance table between all airports" → Floyd-Warshall. "Check if network components connected" → Union-Find. The signals: weights mentioned (not just connectivity), optimization problem (minimum/shortest), all-pairs requirement, special graph properties. Advanced graphs solve optimization, not just traversal.',
    keyPoints: [
      'Keywords: shortest, minimum, weighted, all pairs',
      'Optimization on weighted graphs',
      'Examples: routes with costs, network design',
      'vs Basic: optimization not just traversal',
      'Weights + optimization → advanced algorithms',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your advanced graph interview approach from recognition to implementation.',
    sampleAnswer:
      'First, recognize algorithm type from problem (weighted → shortest path, connect all → MST). Second, clarify: negative weights? Single-source or all-pairs? Need path or just distance? Third, choose algorithm: non-negative → Dijkstra, negative → Bellman-Ford, all-pairs small graph → Floyd-Warshall, connect all → MST (Kruskal/Prim). Fourth, state complexity and why that algorithm. Fifth, draw small example showing algorithm steps. Sixth, discuss data structures: heap for Dijkstra, Union-Find for Kruskal. Seventh, code clearly with proper initialization and relaxation logic. Eighth, test with edge cases: disconnected graph, negative cycles, self-loops. Finally, discuss optimizations: bidirectional search, A* heuristic. This shows: recognition, algorithm selection, implementation, optimization knowledge.',
    keyPoints: [
      'Recognize type: weights, single/all pairs, MST',
      'Clarify: negative weights, paths vs distances',
      'Choose algorithm with justification',
      'State complexity, draw example',
      'Code with proper structures, test edges',
      'Discuss optimizations',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common mistakes in advanced graph problems and how do you avoid them?',
    sampleAnswer:
      'First: using Dijkstra with negative weights (fails silently). Second: forgetting to check negative cycles in Bellman-Ford. Third: initializing distances incorrectly (source != 0 or others != infinity). Fourth: heap contains duplicate entries for same vertex in Dijkstra (inefficient). Fifth: Union-Find without optimizations (O(n) instead of O(α(n))). Sixth: Floyd-Warshall loop order wrong (k must be outermost). Seventh: off-by-one in iterations (V-1 for Bellman-Ford). My strategy: verify weight constraints, always test with negative weights if allowed, initialize carefully, use decrease-key or visited set for Dijkstra, always implement Union-Find with optimizations, remember Floyd-Warshall loop order (alphabetical: k,i,j). Most mistakes from wrong algorithm choice or incorrect initialization.',
    keyPoints: [
      'Wrong algorithm for weight type',
      'Missing negative cycle check',
      'Incorrect initialization',
      'Heap duplicates in Dijkstra',
      'Union-Find without optimizations',
      'Test: negative weights, initialization, loop order',
    ],
  },
];
