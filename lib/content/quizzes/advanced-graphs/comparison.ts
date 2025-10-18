/**
 * Quiz questions for Algorithm Comparison section
 */

export const comparisonQuiz = [
  {
    id: 'q1',
    question:
      'Compare all shortest path algorithms by complexity and use case. Which should you use when?',
    sampleAnswer:
      'BFS: O(V+E), unweighted graphs. Dijkstra: O((V+E) log V), weighted non-negative. Bellman-Ford: O(VE), weighted with negatives. Floyd-Warshall: O(V³), all-pairs any weights. Use BFS when: unweighted or unit weights (simplest, fastest). Use Dijkstra when: weighted non-negative, single-source (almost always use if possible). Use Bellman-Ford when: negative weights, need cycle detection, distributed system. Use Floyd-Warshall when: all-pairs needed, dense graph, negative weights OK. For example: social network distances (unweighted) → BFS. GPS navigation (positive weights) → Dijkstra. Currency exchange (negative edges) → Bellman-Ford. City distance matrix (all pairs, moderate size) → Floyd-Warshall. The choice hierarchy: try simpler first (BFS, Dijkstra), only use complex when necessary.',
    keyPoints: [
      'BFS: O(V+E) unweighted',
      'Dijkstra: O((V+E) log V) weighted non-negative',
      'Bellman-Ford: O(VE) negative weights',
      'Floyd-Warshall: O(V³) all-pairs',
      'Hierarchy: simpler first, complex when necessary',
    ],
  },
  {
    id: 'q2',
    question:
      'When would you need to implement Dijkstra from scratch vs using library? What are the tradeoffs?',
    sampleAnswer:
      'Implement from scratch when: custom graph representation, need to modify algorithm (bidirectional search, A* heuristic), educational purpose, performance-critical with specific optimizations, no suitable library available. Use library when: standard implementation suffices, development speed matters, well-tested code preferred, graph format matches library. Tradeoffs of custom: full control and optimization but time-consuming and bug-prone. Library: fast development but less flexibility. For interviews: implement from scratch (test understanding). For production: prefer library (NetworkX Python, Boost C++, JGraphT Java) unless special needs. Custom optimization examples: persistent data structures for online queries, approximation for huge graphs, parallel implementation for distributed systems. Modern libraries are well-optimized, prefer them unless compelling reason.',
    keyPoints: [
      'Scratch when: custom needs, modifications, learning',
      'Library when: standard case, speed, reliability',
      'Interviews: implement to show understanding',
      'Production: prefer library unless special needs',
      'Libraries: NetworkX, Boost, JGraphT',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe space-time tradeoffs in graph algorithms. When would you optimize for space vs time?',
    sampleAnswer:
      'Time-space tradeoffs: precompute all-pairs O(V²) space vs compute on-demand O(1) space. Store paths O(V³) vs reconstruct O(V²). Implicit graph representation (generate neighbors) vs explicit adjacency list. Optimize for time when: frequent queries, space available, performance critical (real-time systems). Optimize for space when: huge graphs (billions of edges), limited memory, infrequent queries. For example, Google Maps: precomputes many routes O(V²) space for fast query. Mobile app: compute on-demand to save phone memory. Streaming graphs: process edges online, no full storage. Modern trend: compress graphs, external memory algorithms, approximate answers. The choice depends on system constraints: embedded systems (tight memory), data centers (optimize time), mobile (balance both).',
    keyPoints: [
      'Precompute vs on-demand computation',
      'Store paths vs reconstruct',
      'Time when: frequent queries, space available',
      'Space when: huge graphs, limited memory',
      'Modern: compression, streaming, approximation',
    ],
  },
];
