/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain why BFS and DFS are both O(V + E) time. Walk me through what each part represents.',
    sampleAnswer:
      'Both visit every vertex once and explore every edge once, giving O(V + E). V represents visiting each vertex: we mark it visited, process it, add to queue or call recursion. E represents exploring edges: for each vertex, we check all its neighbors via edges. In adjacency list, summing all neighbor lists gives exactly E edges total. For example, graph with 5 vertices and 7 edges: we visit 5 vertices (V work) and check 7 edges (E work), so 5 + 7 = 12 operations. The + not × because we do V work (visiting) and separately E work (edge checking), not nested. This is why adjacency list is efficient - we only examine edges that exist, not all V^2 possible edges.',
    keyPoints: [
      'Visit each vertex once = O(V)',
      'Explore each edge once = O(E)',
      'Sum of all adjacency lists = E',
      'V + E not V × E: separate operations',
      'Adjacency list: only check existing edges',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare time complexity of Dijkstra vs Bellman-Ford. When would you choose the slower one?',
    sampleAnswer:
      'Dijkstra is O(E log V) with min-heap, Bellman-Ford is O(V × E). Dijkstra is faster but requires non-negative edge weights. Bellman-Ford handles negative weights and detects negative cycles. Choose Bellman-Ford when: graph has negative edges, need to detect negative cycles, or graph is small so cubic time acceptable. For example, currency exchange with fees (negative edges) or finding arbitrage (negative cycle detection). Dijkstra fails with negative edges because greedy approach assumes once a node is finalized, no better path exists. Negative edges can create better paths to already-finalized nodes. Most real graphs (roads, networks) have non-negative weights, so Dijkstra is usually preferred.',
    keyPoints: [
      'Dijkstra: O(E log V), needs non-negative weights',
      'Bellman-Ford: O(V × E), handles negative weights',
      'Bellman-Ford: detects negative cycles',
      'Use Bellman-Ford: negative edges, detect cycles',
      'Dijkstra fails with negative: greedy assumption broken',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain Union-Find complexity with path compression and union by rank. Why is it nearly constant?',
    sampleAnswer:
      'Union-Find with optimizations achieves O(α(n)) per operation where α is inverse Ackermann function - grows so slowly it is effectively constant for all practical n (less than 5 for n < 10^80). Path compression: during find, make all nodes point directly to root, flattening tree. Union by rank: attach smaller tree under larger, keeping trees shallow. Together, these prevent deep trees. Without optimizations, trees can be O(n) deep, making operations O(n). With both optimizations, trees stay very flat (height < 5 practically), giving nearly O(1) operations. This makes Union-Find extremely efficient for dynamic connectivity - can handle millions of operations in linear time.',
    keyPoints: [
      'With optimizations: O(α(n)) ≈ O(1) practical',
      'Path compression: flatten tree during find',
      'Union by rank: attach smaller tree under larger',
      'Prevents deep trees (height < 5 practically)',
      'Without optimizations: O(n) per operation',
    ],
  },
];
