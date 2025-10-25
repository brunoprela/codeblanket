/**
 * Quiz questions for Bellman-Ford Algorithm section
 */

export const bellmanfordQuiz = [
  {
    id: 'q1',
    question:
      'Explain Bellman-Ford algorithm. Why can it handle negative weights when Dijkstra cannot?',
    sampleAnswer:
      'Bellman-Ford relaxes all edges V-1 times. For each iteration, for each edge (u,v), if dist[u] + weight (u,v) < dist[v], update dist[v]. After V-1 iterations, all shortest paths found (shortest path has at most V-1 edges). Can handle negative weights because it considers all possible paths systematically, not greedily. Dijkstra assumes once a node is processed, its distance is final. Bellman-Ford reconsiders nodes multiple times, allowing negative edges to improve paths discovered earlier. For example, with negative edge, early iteration might find expensive path, later iteration finds cheaper path through negative edge. The V-1 iterations ensure even longest path (visiting V-1 edges) is discovered. Extra Vth iteration detects negative cycles.',
    keyPoints: [
      'Relax all edges V-1 times',
      'Considers all paths systematically, not greedy',
      'Reconsiders nodes multiple times',
      'Handles negative weights by re-evaluation',
      'V-1 iterations enough for longest path',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe negative cycle detection in Bellman-Ford. Why is this important?',
    sampleAnswer:
      'After V-1 iterations, run one more iteration. If any distance still improves, negative cycle exists. Why important? Negative cycles make shortest path undefined - you can loop infinitely reducing cost. For example, A→B cost 5, B→C cost -10, C→A cost 2. Loop A→B→C→A has cost 5-10+2=-3. Each loop reduces cost by 3, making shortest path -infinity. Applications: currency arbitrage (negative cycle means profit), detecting inconsistent constraints, game balancing. Detection algorithm: if Vth iteration still updates distances, trace back the updated node to find cycle. Some variations mark all nodes reachable from negative cycle as having distance -infinity. Critical for correctness - reporting finite distance when none exists is wrong.',
    keyPoints: [
      'Run Vth iteration, if updates exist → negative cycle',
      'Negative cycle makes shortest path undefined (-infinity)',
      'Example: arbitrage, inconsistent constraints',
      'Trace updated node to find actual cycle',
      'Critical: prevent reporting incorrect finite distances',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare Bellman-Ford O(VE) vs Dijkstra O((V+E) log V). When is each preferred despite complexity difference?',
    sampleAnswer:
      'Bellman-Ford is O(VE), slower than Dijkstra O((V+E) log V). For dense graph with V=1000, E=500K: Bellman-Ford is 500M operations, Dijkstra is 5M. Bellman-Ford preferred when: negative weights present (Dijkstra fails), need negative cycle detection, graph is small or sparse with few edges, simpler to implement (no heap needed), distributed systems (edge-based relaxation parallelizes well). Dijkstra preferred when: all weights non-negative (always use if possible), performance critical, large dense graphs. In practice: use Dijkstra if guaranteed non-negative (GPS, network delays), use Bellman-Ford only when negative weights necessary (currency exchange, potential functions). Never use Bellman-Ford if Dijkstra works - speed difference is significant.',
    keyPoints: [
      'Bellman-Ford: O(VE), Dijkstra: O((V+E) log V)',
      'Bellman-Ford when: negative weights, cycle detection',
      'Dijkstra when: non-negative, performance matters',
      'In practice: always prefer Dijkstra if possible',
      'Speed difference significant for large graphs',
    ],
  },
];
