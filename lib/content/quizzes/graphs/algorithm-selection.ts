/**
 * Quiz questions for Graph Algorithm Selection Guide section
 */

export const algorithmselectionQuiz = [
  {
    id: 'q1',
    question:
      'When would you choose BFS over DFS for finding a path? Give specific scenarios.',
    hint: 'Think about what kind of path you need.',
    sampleAnswer:
      'Choose BFS when you need the SHORTEST path in an unweighted graph. BFS explores level-by-level, so the first time you reach the destination, you have found the shortest path. DFS explores deeply and might find A longer path first. Example scenarios: 1) Social network - degrees of separation (shortest connection), 2) Maze solving where you want minimum steps, 3) Finding minimum number of moves in a game. However, choose DFS when you just need ANY path (not shortest), when detecting cycles, for topological sort, or when memory is constrained and graph is very wide.',
    keyPoints: [
      'BFS guarantees shortest path in unweighted graphs',
      'BFS explores layer by layer',
      'Use for: minimum steps, degrees of separation',
      'DFS better for: any path, cycles, memory constraints',
      'Key difference: BFS finds shortest, DFS finds any',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain when Dijkstra fails and why Bellman-Ford is needed. Give a concrete example.',
    hint: 'Think about edge weights and what happens with negative values.',
    sampleAnswer:
      'Dijkstra fails with negative edge weights because it assumes once a node is visited with a shortest path, no future path can be shorter. Example: Graph with edges A→B (weight 5), A→C (weight 2), C→B (weight -10). Dijkstra would: 1) Visit A (dist 0), 2) Visit C (dist 2), 3) Visit B via A directly (dist 5), mark B as done. It misses that A→C→B gives distance -8, which is shorter! Bellman-Ford works because it relaxes ALL edges V-1 times, allowing negative weights to propagate. It can also detect negative cycles (where total cycle weight < 0), which Dijkstra cannot handle.',
    keyPoints: [
      'Dijkstra assumes processed nodes have final shortest distance',
      'Negative weights can create shorter paths discovered later',
      'Example: direct path seems shorter, but detour with negative weight is better',
      'Bellman-Ford relaxes all edges V-1 times',
      'Bellman-Ford can detect negative cycles',
    ],
  },
  {
    id: 'q3',
    question:
      'How does A* improve on Dijkstra? When would you NOT want to use A*?',
    hint: 'Think about the heuristic function and when you have goal information.',
    sampleAnswer:
      'A* improves Dijkstra by using a heuristic function h(n) that estimates distance to goal. It prioritizes nodes that are both close to start (g score) AND close to goal (h score), using f(n) = g(n) + h(n). This can dramatically reduce explored nodes. Example: Finding path in a 1000×1000 grid. Dijkstra explores nodes in all directions equally, A* with Manhattan distance focuses toward the goal. However, DON NOT use A* when: 1) You don not know the goal location in advance, 2) You need paths to ALL nodes (use Dijkstra), 3) You cannot devise a good admissible heuristic, 4) Graph structure doesn not support distance estimates. A* is specifically for single-source, single-destination with geometric/spatial properties.',
    keyPoints: [
      'A* = Dijkstra + heuristic toward goal',
      'Uses f(n) = g(n) + h(n) for priority',
      'Reduces explored nodes when good heuristic exists',
      'Don not use when: no single goal, need all paths, no heuristic',
      'Best for: grids, maps, spatial problems',
    ],
  },
];
