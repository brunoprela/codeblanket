/**
 * Quiz questions for Common Graph Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the connected components pattern. Why do we need to check all nodes, not just start from one?',
    sampleAnswer:
      'Connected components finds all separate groups in an undirected graph. We iterate through all nodes, running DFS or BFS from each unvisited node. Each DFS/BFS explores one complete component. We need to check all nodes because the graph might be disconnected - not all nodes reachable from one starting point. For example, social network might have isolated friend groups with no connections between them. Starting from one node only finds that person connected group. By checking all nodes and tracking visited, we ensure we discover all components. Each time we start a new DFS from unvisited node, we have found a new component. Count increases with each new starting point.',
    keyPoints: [
      'Find all separate groups in graph',
      'DFS/BFS from each unvisited node',
      'Graph might be disconnected',
      'One start point only finds one component',
      'New DFS start = new component found',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe Kahn algorithm for topological sort. How does tracking in-degrees help?',
    sampleAnswer:
      'Kahn algorithm uses BFS and tracks in-degrees (number of incoming edges) for each node. Start with nodes having in-degree 0 (no dependencies) - add them to queue. Process queue: remove node, add to result, decrease in-degree of all neighbors by 1. If a neighbor in-degree becomes 0, add it to queue. This ensures we only process a node after all its prerequisites are processed. The in-degree tracking tells us when all dependencies are satisfied. If we finish and some nodes remain unprocessed, there is a cycle. For task scheduling, in-degree 0 means task has no prerequisites and can start immediately. As we complete tasks, dependent tasks become ready. This is intuitive and easier to understand than DFS topological sort.',
    keyPoints: [
      'BFS with in-degree tracking',
      'Start with in-degree 0 nodes (no dependencies)',
      'Process node, decrease neighbor in-degrees',
      'In-degree 0 = all dependencies satisfied',
      'Unprocessed nodes at end = cycle exists',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through bipartite checking with coloring. Why does odd-length cycle prevent bipartiteness?',
    sampleAnswer:
      'Bipartite graph can be colored with two colors such that no adjacent nodes have same color. Use BFS/DFS: color starting node with color 0, color all neighbors with color 1, their neighbors with color 0, etc. If we ever try to color a node but it already has different color, graph is not bipartite. Odd-length cycle prevents bipartiteness because as we alternate colors around cycle, we end up trying to give same node two different colors. For example, triangle ABC: A is color 0, B is color 1, C is color 0, but C connects to A (also color 0) - conflict! Even cycles work: alternate colors around cycle ends correctly. Applications: matching problems, scheduling with conflicts.',
    keyPoints: [
      'Two-color the graph, no adjacent same color',
      'BFS/DFS: alternate colors',
      'Already-colored node with different color = not bipartite',
      'Odd cycle: colors conflict when cycle closes',
      'Even cycles: colors alternate correctly',
    ],
  },
];
