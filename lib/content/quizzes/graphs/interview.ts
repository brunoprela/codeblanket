/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize when a problem needs graph algorithms? What keywords or patterns signal this?',
    sampleAnswer:
      'Several signals indicate graph problems. Explicit mentions: "network", "graph", "connections", "dependencies", "path". Implicit: relationships between entities (social connections, prerequisites), grid traversal (islands, word search - grids are implicit graphs), scheduling with dependencies. Keywords like "reachable", "connected", "shortest path", "cycle", "order tasks" suggest graphs. For example, "find shortest path between cities" is clearly graph. "Can you finish all courses given prerequisites" is graph (topological sort). "Count number of islands in grid" is graph (connected components on grid). Ask: are there entities with relationships? Do I need to traverse connections? Is there a network structure?',
    keyPoints: [
      'Explicit: network, graph, connections, dependencies',
      'Implicit: entity relationships, grid traversal',
      'Keywords: reachable, connected, shortest, cycle, order',
      'Prerequisites → topological sort',
      'Grid problems → graph on grid',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your approach to a graph problem in an interview. What questions do you ask?',
    sampleAnswer:
      'First, clarify the graph structure: directed or undirected? Weighted or unweighted? Can it have cycles? How is it represented (adjacency list, matrix, edges list)? Then identify the problem type: shortest path, connectivity, cycle detection, topological sort? Based on type, choose algorithm: BFS for shortest path in unweighted, Dijkstra for weighted, DFS for cycles, Union-Find for connectivity. State complexity: O(V + E) for BFS/DFS. Discuss edge cases: empty graph, disconnected, self-loops. Draw small example: 4 nodes, show algorithm steps. Code with clear structure: build graph, run algorithm, return result. Mention optimizations: early termination, bidirectional search.',
    keyPoints: [
      'Clarify: directed/undirected, weighted, cycles, representation',
      'Identify problem type: path, connectivity, cycle, topological',
      'Choose algorithm based on type',
      'State complexity with reasoning',
      'Draw example, show steps',
      'Code clearly, discuss optimizations',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common pitfalls in graph problems and how do you avoid them?',
    sampleAnswer:
      'First: forgetting to handle disconnected graphs - must iterate through all nodes, not just start from one. Second: not marking visited, causing infinite loops in cycles. Third: marking visited at wrong time in BFS (mark when enqueueing, not dequeueing). Fourth: for undirected graphs, adding edges both directions or checking both. Fifth: off-by-one in adjacency matrix vs list indexing. Sixth: not handling empty graph or single node. Seventh: in topological sort, not checking if all nodes processed (indicates cycle). My strategy: draw the graph, trace algorithm on paper, test with cycle and disconnected cases, verify visited logic, check directed vs undirected handling.',
    keyPoints: [
      'Handle disconnected graphs (check all nodes)',
      'Mark visited to prevent infinite loops',
      'BFS: mark when enqueueing',
      'Undirected: edges both ways',
      'Test: cycles, disconnected, empty graph',
      'Topological: verify all nodes processed',
    ],
  },
];
