/**
 * Quiz questions for Introduction to Graphs section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what a graph is and how it differs from a tree. When would you use a graph over a tree?',
    sampleAnswer:
      'A graph is a collection of nodes (vertices) connected by edges. Unlike trees, graphs can have cycles, multiple paths between nodes, and disconnected components. Trees are special graphs: connected, acyclic, with exactly n-1 edges for n nodes. Use graphs when relationships are not strictly hierarchical - for example, social networks where friendships are bidirectional and can form cycles, or road maps where multiple routes exist between cities. Trees model hierarchical relationships like file systems or org charts. Graphs model peer relationships, networks, dependencies, and any scenario where cycles or multiple connections are natural. Graphs are more general; trees are constrained graphs.',
    keyPoints: [
      'Graph: nodes connected by edges',
      'Can have cycles, multiple paths, disconnected components',
      'Trees: special graphs (connected, acyclic, n-1 edges)',
      'Use graphs: non-hierarchical relationships, cycles',
      'Trees: hierarchical, Graphs: peer relationships',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare adjacency list and adjacency matrix representations. When would you choose each?',
    sampleAnswer:
      'Adjacency list stores for each node a list of its neighbors. Space is O(V + E) where V is vertices, E is edges. Good for sparse graphs where E << V^2. Checking if edge exists is O(degree), adding edge is O(1). Adjacency matrix is VxV grid where matrix[i][j] = 1 if edge exists. Space is O(V^2) regardless of edges. Good for dense graphs or when you frequently check if edge exists (O(1) lookup). For example, social network with 1M users but each connected to ~100: list uses 100M entries, matrix needs 1T entries - list wins. For complete graph where everyone connects to everyone: matrix is better. In practice, most real graphs are sparse, so adjacency list is more common.',
    keyPoints: [
      'List: O(V + E) space, good for sparse',
      'Matrix: O(V^2) space, good for dense',
      'List: O(degree) edge check, Matrix: O(1)',
      'Most real graphs sparse → list preferred',
      'Matrix: when frequent edge existence checks',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe directed vs undirected graphs. Give real-world examples where each is appropriate.',
    sampleAnswer:
      'Directed graphs have edges with direction: A→B does not imply B→A. Used for asymmetric relationships like Twitter follows (I follow you, you might not follow me), web page links, task dependencies (task A must complete before B). Undirected graphs have bidirectional edges: A-B means both A to B and B to A. Used for symmetric relationships like Facebook friendships (mutual), road connections (bidirectional travel), collaboration networks. In code, directed graphs store edges once, undirected store twice (both directions) or check both ways. Directed enables modeling one-way relationships and detecting cycles in dependencies. Undirected is simpler when relationships are naturally symmetric.',
    keyPoints: [
      'Directed: edges have direction (asymmetric)',
      'Examples: Twitter follows, web links, task dependencies',
      'Undirected: bidirectional edges (symmetric)',
      'Examples: friendships, roads, collaborations',
      'Choice depends on relationship symmetry',
    ],
  },
];
