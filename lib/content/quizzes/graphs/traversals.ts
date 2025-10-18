/**
 * Quiz questions for Graph Traversals: BFS and DFS section
 */

export const traversalsQuiz = [
  {
    id: 'q1',
    question:
      'Compare BFS and DFS for graphs. When would you choose one over the other?',
    sampleAnswer:
      'BFS explores level by level using a queue, visiting all neighbors before going deeper. DFS explores as deep as possible using a stack (or recursion), going down one path before backtracking. BFS finds shortest path in unweighted graphs - guaranteed to find closest node first. Use BFS for: shortest path, level-order problems, finding closest node. DFS is better for: detecting cycles, topological sort, exhaustive search (like finding all paths). BFS uses O(width) space for queue, DFS uses O(depth) for stack. In dense graphs, BFS might use more memory. For finding if path exists, either works. For optimization problems, choice depends on whether shortest or any path matters.',
    keyPoints: [
      'BFS: level by level, queue, finds shortest path',
      'DFS: deep first, stack/recursion, exhaustive search',
      'BFS: closest node, level problems',
      'DFS: cycles, topological sort, all paths',
      'Space: BFS O(width), DFS O(depth)',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain cycle detection in graphs. How does the approach differ for directed vs undirected graphs?',
    sampleAnswer:
      'For undirected graphs, use DFS with visited set. If we reach a visited node that is not our immediate parent, there is a cycle - we found two different paths to same node. Track parent to avoid false positive from bidirectional edge. For directed graphs, we need three states: unvisited, visiting (in current DFS path), and visited (completely done). If we reach a node in "visiting" state, there is a cycle - we came back to an ancestor in current path. After exploring a node, mark it visited. The key difference: undirected needs parent tracking, directed needs recursion stack tracking. Both use DFS because we need to track current path.',
    keyPoints: [
      'Undirected: DFS, detect visited node (not parent)',
      'Track parent to avoid false positive',
      'Directed: three states (unvisited, visiting, visited)',
      'Visiting state = in current DFS path',
      'Reach visiting node = cycle found',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through topological sort. Why does it only work for DAGs (Directed Acyclic Graphs)?',
    sampleAnswer:
      'Topological sort produces linear ordering where for every edge u→v, u comes before v. Use DFS: visit all nodes, after finishing a node (all descendants explored), add it to result. Reverse the result for topological order. Why only DAGs? If there is a cycle A→B→C→A, we cannot order them linearly - each should come before the next, creating contradiction. For example, task dependencies: if A depends on B, B on C, C on A, we cannot determine start order. Acyclic ensures no circular dependencies. Applications: task scheduling, course prerequisites, build systems. Kahn algorithm alternative uses BFS and in-degrees - repeatedly remove nodes with zero incoming edges.',
    keyPoints: [
      'Linear ordering: for edge u→v, u before v',
      'DFS: add node after finishing, then reverse',
      'Only DAGs: cycles create ordering contradiction',
      'Example: circular task dependencies impossible',
      'Alternative: Kahn algorithm with BFS and in-degrees',
    ],
  },
];
