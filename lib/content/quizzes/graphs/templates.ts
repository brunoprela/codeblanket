/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the BFS template. Why do we use a queue and how does the visited set prevent issues?',
    sampleAnswer:
      'BFS uses queue for level-order traversal: process all nodes at distance d before distance d+1. Start by adding source to queue and marking visited. While queue not empty: dequeue node, process it, enqueue unvisited neighbors and mark them visited. Queue ensures FIFO - first discovered are first processed, guaranteeing level-order. Visited set prevents two issues: infinite loops (cycles cause revisiting) and redundant work (processing same node multiple times). Mark visited when enqueueing, not when dequeueing - prevents adding same node to queue multiple times. This template finds shortest path in unweighted graphs because we explore by distance from source.',
    keyPoints: [
      'Queue: FIFO for level-order traversal',
      'Mark visited when enqueueing',
      'Prevents: infinite loops and redundant work',
      'Process level d before d+1',
      'Finds shortest path in unweighted graphs',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the DFS recursive template. How does the visited set interact with recursion?',
    sampleAnswer:
      'DFS explores as deep as possible before backtracking. Mark current node visited, process it, then recursively visit each unvisited neighbor. The recursion stack implicitly tracks current path - when a recursive call returns, we backtrack to try other branches. Visited set ensures we do not revisit nodes, preventing infinite loops in cycles. Mark visited before recursing to neighbors - this is the "choose" step. Unlike backtracking where we unmark (unchoose), in graph traversal we keep nodes marked because we do not need to revisit. The combination of recursion (for path) and visited set (for seen nodes) enables complete graph exploration without redundancy.',
    keyPoints: [
      'Recursive: explore deep first',
      'Mark visited before recursing to neighbors',
      'Recursion stack tracks current path',
      'Visited set prevents revisiting (no unmark needed)',
      'Returns = backtrack to try other branches',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the topological sort DFS template. Why do we add to result after exploring neighbors?',
    sampleAnswer:
      'Topological sort DFS visits all nodes, recursively exploring neighbors first, then adding current node to result after finishing. This post-order traversal ensures dependencies are added before dependents. Reverse the result to get topological order. Why add after exploring? Consider edge A→B: we must output A before B. When visiting A, we recurse to B first. B finishes and gets added. Then A finishes and gets added. Result has [B, A], reverse gives [A, B] - correct! Adding before recursing would give wrong order. The key insight: node is added when all descendants are processed - exactly when all dependencies are satisfied. This is why DFS naturally produces reverse topological order.',
    keyPoints: [
      'Post-order: add after exploring all neighbors',
      'Ensures dependencies added before dependents',
      'Reverse result for topological order',
      'Node added when all descendants processed',
      'DFS naturally produces reverse topological order',
    ],
  },
];
