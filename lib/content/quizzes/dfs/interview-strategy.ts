/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize a DFS problem? What keywords and patterns signal DFS?',
    sampleAnswer:
      'Keywords: "traverse", "visit all", "path", "connected", "cycle", "component", "island", "region", "validate". Patterns: 1) Tree traversal (inorder, preorder, postorder). 2) Path finding (root-to-leaf). 3) Graph exploration (all nodes, components). 4) Backtracking (permutations, combinations). 5) Cycle detection. 6) Topological sort. Signals: tree/graph structure, need to explore deeply, no "shortest" requirement (else BFS). For example, "validate binary search tree" → tree DFS. "Find all root-to-leaf paths" → DFS with path tracking. "Count connected components" → DFS from each unvisited. "Course schedule" → cycle detection DFS. If problem needs shortest path or level-order → BFS. If exploring all possibilities → DFS.',
    keyPoints: [
      'Keywords: traverse, path, connected, cycle, component',
      'Patterns: tree traversal, paths, exploration',
      'Tree/graph + explore deeply → DFS',
      'Need shortest → BFS',
      'Explore all possibilities → DFS',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your DFS interview approach from recognition to implementation.',
    sampleAnswer:
      'First, recognize DFS from keywords (tree, path, component, cycle). Second, identify pattern (traversal? path sum? cycle?). Third, choose recursive vs iterative (recursive simpler unless deep). Fourth, define base case (None/null, leaf, visited). Fifth, define recursive case (process node, recurse on children/neighbors). Sixth, handle visited set for graphs. Seventh, test with examples and edges. Finally, analyze complexity. For example, "validate BST": recognize tree DFS, pattern is tree property validation, recursive approach, base case is None (True), recursive case check val in range and recurse with updated ranges, test with valid/invalid BSTs, O(n) time O(h) space. Show: recognition, pattern, implementation, testing.',
    keyPoints: [
      'Recognize: keywords, tree/graph',
      'Pattern: traversal, path, cycle, component',
      'Choose: recursive (simpler) or iterative',
      'Define: base case, recursive case',
      'Graphs: add visited set',
      'Test and analyze complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in DFS problems? How do you avoid them?',
    sampleAnswer:
      'First: forgetting visited set in graphs (infinite loop on cycles). Second: wrong base case (None not handled → crash). Third: modifying global state without reset (multiple test cases fail). Fourth: marking visited at wrong time (before vs after recursion matters). Fifth: not handling disconnected graphs (only explore from one start). Sixth: stack overflow on deep trees (use iterative). My strategy: 1) Always check None first. 2) Use visited set for graphs. 3) Mark visited before recursing (avoid re-visiting). 4) For problems needing all paths, backtrack by unmarking. 5) Test: empty, single node, cycle, disconnected. 6) For very deep, consider iterative. Most mistakes from missing visited set or wrong base case.',
    keyPoints: [
      'Mistakes: no visited, wrong base, global state',
      'Visited timing: mark before recursing',
      'Handle: None, cycles, disconnected',
      'Very deep: use iterative',
      'Test: empty, single, cycle, disconnected',
      'Most common: missing visited set',
    ],
  },
];
