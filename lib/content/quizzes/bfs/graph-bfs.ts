/**
 * Quiz questions for BFS on Graphs section
 */

export const graphbfsQuiz = [
  {
    id: 'q1',
    question:
      'Explain BFS on graphs. What must you track that trees do not require?',
    sampleAnswer:
      'Graph BFS requires visited set because graphs have cycles. Without visited, infinite loop: A→B→C→A→B... Tree BFS does not need visited (no cycles, clear parent-child). Graph BFS: queue with start, visited set, loop: dequeue node, for each unvisited neighbor: mark visited, enqueue. For example, graph 0→1, 1→2, 2→0, 0→3: start 0, queue=[0], visit 0, enqueue 1,3, queue=[1,3], visit 1, enqueue 2, queue=[3,2], visit 3, queue=[2], visit 2, try 0 (visited, skip). Mark visited WHEN ENQUEUING not when dequeuing (avoid duplicates in queue). Time O(V+E), space O(V) for visited and queue.',
    keyPoints: [
      'Graph needs visited set (cycles exist)',
      'Tree: no cycles, no visited needed',
      'Mark visited when enqueuing',
      'Prevents: infinite loops, duplicate queue entries',
      'O(V+E) time, O(V) space',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe connected components using BFS. How does it compare to DFS for this problem?',
    sampleAnswer:
      'Connected components: count disconnected groups. Algorithm: for each unvisited node, BFS to mark entire component, count++. For example, graph nodes {0,1,2,3,4,5}, edges 0-1, 2-3-4: BFS from 0 marks 0,1 (component 1), BFS from 2 marks 2,3,4 (component 2), node 5 alone (component 3). Total 3 components. BFS vs DFS for components: both O(V+E), both work equally well. BFS uses queue, iterative only. DFS uses stack or recursion, simpler code. Choice does not matter for correctness or complexity. Preference: DFS slightly simpler (recursive), but BFS fine too. This is one problem where BFS and DFS are equivalent.',
    keyPoints: [
      'Count: BFS from each unvisited node',
      'Each BFS marks one component',
      'BFS vs DFS: both O(V+E), equivalent',
      'BFS: queue, iterative. DFS: recursive',
      'Either works, DFS slightly simpler',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through BFS on undirected vs directed graphs. What changes?',
    sampleAnswer:
      'Undirected: edge A-B means A→B and B→A. BFS explores both directions. For cycle detection, need to track parent to avoid revisiting immediate predecessor. Directed: edge A→B only allows A to B. BFS explores one direction only. Example undirected: 0-1-2, BFS from 0: visit 0, enqueue 1, visit 1, try 0 (parent, skip) enqueue 2, visit 2, try 1 (visited, skip). Example directed: 0→1→2, same but 2 cannot reach 1 (no back edge). For shortest path, both work same way. For cycle detection, directed needs three states (visiting/visited), undirected needs parent tracking. BFS complexity same for both: O(V+E). The key: undirected treats each edge as bidirectional.',
    keyPoints: [
      'Undirected: edge A-B is both directions',
      'BFS explores both ways, track parent',
      'Directed: edge A→B is one direction only',
      'Undirected cycle: needs parent tracking',
      'Both: O(V+E), BFS works same',
    ],
  },
];
