/**
 * Quiz questions for Introduction to BFS section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain BFS (Breadth-First Search). How does it work and what makes it "breadth-first"?',
    sampleAnswer:
      'BFS explores all nodes at current level before moving to next level. Uses queue (FIFO - First In First Out). Start at root/source, add to queue, loop: dequeue node, process, enqueue unvisited children. "Breadth-first" means explore wide (all neighbors) before deep. For tree [1, [2, [4, 5]], [3]]: BFS visits level 0: 1, level 1: 2,3, level 2: 4,5. Compare DFS: 1→2→4→5→3 (deep first). BFS guarantees shortest path in unweighted graphs - first time you reach node is via shortest path. Time O(V+E) for graphs, O(n) for trees. Space O(w) where w is maximum width. Natural for: level-order traversal, shortest path, nearest neighbors.',
    keyPoints: [
      'Explore level by level, uses queue (FIFO)',
      'Process all neighbors before going deeper',
      'Guarantees shortest path in unweighted',
      'O(V+E) time, O(w) space (width)',
      'Uses: level order, shortest path, nearest',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare BFS vs DFS fundamentally. What is the key algorithmic difference?',
    sampleAnswer:
      'Key difference: data structure. BFS uses queue (FIFO), DFS uses stack (LIFO). This determines exploration order. BFS: dequeue explores oldest added (level by level). DFS: pop explores newest added (deep first). For tree [1, [2, [4, 5]], [3]]: BFS queue [1]→[2,3]→[3,4,5]→[4,5]→[5]→[], visits 1,2,3,4,5. DFS stack [1]→[2,3]→[2,4,5]→[2,4]→[2]→[], visits 1,3,2,5,4 (preorder with right-first). BFS finds shortest path (first arrival), DFS explores deeply (may find longer paths first). Space: BFS O(width), DFS O(height). Implementation: BFS always iterative (queue), DFS recursive or iterative (stack).',
    keyPoints: [
      'BFS: queue (FIFO), DFS: stack (LIFO)',
      'BFS: level by level, DFS: deep first',
      'BFS: shortest path, DFS: explores deeply',
      'BFS space: O(width), DFS: O(height)',
      'BFS always iterative, DFS can be recursive',
    ],
  },
  {
    id: 'q3',
    question:
      'Why does BFS guarantee shortest path in unweighted graphs? Prove or explain intuition.',
    sampleAnswer:
      'BFS explores nodes in order of increasing distance from source. Queue ensures nodes at distance d are all processed before nodes at distance d+1. When node first discovered, it is via shortest path because all shorter paths explored already. Proof by contradiction: suppose node X reached at distance d+1, but shorter path of length d exists. Then node Y on that path at distance d-1 would have enqueued X at distance d, contradiction (X already discovered). For example, graph A→B, A→C, B→D, C→D: BFS from A discovers B,C at distance 1, then D at distance 2 via both paths. First discovery is shortest. DFS may find D via A→C→D before A→B→D, no guarantee. This is why BFS is standard for shortest path in unweighted graphs.',
    keyPoints: [
      'Explores in order of increasing distance',
      'Queue processes distance d before d+1',
      'First discovery = shortest path',
      'Proof: shorter path would discover earlier',
      'DFS no guarantee: may explore long paths first',
    ],
  },
];
