/**
 * Quiz questions for Common BFS Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'What are common BFS patterns? How do you recognize when to use each?',
    sampleAnswer:
      'Pattern 1: Level-order traversal - recognize by "level", "depth", "row by row". Pattern 2: Shortest path - recognize by "minimum steps", "shortest", unweighted graph. Pattern 3: Nearest/closest - recognize by "nearest", "closest", "minimum distance from point". Pattern 4: Multi-source BFS - recognize by multiple start points, "distance from any". Pattern 5: State-space search - recognize by transformations, puzzle, "minimum moves". For example, "print tree level by level" → level-order pattern. "Shortest path from A to B" → shortest path pattern. "01 Matrix: distance to nearest 0" → multi-source BFS from all 0s. "Minimum moves to solve sliding puzzle" → state-space BFS.',
    keyPoints: [
      'Level-order: by level traversal',
      'Shortest path: minimum steps, unweighted',
      'Nearest: closest node/cell',
      'Multi-source: multiple starts',
      'State-space: transformations, puzzles',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe multi-source BFS. How does it differ from single-source?',
    sampleAnswer:
      'Multi-source BFS: start from multiple sources simultaneously, find distances to all. Initialize queue with all sources (distance 0), mark all as visited, run standard BFS. All sources explored at same "level". Example: "01 Matrix - distance to nearest 0": instead of BFS from each cell to find nearest 0 (O(n² × n²)), start BFS from all 0s at once. Queue initially has all 0 cells (distance 0), then explores all cells distance 1 from any 0, then distance 2, etc. Each cell reached first time is at correct minimum distance. Time O(rows×cols) once vs O((rows×cols)²) for single-source from each cell. Pattern: "distance to nearest X" where X is multiple cells.',
    keyPoints: [
      'Start BFS from multiple sources simultaneously',
      'Initialize queue with all sources',
      'All sources at distance 0',
      'Example: distance to nearest 0 in grid',
      'O(V+E) once vs O(V×(V+E)) for V single-source',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through solving "Minimum Knight Moves" with BFS. Why is BFS the right choice?',
    sampleAnswer:
      'Minimum knight moves: chess knight from (0,0) to (x,y), minimum moves? BFS because: unweighted graph (each move costs 1), need shortest path. State space: positions (row, col). Algorithm: queue with (0, 0, 0) (start position, moves=0), visited set, BFS: dequeue (r, c, moves), if target return moves, for each of 8 knight moves: new position (r+dr, c+dc), if not visited: mark visited, enqueue (new_r, new_c, moves+1). First reach of target is minimum moves. For example, (0,0) to (2,1): (0,0)→(1,2) or (2,1) both 1 move, BFS finds immediately. vs DFS: might explore (0,0)→(1,2)→(3,1)→... longer path first. BFS guarantees minimum.',
    keyPoints: [
      'State space: board positions',
      'BFS: unweighted, finds shortest',
      'Queue: (position, moves)',
      '8 knight moves from each position',
      'First reach = minimum moves',
    ],
  },
];
