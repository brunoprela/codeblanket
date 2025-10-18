/**
 * Quiz questions for Time and Space Complexity section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question: 'Analyze DFS time and space complexity for different structures.',
    sampleAnswer:
      'Trees: Time O(n) visit each node once. Space O(h) for call stack where h is height. Balanced tree: O(log n) space. Skewed tree (linear): O(n) space. Graphs: Time O(V+E) visit each vertex and edge once. Space O(V) for visited set plus O(h) for call stack, worst case O(V). Matrix (grid): Time O(rows × cols) visit each cell. Space O(rows × cols) for visited (can optimize with in-place marking). For example, complete binary tree with 1M nodes: height = log(1M) ≈ 20, space O(20). Linked list tree with 1M nodes: height = 1M, space O(1M). Graph with 1000 vertices, 5000 edges: time O(6000), space O(1000).',
    keyPoints: [
      'Trees: O(n) time, O(h) space',
      'Balanced: O(log n) space, skewed: O(n)',
      'Graphs: O(V+E) time, O(V) space',
      'Matrix: O(rows×cols) time and space',
      'Space depends on recursion depth',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare DFS space for recursive vs iterative across different tree shapes.',
    sampleAnswer:
      'Balanced tree: recursive O(log n), iterative O(log n) - same. Skewed tree: recursive O(n), iterative O(n) - same. Complete tree: recursive O(log n), iterative O(w) where w is max width - iterative can be worse. For example, complete binary tree level k has 2^k nodes. At bottom, width is n/2, so iterative O(n) space. Recursive only O(log n) for depth. General: recursive space = height, iterative space = maximum stack size (can be all children of a level). The key: recursive follows one path at a time (height), iterative may have multiple branches in stack. For DFS, iterative is usually O(h) too. For BFS (queue), iterative is O(width).',
    keyPoints: [
      'Balanced: both O(log n)',
      'Skewed: both O(n)',
      'Complete: recursive O(log n) better',
      'Recursive: O(height), iterative: O(max stack)',
      'DFS iterative usually also O(h)',
    ],
  },
  {
    id: 'q3',
    question:
      'How does visited set affect space complexity? When can you optimize it?',
    sampleAnswer:
      'Visited set: O(V) space for V vertices. Cannot avoid for graphs with cycles (infinite loop without it). Trees: do not need visited (no cycles). Optimization 1: for grids, mark visited in-place (change cell value to avoid separate set). Saves O(rows×cols). Optimization 2: for trees, no visited needed. Optimization 3: for graphs where you can modify input, mark nodes in-place. For example, grid "1"→"0" after visiting, saves visited set. Cannot optimize when: cannot modify input, need to preserve state, need to run multiple DFS (must clear visited each time). Tradeoff: in-place marking destroys input.',
    keyPoints: [
      'Visited set: O(V) space',
      'Trees: no visited needed',
      'Grid: mark in-place saves O(rows×cols)',
      'Cannot optimize: cannot modify, multiple runs',
      'Tradeoff: space vs preserving input',
    ],
  },
];
