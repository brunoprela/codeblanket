/**
 * Quiz questions for Time and Space Complexity section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question: 'Analyze BFS time and space complexity for trees vs graphs.',
    sampleAnswer:
      'Trees: Time O(n) visit each node once. Space O(w) where w is maximum width. Balanced binary tree: width at bottom level is n/2, so O(n) space worst case. Graphs: Time O(V+E) visit each vertex once, explore each edge once (or twice for undirected). Space O(V) for queue and visited set. For example, complete binary tree 1M nodes: width at level log(1M) ≈ 500K, space O(500K). Graph 1000 vertices, 5000 edges: time O(6000), space O(1000). BFS space often larger than DFS O(height). For balanced tree height log n << width n/2. For skewed tree (linked list): height n, width 1, BFS better than DFS.',
    keyPoints: [
      'Trees: O(n) time, O(w) space (width)',
      'Graphs: O(V+E) time, O(V) space',
      'Balanced tree: width n/2, space O(n)',
      'Skewed tree: width 1, space O(1)',
      'BFS space often > DFS for balanced trees',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare BFS vs DFS space for different tree shapes. Which is more space-efficient?',
    sampleAnswer:
      'Balanced tree: BFS O(w) = O(n/2) = O(n), DFS O(h) = O(log n). DFS more space-efficient. Skewed tree (linked list): BFS O(w) = O(1), DFS O(h) = O(n). BFS more space-efficient. Complete binary tree: width doubles each level, bottom has n/2 nodes. BFS O(n), DFS O(log n). DFS much better. For example, 1M node balanced tree: BFS uses 500K space, DFS uses 20 space (log 1M ≈ 20). For 1M node linked list: BFS uses 1 space, DFS uses 1M space. General rule: balanced/wide trees favor DFS, skewed/narrow trees favor BFS. Most real trees are balanced, so DFS usually more space-efficient.',
    keyPoints: [
      'Balanced: BFS O(n), DFS O(log n) - DFS better',
      'Skewed: BFS O(1), DFS O(n) - BFS better',
      'Complete: BFS O(n/2), DFS O(log n) - DFS much better',
      'Wide trees: DFS better, narrow: BFS better',
      'Most real trees balanced: DFS more efficient',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the queue space in BFS. Why does it grow to width of tree?',
    sampleAnswer:
      'BFS queue holds all nodes at current level before processing next level. Maximum size is width of widest level. For complete binary tree, widest level is bottom (n/2 nodes). Process level k: queue has all 2^k nodes at level k. After processing, queue has all 2^(k+1) nodes at level k+1. Peak queue size occurs at widest level. For example, tree with 4 levels (1+2+4+8=15 nodes): level 0 queue size 1, level 1 size 2, level 2 size 4, level 3 size 8 (peak). The nodes must all wait in queue before being processed. This is why BFS space is O(width) not O(depth). DFS uses stack, only one path active at a time, so O(depth).',
    keyPoints: [
      'Queue holds all nodes at current level',
      'Max size = widest level width',
      'Complete tree: widest is bottom (n/2)',
      'All nodes wait before processing',
      'vs DFS: only one path (depth)',
    ],
  },
];
