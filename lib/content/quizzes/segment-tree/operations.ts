/**
 * Quiz questions for Core Operations section
 */

export const operationsQuiz = [
  {
    id: 'segment-operations-1',
    question: 'Why is building a Segment Tree O(N) instead of O(N log N)?',
    hint: 'Think about how many times each node is visited during build.',
    sampleAnswer:
      'Building is O(N) because we visit each of the 2N-1 nodes exactly once in a bottom-up or top-down traversal. Although the tree has log N levels, we do not repeat work - each node computes its value from its two children in O(1) time. Total work is O(2N-1) = O(N).',
    keyPoints: [
      'Visit each node exactly once',
      'Total nodes: 2N-1',
      'Each node: O(1) computation',
    ],
  },
  {
    id: 'segment-operations-2',
    question: 'Explain why range query in Segment Tree is O(log N).',
    hint: 'Consider how many nodes you visit at each level.',
    sampleAnswer:
      'At each level of the tree, the query range can intersect at most 4 nodes (2 at the boundaries). Since the tree has O(log N) levels, the total nodes visited is O(4 * log N) = O(log N). The key insight is that we prune branches that are completely inside or outside the query range.',
    keyPoints: [
      'At most 4 nodes per level',
      'Tree height: O(log N)',
      'Pruning reduces work dramatically',
    ],
  },
  {
    id: 'segment-operations-3',
    question:
      'What is the update process in a Segment Tree after modifying a leaf node?',
    hint: 'Think about which nodes are affected by a single element change.',
    sampleAnswer:
      'After updating a leaf, you must update all ancestors up to the root. Each ancestor recomputes its value from its two children. Since the tree height is log N, you update O(log N) nodes. The path from leaf to root visits one node per level.',
    keyPoints: [
      'Update all ancestors up to root',
      'Path from leaf to root: O(log N)',
      'Each ancestor recomputes from children',
    ],
  },
];
