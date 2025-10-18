/**
 * Quiz questions for Segment Tree Structure section
 */

export const structureQuiz = [
  {
    id: 'segment-structure-1',
    question:
      'Why do we typically allocate 4N space for a Segment Tree when it has 2N-1 nodes?',
    hint: 'Think about how the tree is stored in an array.',
    sampleAnswer:
      'We allocate 4N to simplify implementation and handle worst-case scenarios. When storing a tree in an array with indices 2*i and 2*i+1 for children, some array positions remain unused. The 4N bound ensures we never run out of space regardless of how the tree is built, avoiding index calculations.',
    keyPoints: [
      'Simplifies implementation',
      'Handles worst-case array storage',
      'Avoids complex index calculations',
    ],
  },
  {
    id: 'segment-structure-2',
    question:
      'Explain how parent-child relationships work in the array representation of a Segment Tree.',
    hint: 'Consider the indices used for left and right children.',
    sampleAnswer:
      'In array representation, if a node is at index i, its left child is at 2*i and right child is at 2*i+1. The parent of node i is at i//2. This binary heap-like structure allows efficient navigation without storing explicit pointers.',
    keyPoints: [
      'Left child: 2*i, Right child: 2*i+1',
      'Parent: i // 2',
      'Similar to binary heap indexing',
    ],
  },
  {
    id: 'segment-structure-3',
    question:
      'How many leaf nodes and internal nodes does a Segment Tree have for an array of size N?',
    hint: 'Leaf nodes represent individual elements.',
    sampleAnswer:
      'A Segment Tree has exactly N leaf nodes (one for each array element) and N-1 internal nodes. The total is 2N-1 nodes. Each internal node represents the union of two child segments, and you need N-1 such combinations to cover all levels.',
    keyPoints: ['Leaf nodes: N', 'Internal nodes: N-1', 'Total: 2N-1 nodes'],
  },
];
