/**
 * Quiz questions for Segment Tree Variations section
 */

export const variationsQuiz = [
  {
    id: 'segment-variations-1',
    question:
      'Explain how lazy propagation works and why it is necessary for range updates.',
    hint: 'Think about what happens if you update every element in a range individually.',
    sampleAnswer:
      'Without lazy propagation, updating a range of N elements would take O(N log N) time. Lazy propagation defers updates by marking nodes with pending changes instead of immediately propagating to children. Updates are pushed down only when needed during queries. This makes range updates O(log N).',
    keyPoints: [
      'Defers updates to avoid O(N log N)',
      'Marks nodes with pending changes',
      'Push down only when queried',
      'Range update becomes O(log N)',
    ],
  },
  {
    id: 'segment-variations-2',
    question:
      'What is the difference between a Range Sum Tree and a Range Min Tree?',
    hint: 'Consider the merge operation and the identity element.',
    sampleAnswer:
      'The difference is only in the merge operation. Range Sum Tree uses addition to combine children (left + right), with identity 0. Range Min Tree uses minimum (min (left, right)), with identity infinity. The tree structure and algorithms remain identical - only the combine function changes.',
    keyPoints: [
      'Sum Tree: merge = addition, identity = 0',
      'Min Tree: merge = minimum, identity = ∞',
      'Same structure, different merge function',
    ],
  },
  {
    id: 'segment-variations-3',
    question:
      'How do you handle a Segment Tree that needs to support both range sum and range min queries?',
    hint: 'Think about what information each node must store.',
    sampleAnswer:
      'Store both sum and min at each node. During build and update, compute both values. For queries, return both pieces of information. This doubles the space per node but does not change the time complexity. Each node becomes a struct/object with multiple fields.',
    keyPoints: [
      'Store multiple values per node',
      'Compute all values during build/update',
      'Space: doubles, Time complexity: unchanged',
    ],
  },
];
