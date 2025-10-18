/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'segment-complexity-1',
    question:
      'Why does Segment Tree use O(4N) space instead of O(2N-1) for 2N-1 nodes?',
    hint: 'Consider how we store the tree in an array.',
    sampleAnswer:
      'We use O(4N) for simplicity. When using array indexing (left child at 2*i, right child at 2*i+1), some positions remain unused. The tight bound is closer to 2N, but 4N is a safe upper bound that avoids careful analysis and makes implementation simpler. The space is still O(N).',
    keyPoints: [
      'Tight bound: ~2N, Safe bound: 4N',
      'Array storage leaves gaps',
      'Simplifies implementation',
    ],
  },
  {
    id: 'segment-complexity-2',
    question:
      'Compare the build time of Segment Tree (O(N)) versus Fenwick Tree (O(N log N)).',
    hint: 'Think about what each build process does.',
    sampleAnswer:
      'Segment Tree builds in O(N) by visiting each of its 2N-1 nodes once. Fenwick Tree builds in O(N log N) by calling update() N times, where each update is O(log N). In practice, both are fast enough, but Segment Tree has a faster build. Fenwick Tree compensates with less space.',
    keyPoints: [
      'Segment Tree: O(N) build, O(4N) space',
      'Fenwick Tree: O(N log N) build, O(N) space',
      'Tradeoff: build time vs space',
    ],
  },
  {
    id: 'segment-complexity-3',
    question: 'What is the space complexity of a 2D Segment Tree?',
    hint: 'Consider nesting two segment trees.',
    sampleAnswer:
      'A 2D Segment Tree for an M×N matrix uses O(M*N) space if implemented efficiently, or up to O(16*M*N) with simple array allocation. The structure is a tree of trees - each node of the outer tree contains an inner tree. Time complexity for queries/updates is O(log M * log N).',
    keyPoints: [
      'Efficient: O(M*N), Simple: O(16*M*N)',
      'Tree of trees structure',
      'Query/Update: O(log M * log N)',
    ],
  },
];
