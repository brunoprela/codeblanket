/**
 * Quiz questions for Fenwick Tree vs Segment Tree section
 */

export const comparisonQuiz = [
  {
    id: 'fenwick-comparison-1',
    question:
      'Explain why Fenwick Tree cannot handle range minimum queries (RMQ) while Segment Tree can.',
    hint: 'Think about the inverse operation requirement.',
    sampleAnswer:
      'Fenwick Tree requires an inverse operation to compute range queries using prefix subtraction. For sums, range(L,R) = prefix(R) - prefix(L-1) works because subtraction is the inverse of addition. But for min, there is no inverse - you cannot "un-min" values. Segment Tree stores ranges directly without needing inverses, so it can handle min/max.',
    keyPoints: [
      'Fenwick needs inverse operation',
      'Min/max have no inverse',
      'Segment Tree stores ranges directly',
    ],
  },
  {
    id: 'fenwick-comparison-2',
    question:
      'When would you choose the simpler Fenwick Tree over Segment Tree in an interview?',
    hint: 'Consider time pressure and implementation complexity.',
    sampleAnswer:
      'Choose Fenwick Tree when the problem only needs sum queries with updates and you are under time pressure. Fenwick Tree is 20 lines vs 50+ for Segment Tree, easier to implement correctly, and has the same time complexity. Only use Segment Tree when you need operations Fenwick cannot do, like min/max or lazy propagation.',
    keyPoints: [
      'Fenwick: faster to code correctly',
      'Same O(log N) complexity for sums',
      'Less prone to implementation bugs',
      'Only sacrifice: cannot do min/max',
    ],
  },
  {
    id: 'fenwick-comparison-3',
    question:
      'What advantage does Segment Tree have in terms of memory and build time?',
    hint: 'Compare the build complexity and space usage.',
    sampleAnswer:
      'Segment Tree actually builds in O(N) time, while Fenwick Tree takes O(N log N) with the standard build method. However, Segment Tree uses O(4N) space versus O(N) for Fenwick. So Segment Tree is faster to build but uses more memory. For most problems, the O(N log N) build is acceptable.',
    keyPoints: [
      'Segment Tree: O(N) build, O(4N) space',
      'Fenwick Tree: O(N log N) build, O(N) space',
      'Trade-off: build speed vs memory',
    ],
  },
];
