/**
 * Quiz questions for Introduction to Segment Trees section
 */

export const introductionQuiz = [
  {
    id: 'segment-intro-1',
    question:
      'What is the main advantage of Segment Tree over a simple array with recalculation?',
    hint: 'Consider the time complexity of range queries.',
    sampleAnswer:
      'Segment Tree reduces range query time from O(N) to O(log N). For example, finding the sum of a range in a simple array requires iterating through all elements, but Segment Tree precomputes and stores ranges, allowing you to combine at most log N nodes to answer any query.',
    keyPoints: [
      'Range queries: O(N) → O(log N)',
      'Precomputes and stores range information',
      'Combines at most log N nodes per query',
    ],
  },
  {
    id: 'segment-intro-2',
    question: 'Why would you choose Segment Tree over Fenwick Tree?',
    hint: 'Think about operations that do not have an inverse.',
    sampleAnswer:
      'Use Segment Tree when you need operations without an inverse, like min, max, or GCD. Fenwick Tree only works for operations with inverses (like addition/subtraction). Segment Tree also supports more complex operations and lazy propagation for range updates, making it more versatile despite being harder to implement.',
    keyPoints: [
      'Supports min, max, GCD (no inverse needed)',
      'Lazy propagation for range updates',
      'More versatile but more complex code',
    ],
  },
  {
    id: 'segment-intro-3',
    question: 'When should you use Sqrt Decomposition instead of Segment Tree?',
    hint: 'Consider implementation complexity and time limits.',
    sampleAnswer:
      'Use Sqrt Decomposition when time limits are generous and you want simpler code. Sqrt Decomposition is much easier to implement (20 lines vs 50+ for Segment Tree) and has O(√N) complexity, which is acceptable for N up to 10^6. Use Segment Tree when you need O(log N) or the problem requires sophisticated range operations.',
    keyPoints: [
      'Sqrt Decomposition: simpler, O(√N)',
      'Segment Tree: more complex, O(log N)',
      'Choose based on time limits and problem complexity',
    ],
  },
];
