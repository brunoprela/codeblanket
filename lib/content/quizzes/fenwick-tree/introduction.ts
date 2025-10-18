/**
 * Quiz questions for Introduction to Fenwick Tree section
 */

export const introductionQuiz = [
  {
    id: 'fenwick-intro-1',
    question:
      'Why would you choose a Fenwick Tree over a simple prefix sum array when you need both queries and updates?',
    hint: 'Think about what happens when you update a value in a prefix sum array.',
    sampleAnswer:
      'With a simple prefix sum array, queries are O(1) but updates are O(N) because you must recalculate all prefix sums after the update. Fenwick Tree balances this tradeoff by making both operations O(log N), which is much better when you have many updates.',
    keyPoints: [
      'Prefix sum array: O(1) query, O(N) update',
      'Fenwick Tree: O(log N) query, O(log N) update',
      'Better when you have frequent updates',
    ],
  },
  {
    id: 'fenwick-intro-2',
    question:
      'What is the key limitation of Fenwick Tree compared to Segment Tree, and why does this limitation exist?',
    hint: 'Consider what mathematical property is needed for a Fenwick Tree to work.',
    sampleAnswer:
      'Fenwick Tree cannot handle operations without an inverse, like min/max or GCD. This is because Fenwick Tree relies on being able to add and subtract ranges. For addition, subtraction is the inverse. But there is no inverse for min or max - you cannot "un-min" a value.',
    keyPoints: [
      'Fenwick Tree needs an inverse operation',
      'Works for: addition (inverse: subtraction), XOR (inverse: XOR)',
      'Does not work for: min, max, GCD (no inverse)',
    ],
  },
  {
    id: 'fenwick-intro-3',
    question:
      'When should you prefer Fenwick Tree over Segment Tree for a range sum problem?',
    hint: 'Think about code complexity and implementation time.',
    sampleAnswer:
      'Use Fenwick Tree when the problem only needs range sums or prefix sums with updates. Fenwick Tree is simpler to implement (around 20 lines vs 50+ for Segment Tree), uses less memory, and is less error-prone in interviews. Only use Segment Tree when you need operations like min/max or lazy propagation.',
    keyPoints: [
      'Fenwick Tree: simpler, fewer lines of code',
      'Same time complexity for range sums',
      'Less memory usage',
      'Easier to implement correctly under time pressure',
    ],
  },
];
