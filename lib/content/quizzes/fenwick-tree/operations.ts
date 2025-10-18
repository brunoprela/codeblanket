/**
 * Quiz questions for Core Operations section
 */

export const operationsQuiz = [
  {
    id: 'fenwick-operations-1',
    question:
      'Why is building a Fenwick Tree O(N log N) instead of O(N) like a Segment Tree?',
    hint: 'Think about what happens when you call update() for each element.',
    sampleAnswer:
      'Building by calling update() N times gives O(N log N) because each update is O(log N). There is actually an O(N) build method that fills the tree directly without updates, but it is less commonly taught. The O(N log N) method is simpler and more intuitive, though slightly slower.',
    keyPoints: [
      'Naive build: N elements × O(log N) per update = O(N log N)',
      'Optimized build exists but rarely needed',
      'Still acceptable for most problems',
    ],
  },
  {
    id: 'fenwick-operations-2',
    question:
      'Explain how to compute range_sum(L, R) using only prefix_sum operations.',
    hint: 'This uses the principle of inclusion-exclusion.',
    sampleAnswer:
      'Range sum from L to R is computed as prefix_sum(R) - prefix_sum(L-1). This works because prefix_sum(R) includes everything from 1 to R, and prefix_sum(L-1) includes everything from 1 to L-1. Subtracting removes the unwanted prefix, leaving only elements from L to R.',
    keyPoints: [
      'range_sum(L, R) = prefix_sum(R) - prefix_sum(L-1)',
      'Inclusion-exclusion principle',
      'Both queries are O(log N), so total is O(log N)',
    ],
  },
  {
    id: 'fenwick-operations-3',
    question:
      'What is the update operation actually doing? Does it set a value or add a delta?',
    hint: 'Think about why it is called "add delta" not "set value".',
    sampleAnswer:
      'Fenwick Tree update adds a delta to an index, it does not set an absolute value. To set arr[i] to new_val, you must compute delta = new_val - arr[i], then call update(i, delta). This is because Fenwick Tree stores cumulative sums, not individual values, so it can only support additive updates efficiently.',
    keyPoints: [
      'Updates add a delta, not set a value',
      'To set: delta = new_value - old_value, then update(i, delta)',
      'Fenwick stores cumulative data, not raw values',
    ],
  },
];
