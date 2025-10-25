/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'heap-templates-1',
    question:
      'Why do we use a min heap of size k to find the k largest elements?',
    hint: 'Think about what gets removed when the heap exceeds size k.',
    sampleAnswer:
      'A min heap of size k keeps the k largest elements because when we add a new element and the heap exceeds k, we remove the smallest element. This ensures only the k largest elements remain. The smallest of these k elements (heap root) is larger than all discarded elements.',
    keyPoints: [
      'Min heap removes smallest when size > k',
      'Keeps k largest elements',
      'Root is kth largest element',
    ],
  },
  {
    id: 'heap-templates-2',
    question:
      'Explain the two-heap technique for finding the median of a stream.',
    hint: 'Consider how you split elements to maintain balance.',
    sampleAnswer:
      'Use a max heap for the smaller half and a min heap for the larger half. Keep them balanced (size difference ≤ 1). The median is either the root of the larger heap (if sizes differ) or the average of both roots (if sizes are equal). This maintains O(log N) insertions and O(1) median queries.',
    keyPoints: [
      'Max heap: smaller half, Min heap: larger half',
      'Balance heaps: size difference ≤ 1',
      'Median from roots: O(1) access',
    ],
  },
  {
    id: 'heap-templates-3',
    question: 'How do you implement a max heap in Python using heapq?',
    hint: 'Python heapq only provides min heap.',
    sampleAnswer:
      'Negate all values when pushing to the heap and negate them again when popping. This converts max heap operations to min heap operations. For example, to add 5, push -5. When popping -5, return 5. This works because -max (values) = min(-values).',
    keyPoints: [
      'Python heapq is min heap only',
      'Negate values: push -value, return -popped',
      '-max (x) = min(-x)',
    ],
  },
];
