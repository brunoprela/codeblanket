/**
 * Quiz questions for Heap Operations in Detail section
 */

export const operationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the bubble-up operation for insertion. Why is it O(log n) and not O(n)?',
    sampleAnswer:
      'Bubble-up inserts element at the end of the heap (to maintain complete tree property), then compares with parent. If heap property violated (child greater than parent in max heap), swap with parent and repeat. This continues up the tree until heap property restored or reaching root. It is O(log n) because in a complete binary tree of n nodes, height is log n. We move up at most one level per comparison, so at most log n comparisons and swaps. We do not visit all n nodes, only one path from leaf to root. For example, in a heap of 1000 nodes (height 10), insertion takes at most 10 comparisons, not 1000.',
    keyPoints: [
      'Insert at end, then bubble up',
      'Compare with parent, swap if violates',
      'Move up one path from leaf to root',
      'Height is log n in complete tree',
      'O(log n) comparisons, not O(n)',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the bubble-down operation for deletion. Why do we swap with the smaller child in a min heap?',
    sampleAnswer:
      'Bubble-down removes root (the min), moves last element to root, then restores heap property. At each node, compare with children. In min heap, swap with the smaller child if current is greater than smaller child. We swap with smaller child because after swapping, that child becomes parent - it must be less than the other child to maintain heap property. If we swapped with larger child, the other child would be less than its new parent, violating heap property. This continues down until heap property restored or reaching leaf. Like bubble-up, it is O(log n) because we traverse one path down the tree, at most log n levels.',
    keyPoints: [
      'Remove root, move last element to root',
      'Bubble down: compare with children',
      'Swap with smaller child (min heap)',
      'Smaller ensures heap property maintained',
      'O(log n): one path down',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through how to implement a max heap using Python heapq (which only provides min heap). Why does negating work?',
    sampleAnswer:
      'Python heapq is min heap only. To simulate max heap, negate all values before insertion: instead of pushing x, push -x. When popping, negate the result: instead of popping x, pop and return -x. This works because negation reverses the ordering: the smallest negative value corresponds to the largest positive value. For example, values [3, 1, 5] become [-3, -1, -5]. Min of negatives is -5, which corresponds to max of originals which is 5. When we pop -5 and negate, we get 5 - the max. The heap structure and operations remain the same, we just transform values to reverse the comparison order.',
    keyPoints: [
      'Python heapq only provides min heap',
      'Max heap: negate values before insertion',
      'Pop and negate to get max',
      'Negation reverses ordering',
      'Min of negatives = max of originals',
    ],
  },
];
