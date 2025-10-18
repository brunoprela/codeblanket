/**
 * Quiz questions for Common Heap Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the Top K pattern with min heap. Why use a min heap of size K rather than a max heap?',
    sampleAnswer:
      'For Top K largest elements, we maintain a min heap of size K. The key insight: the smallest of the K largest elements is the threshold - anything smaller than this is not in Top K. As we iterate through elements, if element is larger than heap top (smallest in our K), it replaces it by popping and pushing. The min heap of size K efficiently tracks the K largest because the root is the boundary - elements must beat this to enter Top K. A max heap would not work because we would not know the threshold (smallest of K largest). Time is O(n log k) - process n elements, each heap operation is log k. Space is O(k) for the heap. This is better than sorting entire array O(n log n).',
    keyPoints: [
      'Top K largest: use min heap of size K',
      'Heap root = threshold (smallest of K largest)',
      'Element > root: pop and push',
      'Max heap would not give threshold',
      'O(n log k) time, O(k) space',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the two-heap pattern for finding running median. Why do you need both a max heap and a min heap?',
    sampleAnswer:
      'For running median, we split elements into two halves: max heap stores smaller half (root is largest of small), min heap stores larger half (root is smallest of large). The median is at the boundary between halves. If heaps balanced, median is average of two roots. If one heap larger by 1, median is root of larger heap. We need both heaps because median is middle value - we need to access largest of lower half and smallest of upper half in O(1). With two heaps, adding element is O(log n) - insert to appropriate heap and rebalance if needed. Finding median is O(1) - just look at roots. Without two heaps, we would need O(n log n) sorting for each median query.',
    keyPoints: [
      'Max heap: lower half, min heap: upper half',
      'Median at boundary between halves',
      'Balanced: avg of roots, Unbalanced: root of larger',
      'Need both: access both halves in O(1)',
      'Add O(log n), median O(1)',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the K-way merge pattern. How does a heap enable efficient merging of K sorted lists?',
    sampleAnswer:
      'K-way merge uses a min heap to merge K sorted lists efficiently. Initialize heap with first element from each list (along with list index and position). The heap root is always the smallest among K candidates. Pop root, add to result, push next element from that list to heap. Repeat until all elements processed. This works because at each step, we need the smallest of K candidates - heap gives this in O(log k). Without heap, comparing K elements would be O(k) per element, giving O(nk) total. With heap, each of n elements involves O(log k) heap operation, giving O(n log k). For merging k sorted arrays of total n elements, this is optimal. The heap efficiently maintains sorted order across K sources.',
    keyPoints: [
      'Initialize heap with first from each list',
      'Pop min, add to result, push next from that list',
      'Heap root = smallest of K candidates',
      'O(n log k) vs O(nk) without heap',
      'Optimal for K-way merge',
    ],
  },
];
