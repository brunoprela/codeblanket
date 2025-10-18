/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the Top K template. Why maintain heap of size K, and when do you pop?',
    sampleAnswer:
      'Top K template maintains min heap of size K for K largest elements. Initialize empty heap. For each element: push to heap, if heap size exceeds K, pop (removes smallest). After processing all, heap contains K largest. We pop when size exceeds K because we only need K elements - the smallest of K largest is the threshold for entry. Popping smallest maintains only elements that beat threshold. At end, heap has exactly K largest. Time O(n log k) - each of n elements involves push and maybe pop, both O(log k). This works because min heap root is smallest of K largest - perfect boundary. For K smallest, use max heap and same logic.',
    keyPoints: [
      'Min heap size K for K largest',
      'Push each element, pop if size > K',
      'Pop smallest when exceeds K',
      'Heap root = threshold for entry',
      'O(n log k) time',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the two-heap median template. How do you balance the heaps and when do you rebalance?',
    sampleAnswer:
      'Two-heap median uses max heap for lower half, min heap for upper half. To add element: compare with max heap root (largest of lower). If less or equal, add to max heap, else add to min heap. Then rebalance: if size difference exceeds 1, move root from larger to smaller heap. Median: if sizes equal, average of roots. If one larger, root of larger heap. Rebalance ensures heaps differ by at most 1 element so median is always at boundary. Key invariant: all elements in max heap ≤ all elements in min heap. This is maintained by comparing with max heap root before inserting. Add is O(log n), median is O(1).',
    keyPoints: [
      'Max heap: lower half, Min heap: upper half',
      'Compare with max heap root to decide heap',
      'Rebalance if difference > 1',
      'Median: avg if equal, root of larger if not',
      'Add O(log n), median O(1)',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the K-way merge template. What information do you store in the heap besides values?',
    sampleAnswer:
      'K-way merge heap stores tuples: (value, list index, position in list). Initialize heap with (first element, list index, 0) for each list. Pop smallest tuple, add value to result. Push next element from that list: (lists[list_idx][pos+1], list_idx, pos+1) if exists. Repeat until heap empty. We store indices because we need to know which list to pull next element from. Python heapq compares tuples by first element (value), so smallest value is at root. If values equal, uses list index as tiebreaker. This enables merging K sorted lists in O(n log k) where n is total elements. Without heap, comparing K heads each time would be O(nk).',
    keyPoints: [
      'Store (value, list_idx, position) tuples',
      'Initialize with first from each list',
      'Pop smallest, push next from that list',
      'Indices tell which list to pull from',
      'O(n log k) vs O(nk) without heap',
    ],
  },
];
