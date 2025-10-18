/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare the complexity of heapify (building a heap from array) vs inserting elements one by one. Why is heapify O(n)?',
    sampleAnswer:
      'Inserting n elements one by one takes O(n log n) - each insert is O(log n) and we do n inserts. Heapify builds heap from array in O(n) by starting from bottom level and bubbling down. The key insight: most nodes are near bottom with short bubble-down distance. Half the nodes are leaves (no bubbling), quarter are one level up (1 step), eighth are two levels up (2 steps), etc. The work sum is n/2×0 + n/4×1 + n/8×2 + ... This geometric series sums to O(n), not O(n log n). Heapify is faster for batch initialization. Use heapify when you have all elements upfront. Use repeated insert when adding elements dynamically.',
    keyPoints: [
      'Insert one by one: O(n log n)',
      'Heapify: O(n) by bubbling down from bottom',
      'Most nodes near bottom, short bubble distance',
      'Work sum is geometric series = O(n)',
      'Use heapify for batch, insert for dynamic',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why Top K problems with heap are O(n log k) not O(n log n). How does heap size affect complexity?',
    sampleAnswer:
      'Top K uses heap of size k, not n. Each heap operation (push/pop) takes O(log k) where k is heap size, not O(log n). We process n elements, each taking O(log k) heap operation, giving O(n log k) total. For example, finding top 10 in million elements: O(million × log 10) ≈ O(million × 3.3) - much better than sorting O(million × log million) ≈ O(million × 20). The heap size constraint is crucial. If k is small relative to n, we get near-linear performance. If k = n, it degrades to O(n log n) same as sorting. This is why Top K pattern is powerful - we optimize by bounding heap size.',
    keyPoints: [
      'Heap size k, not n',
      'Each operation: O(log k)',
      'Total: O(n log k) for n elements',
      'Small k: near-linear performance',
      'k = n: degrades to O(n log n)',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the space complexity of different heap operations. When do we use O(n) space vs O(k)?',
    sampleAnswer:
      'Space complexity depends on heap contents. Full heap with all n elements: O(n) space. Top K heap: O(k) space - we only maintain k elements at any time. Two-heap median: O(n) space - we store all elements seen so far split across two heaps. K-way merge: O(k) space for heap plus O(n) for result. Heap operations themselves use O(1) extra space for iterative, O(log n) for recursive due to call stack. The key question: do we store all elements or subset? Top K and K-way merge use bounded heap size for space efficiency. Problems storing all seen elements like median use O(n). Choose pattern based on space constraints.',
    keyPoints: [
      'Full heap: O(n) space',
      'Top K: O(k) space (bounded)',
      'Two-heap median: O(n) (all elements)',
      'Operations: O(1) iterative, O(log n) recursive',
      'Choose based on whether storing all or subset',
    ],
  },
];
