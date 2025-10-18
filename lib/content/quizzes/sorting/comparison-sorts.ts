/**
 * Quiz questions for Comparison-Based Sorting Algorithms section
 */

export const comparisonsortsQuiz = [
  {
    id: 'q1',
    question:
      'Compare merge sort and quick sort. When would you choose one over the other?',
    hint: 'Think about stability, space, and performance guarantees.',
    sampleAnswer:
      'Both are O(n log n) on average, but they have different tradeoffs. Merge sort is stable and has guaranteed O(n log n) worst-case performance, but requires O(n) extra space. Quick sort is typically faster in practice due to better cache performance and lower constants, is in-place (O(log n) space for recursion), but has O(n²) worst case and is not stable. I would choose merge sort when: 1) Stability is required, 2) I need guaranteed O(n log n) performance, 3) Memory is not a concern, or 4) Sorting linked lists. I would choose quick sort when: 1) Average performance matters more than worst-case, 2) Memory is limited, 3) Stability is not needed. In practice, quicksort with randomized pivots is usually the go-to for general purpose sorting.',
    keyPoints: [
      'Merge sort: stable, O(n log n) guaranteed, O(n) space',
      'Quick sort: faster in practice, in-place, O(n²) worst case, unstable',
      'Choose merge for stability and guaranteed performance',
      'Choose quick for speed and memory efficiency',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why the worst case of quicksort is O(n²). How can you avoid this?',
    sampleAnswer:
      'Quicksort degrades to O(n²) when the pivot choices are consistently bad - when the pivot is always the smallest or largest element. This creates unbalanced partitions where one side has n-1 elements and the other has 0. You end up with n levels of recursion instead of log n, and each level does O(n) work, giving O(n²) total. This happens when sorting already-sorted data with a naive pivot selection like choosing the first or last element. You can avoid this by: 1) Using randomized pivot selection, 2) Median-of-three pivot selection (choose median of first, middle, last), or 3) Using three-way partitioning for arrays with many duplicates. Randomization essentially guarantees O(n log n) average performance regardless of input.',
    keyPoints: [
      'Worst case when pivot always creates unbalanced partitions',
      'Happens on sorted data with poor pivot selection',
      'Results in n levels × O(n) work = O(n²)',
      'Avoid with: randomized pivots, median-of-three, three-way partitioning',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is insertion sort O(n) on already-sorted data but O(n²) on random data?',
    sampleAnswer:
      'Insertion sort is adaptive - its performance depends on how sorted the input already is. On already-sorted data, each element is already in the correct position. The inner while loop never executes because arr[j] is never greater than key. So we just scan through the array once, doing O(1) work per element, giving O(n) total. On random or reverse-sorted data, each element might need to be compared with and moved past many elements to find its correct position. On average, each element moves halfway back, which is O(n) comparisons and shifts for each of n elements, giving O(n²). This adaptive property makes insertion sort excellent for nearly-sorted data or online sorting where elements arrive one at a time.',
    keyPoints: [
      'Adaptive: performance depends on initial order',
      'Already sorted: inner loop never runs → O(n)',
      'Random data: each element shifts ~n/2 positions → O(n²)',
      'Makes it great for nearly-sorted data and online sorting',
    ],
  },
];
