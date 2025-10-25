/**
 * Quiz questions for Why Sorting Matters section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what we mean by a "stable" sorting algorithm. Why would stability matter in practice?',
    hint: 'Think about what happens to equal elements.',
    sampleAnswer:
      'A stable sorting algorithm preserves the relative order of equal elements. If you have two items with the same value, and item A came before item B in the original array, then after a stable sort, A will still come before B. This matters when you are sorting by one field but want to preserve the order of another field. For example, if you have students sorted by name and you want to sort them by grade, a stable sort will keep students with the same grade in alphabetical order. Merge sort and insertion sort are stable, but quicksort and heapsort are not. In production, stability often matters for user-facing features where order needs to be predictable.',
    keyPoints: [
      'Stable: equal elements maintain relative order',
      'Important for multi-level sorting',
      'Example: sort by grade, preserve alphabetical order within same grade',
      'Merge sort, insertion sort are stable; quicksort, heapsort are not',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through the difference between in-place and not in-place sorting. What is the tradeoff?',
    sampleAnswer:
      'In-place sorting means the algorithm sorts the array using only O(1) extra space - it rearranges elements within the original array without creating a copy. Not in-place sorting uses O(n) additional space, typically by creating temporary arrays. The tradeoff is space versus implementation simplicity. Quicksort is in-place - it sorts by swapping elements within the array, using minimal extra memory. Merge sort is not in-place - it creates temporary arrays during the merge step, requiring O(n) space. In-place is better when memory is limited, but not-in-place algorithms like merge sort can be easier to implement correctly and are stable. For large datasets, the O(n) space overhead of merge sort can be significant.',
    keyPoints: [
      'In-place: O(1) extra space, sorts within original array',
      'Not in-place: O(n) extra space, creates temporary copies',
      'In-place saves memory but can be more complex',
      'Examples: Quicksort (in-place), Merge sort (not in-place)',
    ],
  },
  {
    id: 'q3',
    question:
      'Why would you ever use a simple O(n²) sorting algorithm like insertion sort when O(n log n) algorithms exist?',
    sampleAnswer:
      "There are actually several good reasons. First, for small arrays (say under 20 elements), insertion sort can be faster than quicksort or mergesort because it has very low overhead - no recursive calls, no complex partitioning. Second, insertion sort is adaptive - it runs in O(n) time on already-sorted or nearly-sorted data. If you know your data is mostly sorted, insertion sort is excellent. Third, it is stable and in-place, which matters for certain use cases. Fourth, it is extremely simple to implement correctly. In fact, many production implementations of quicksort switch to insertion sort for small subarrays. Python\'s Timsort uses insertion sort as one of its building blocks.",
    keyPoints: [
      'Faster for small arrays due to low overhead',
      'Adaptive: O(n) on nearly-sorted data',
      'Stable and in-place simultaneously',
      'Simple to implement correctly',
      'Used in hybrid algorithms like Timsort',
    ],
  },
];
