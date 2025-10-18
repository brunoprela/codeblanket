/**
 * Quiz questions for Algorithm Comparison & Selection Guide section
 */

export const comparisonguideQuiz = [
  {
    id: 'q1',
    question:
      'Why would you choose merge sort over quick sort even though quick sort is typically faster in practice?',
    hint: 'Think about worst-case guarantees and stability.',
    sampleAnswer:
      'I would choose merge sort when I need guaranteed O(n log n) worst-case performance or when stability is required. Quick sort can degrade to O(n²) on certain inputs (like sorted arrays with bad pivot selection), while merge sort always performs at O(n log n). Merge sort is also stable, meaning it preserves the relative order of equal elements, which matters when sorting by multiple criteria. For example, if I am implementing a critical system where performance unpredictability could cause issues, or if I am sorting already sorted data by a secondary key and need to preserve the primary sort order, merge sort is the better choice despite being slightly slower on average.',
    keyPoints: [
      'Merge sort guarantees O(n log n) worst case',
      'Quick sort can degrade to O(n²)',
      'Merge sort is stable, quick sort is not',
      'Choose merge for predictability and stability',
      'Trade slightly slower average for guaranteed performance',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through when you would use counting sort instead of a comparison-based sort like quick sort.',
    hint: 'Consider the nature and range of the data.',
    sampleAnswer:
      'I would use counting sort when the data consists of integers within a known, relatively small range. Counting sort is O(n + k) where k is the range, which can be faster than O(n log n) comparison sorts if k is not too large. For example, if I am sorting ages (0-150), test scores (0-100), or ratings (1-5), counting sort is perfect. It is linear time and stable. However, I would not use it for arbitrary integers, floating-point numbers, or when the range is huge (like sorting all 32-bit integers), because the O(k) space and time for the count array would be prohibitive. The key decision factor is: can I afford O(k) extra space, and is k reasonable relative to n?',
    keyPoints: [
      'Use counting sort for integers in small range',
      'O(n + k) can beat O(n log n) when k is reasonable',
      'Example: ages, scores, ratings',
      'Not suitable for large ranges or non-integers',
      'Requires O(k) extra space',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain why you should avoid sorting if you only need to find the k largest elements. What algorithm would you use instead?',
    hint: 'Think about partial vs complete ordering.',
    sampleAnswer:
      'Full sorting is overkill if you only need k elements because you are doing more work than necessary. Sorting takes O(n log n), but you do not need everything in order - just the top k. The better approach is to use a min-heap of size k. As you iterate through the array, you maintain a heap of the k largest elements seen so far. This takes O(n log k) time, which is much better than O(n log n) when k is small. For example, if n is 1 million and k is 10, that is the difference between log(10) ≈ 3 and log(1000000) ≈ 20 per element. Alternatively, QuickSelect can find the kth largest in O(n) average time. The key insight: partial ordering problems should not use full sorting algorithms.',
    keyPoints: [
      'Sorting everything is overkill for top-k',
      'Min-heap approach: O(n log k) vs O(n log n)',
      'Much faster when k << n',
      'QuickSelect: O(n) average for kth element',
      'Partial ordering problems need partial solutions',
    ],
  },
];
