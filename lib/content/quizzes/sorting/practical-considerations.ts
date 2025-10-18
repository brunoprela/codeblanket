/**
 * Quiz questions for Practical Sorting Strategies section
 */

export const practicalconsiderationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain what makes Timsort adaptive and why this is useful for real-world data.',
    sampleAnswer:
      'Timsort is adaptive because it detects and exploits existing order in the data. Instead of treating all data the same, it looks for "runs" - sequences that are already sorted (either ascending or descending). When it finds these runs, it takes advantage of them rather than breaking them apart and resorting from scratch. For already-sorted data, Timsort runs in O(n) time, just verifying the order. For partially-sorted data, it does less work than algorithms like quicksort which don\'t recognize the existing order. This is incredibly useful in practice because real-world data is often partially sorted - think of adding new records to a database, logging events with timestamps, or updating ranked lists. Timsort handles these cases very efficiently.',
    keyPoints: [
      'Adaptive: performance improves with existing order',
      'Detects and preserves sorted runs',
      'O(n) for already-sorted data',
      'Real-world data often has patterns and partial order',
      'Much faster than non-adaptive algorithms on real data',
    ],
  },
  {
    id: 'q2',
    question:
      'You need to find the top 10 elements from a million-element array. Would you sort the entire array? Why or why not?',
    hint: 'Think about what complexity you actually need.',
    sampleAnswer:
      'No, I would not sort the entire array. Sorting takes O(n log n) time, which for a million elements is about 20 million operations. Instead, I would use a min-heap of size 10. I iterate through the array once, and for each element, if it is larger than the smallest element in my heap (the heap root), I remove the root and add the new element. This takes O(n log k) time where k is 10, so about 1 million × 3 = 3 million operations. That is almost 7 times faster. The key insight is that I do not need a fully sorted array - I just need the top k elements. Using a heap for partial sorting is way more efficient than full sorting. In Python, heapq.nlargest() does exactly this.',
    keyPoints: [
      'Full sort: O(n log n) - wasteful for top-k problem',
      'Min-heap approach: O(n log k) where k=10',
      'For k << n, heap is much faster',
      "Don't do more work than necessary",
      'Use heapq.nlargest() or heapq.nsmallest()',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through when you would choose an unstable sort over a stable sort, given that stability seems strictly better.',
    sampleAnswer:
      'Stability is not always free - it can come with performance or implementation complexity costs. I would choose an unstable sort when: 1) Stability does not matter for my use case - if I am sorting primitive values like integers where there is no "secondary" ordering to preserve. 2) Performance is critical and the unstable sort is faster - quicksort is generally faster than merge sort due to better cache locality and lower constants, even though it is unstable. 3) Memory is constrained - heapsort is O(1) space and O(n log n) guaranteed time, which is hard to beat if stability is not needed. In practice, if you are sorting simple values and stability is not a requirement, using the faster unstable sort is the right engineering decision. The key is knowing what you need.',
    keyPoints: [
      'Stability can have performance/space costs',
      'Unstable OK when: sorting primitives, no secondary ordering matters',
      'Quicksort often faster than merge sort',
      'Heapsort: in-place + O(n log n) guarantee',
      'Choose based on requirements, not "stability is always better"',
    ],
  },
];
