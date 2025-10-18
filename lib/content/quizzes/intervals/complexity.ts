/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare time complexity of different interval approaches: sorting vs heap vs events. When is each best?',
    sampleAnswer:
      'Sorting approach: O(n log n) sort + O(n) iterate = O(n log n) total. Works for merge, basic overlap detection. Heap approach: O(n log n) sort + O(n log n) heap operations = O(n log n) total. Best for tracking dynamic set like meeting rooms (need earliest ending). Event-based: O(n) create events + O(n log n) sort + O(n) sweep = O(n log n) total. Simpler than heap for max overlap. All are O(n log n) asymptotically, but constants differ. Sorting is simplest, use when one pass after sort suffices. Heap when need priority queue (earliest/latest). Events when counting overlaps. Space: sorting O(1) extra, heap O(n), events O(n). Choose based on problem needs, not just complexity.',
    keyPoints: [
      'All common approaches: O(n log n) time',
      'Sorting: simplest, O(1) space',
      'Heap: O(n) space, dynamic priority',
      'Events: O(n) space, counting overlaps',
      'Choose by problem needs, not just complexity',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain when interval problems require O(n²) time. What signals this?',
    sampleAnswer:
      'Some interval problems cannot avoid O(n²). Signal: need to compare every pair or track interactions between all intervals. Examples: find all pairs of overlapping intervals (must compare all pairs), interval queries without preprocessing, problems where order matters for all pairs. However, many apparent O(n²) problems reduce to O(n log n) with sorting. Rule: if can sort and process in order, likely O(n log n). If need all pairwise comparisons with no structure to exploit, likely O(n²). For example, "which intervals overlap with interval i?" is O(n) per query without preprocessing, O(n²) for all queries. With sorting and binary search, O(n log n) preprocess + O(log n) per query. Always try sorting first - it converts many O(n²) to O(n log n).',
    keyPoints: [
      'O(n²) when: all pairs, no exploitable structure',
      'Examples: all overlapping pairs, no sorting helps',
      'Many apparent O(n²) reduce to O(n log n)',
      'Sorting converts many O(n²) to O(n log n)',
      'Try sorting first as optimization',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe space complexity tradeoffs in interval problems. When do you need O(n) vs O(1) extra space?',
    sampleAnswer:
      'O(1) extra space: in-place merge after sorting, reusing input array. Only need a few variables to track current interval. Possible when output size is at most input size. O(n) extra space: heap for meeting rooms (stores end times), event list (2n events), result array when output size unbounded. Cannot avoid when: need data structure (heap, set), result is separate from input, cannot modify input. For example, merge intervals can be O(1) if allowed to modify input and output fits in input. Meeting rooms II needs O(n) heap. Choose O(1) when possible (cleaner, less memory), but O(n) is fine for most interview problems. Clarify if in-place required. Often the clarity of O(n) space outweighs O(1) complexity.',
    keyPoints: [
      'O(1): in-place merge, reuse input, few variables',
      'O(n): heap, event list, unbounded output',
      'O(n) needed: data structures, separate result',
      'In-place often more complex to implement',
      'Clarity vs space: O(n) often acceptable',
    ],
  },
];
