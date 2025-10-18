/**
 * Quiz questions for Interval Operations section
 */

export const operationsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the merge operation for intervals. How do you determine the new merged interval?',
    sampleAnswer:
      'Merge combines overlapping intervals into one. After sorting, iterate through intervals. Track current merged interval. For each interval: if it overlaps with current (start <= current.end), extend current by taking max of end times. If no overlap (start > current.end), current is complete, add to result, start new current. At end, add final current. The key: merged interval start is min of all starts (first interval start after sorting), end is max of all ends (keep extending). For example, [1,3], [2,6], [8,10]: merge [1,3] and [2,6] into [1,6] (max of 3,6 is 6), then [8,10] separate. The operation squashes overlapping intervals into minimal set of non-overlapping intervals.',
    keyPoints: [
      'After sort, iterate tracking current merged',
      'Overlap: extend current.end to max(current.end, interval.end)',
      'No overlap: add current, start new',
      'Merged start: first start, end: max of all ends',
      'Output: minimal non-overlapping set',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the insert operation. How do you handle the three phases: before, overlapping, after?',
    sampleAnswer:
      'Insert adds new interval and merges overlaps. Three phases: Phase 1 (before): add all intervals that end before new starts (interval.end < new.start). Phase 2 (overlapping): merge all intervals that overlap with new. Track min start and max end of merged interval. Intervals overlap if interval.start <= new.end (since we are extending new). Phase 3 (after): add all remaining intervals that start after new ends. The beauty: we process intervals in one pass, identifying which phase based on start/end comparisons. For example, insert [4,8] into [[1,2], [3,5], [6,7], [9,10]]: before=[1,2], merge [3,5], [4,8], [6,7] into [3,8], after=[9,10]. Result: [[1,2], [3,8], [9,10]].',
    keyPoints: [
      'Phase 1: add intervals ending before new starts',
      'Phase 2: merge overlapping, track min start, max end',
      'Phase 3: add intervals starting after new ends',
      'One pass: identify phase by comparisons',
      'Overlap check: interval.start <= new.end',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the interval scheduling problem using min heap. Why use a heap?',
    sampleAnswer:
      'Meeting Rooms II: find minimum rooms needed for overlapping meetings. Sort meetings by start time. Use min heap to track end times of ongoing meetings. For each meeting: if heap not empty and earliest ending meeting (heap top) ends before current starts, that room is free (pop heap). Add current meeting end time to heap (allocate room). Heap size at any point is number of active meetings (rooms needed). Return max heap size seen. Why heap? It efficiently tracks which meeting ends earliest - the one we should check first for freeing up. Without heap, we would need to scan all active meetings O(n) each time. With heap, pop and push are O(log n). Total: O(n log n) for sort + O(n log n) for heap operations = O(n log n).',
    keyPoints: [
      'Sort by start, heap tracks end times',
      'Earliest ending meeting at heap top',
      'Pop if ends before current starts (room free)',
      'Push current end (allocate room)',
      'Heap size = active meetings = rooms needed',
      'Why heap: O(log n) vs O(n) to find earliest',
    ],
  },
];
