/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the merge intervals template. What are the key steps?',
    sampleAnswer:
      'Merge intervals template has four steps. First, sort intervals by start time. Second, initialize result with first interval. Third, iterate from second interval: if current overlaps with last in result (current.start <= last.end), merge by updating last.end to max (last.end, current.end). If no overlap, append current to result. Fourth, return result. The key insight: after sorting, only need to check current against last merged interval, not all previous intervals. For example, [[1,3], [2,6], [8,10]]: start with [1,3]. [2,6] overlaps (2 <= 3), merge to [1,6]. [8,10] no overlap (8 > 6), append. Result: [[1,6], [8,10]]. This template is foundation for many interval problems.',
    keyPoints: [
      'Step 1: sort by start time',
      'Step 2: result starts with first interval',
      'Step 3: iterate, merge or append',
      'Merge: update last.end to max',
      'Only check against last merged',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the meeting rooms heap template. How do you track end times efficiently?',
    sampleAnswer:
      'Meeting rooms template uses min heap for end times. First, sort meetings by start time. Second, create min heap. Third, for each meeting: if heap not empty and heap top (earliest ending meeting) ends before or at current start, pop heap (room freed). Push current meeting end time to heap (allocate room). Fourth, heap size is rooms needed. Return max heap size or final heap size. The min heap automatically gives us earliest ending meeting at top in O(log n). For example, [[0,30], [5,10], [15,20]]: heap=[30], then [10,30], then [20,30] (10 popped), max size 2. The heap efficiently tracks which meeting ends next without scanning all active meetings.',
    keyPoints: [
      'Sort by start, min heap for end times',
      'For each: pop if top ends before current',
      'Push current end (allocate room)',
      'Heap size = active meetings',
      'Min heap: O(log n) to get earliest',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the overlap check helper function. Why is it useful to abstract this?',
    sampleAnswer:
      'Overlap check helper: overlaps (a, b) returns True if intervals overlap. Implementation: return not (a.end <= b.start or b.end <= a.start). Or alternatively: a.start < b.end and b.start < a.end. Abstracting to helper function: makes code cleaner, reusable across problems, easier to test, centralizes overlap logic. If overlap logic changes (inclusive vs exclusive ends), only update one place. Many interval problems need overlap check repeatedly: merge, intersection, conflict detection. For example, in My Calendar problem, check if new interval overlaps with any existing. Helper makes this simple: any (overlaps (new, existing) for existing in booked). Without helper, repeat complex condition everywhere, risking bugs. Clean code principle: abstract repeated logic.',
    keyPoints: [
      'Helper: overlaps (a, b) checks overlap',
      'Implementation: not (a.end <= b.start or b.end <= a.start)',
      'Benefits: cleaner, reusable, testable',
      'Centralize logic: change once, effect everywhere',
      'Many problems need repeated overlap checks',
    ],
  },
];
