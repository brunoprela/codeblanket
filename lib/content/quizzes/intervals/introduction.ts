/**
 * Quiz questions for Introduction to Intervals section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what intervals are and why sorting is almost always the first step. What does sorting enable?',
    sampleAnswer:
      'Intervals represent ranges with start and end points, like [1,3] means 1 to 3. Sorting by start time is usually first step because it establishes order, making overlaps detectable. After sorting, if current interval starts before previous ends, they overlap. Without sorting, you would need to compare every pair - O(n²). With sorting, one pass detects all overlaps - O(n). Sorting also enables greedy algorithms: process intervals in order, make local decisions. For example, merge intervals: after sorting, only need to check current vs last merged. Meeting rooms: after sorting, compare consecutive intervals for conflicts. The pattern: sort enables linear-time algorithms instead of quadratic brute force. Sorting cost O(n log n) is worth it for O(n) processing.',
    keyPoints: [
      'Intervals: ranges with start and end',
      'Sort by start time establishes order',
      'Enables O(n) overlap detection vs O(n²)',
      'Greedy algorithms process in order',
      'Sort O(n log n) worth it for O(n) processing',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe how to detect if two intervals overlap. What is the overlap condition?',
    sampleAnswer:
      'Two intervals [a_start, a_end] and [b_start, b_end] overlap if they share any points. The overlap condition: a_start < b_end AND b_start < a_end. Or equivalently, NOT (a_end <= b_start OR b_end <= a_start). Intuition: they do NOT overlap if a completely before b (a_end <= b_start) or b completely before a (b_end <= a_start). Otherwise they overlap. For example, [1,3] and [2,5] overlap because 1 < 5 AND 2 < 3. [1,2] and [3,4] do not overlap because 2 <= 3 (first ends before second starts). This condition works regardless of which interval starts first. After sorting, can simplify: if current.start < previous.end, they overlap. The overlap check is fundamental to all interval problems.',
    keyPoints: [
      'Overlap: a_start < b_end AND b_start < a_end',
      'Or: NOT (a before b OR b before a)',
      'After sorting: current.start < previous.end',
      'Works regardless of which starts first',
      'Fundamental to all interval problems',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through common interval problems. What patterns do you see?',
    sampleAnswer:
      'Common problems: Merge Intervals (combine overlapping), Meeting Rooms I (can attend all?), Meeting Rooms II (min rooms needed), Insert Interval (add and merge), Interval List Intersections (find common parts). Patterns: Most start with sorting by start time. Merge pattern: iterate, track current merged interval, extend or add new. Counting pattern: track active intervals using start/end events or heap. Greedy pattern: sort by end time, select maximum non-overlapping. For example, Merge Intervals sorts then iterates comparing current with last merged. Meeting Rooms II uses min heap to track end times of ongoing meetings. Insert Interval handles three phases: before, overlapping, after. Recognize which pattern by what problem asks: merge, count overlaps, select maximum, find intersections.',
    keyPoints: [
      'Common: merge, meeting rooms, insert, intersections',
      'Pattern 1: merge by extending or adding',
      'Pattern 2: count active with events or heap',
      'Pattern 3: greedy with end-time sorting',
      'Recognize pattern from problem requirement',
    ],
  },
];
