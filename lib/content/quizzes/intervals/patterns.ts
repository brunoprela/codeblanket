/**
 * Quiz questions for Common Interval Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the greedy interval selection pattern. Why sort by end time instead of start time?',
    sampleAnswer:
      'Interval selection maximizes non-overlapping intervals chosen. Classic example: activity selection, non-overlapping intervals. Greedy: sort by end time, iterate, select interval if it starts after last selected ends. Why end time? Finishing early leaves more room for future intervals. If we sort by start time, we might pick long interval that blocks many short ones. Proof: if optimal solution differs, we can replace its first interval with our choice (earliest ending) without making solution worse. For example, intervals [[1,5], [2,3], [4,6]]: sort by end gives [2,3], [4,6] (2 selected). Sort by start gives [1,5], [4,6] but [1,5] blocks [2,3]. Greedy by end time is provably optimal for maximizing count.',
    keyPoints: [
      'Goal: maximize non-overlapping intervals',
      'Greedy: sort by end, select if non-overlapping',
      'End time: finishing early leaves more room',
      'Start time: might pick long blocking interval',
      'Provably optimal for maximizing count',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe using event-based counting for interval overlap problems. How does it work?',
    sampleAnswer:
      'Event-based treats interval as two events: start (+1) and end (-1). Create list of all events with times and types. Sort events by time. Sweep through events, maintain active count. Start event increments count, end event decrements. Track maximum count seen - this is maximum overlapping intervals. For Meeting Rooms II: max count is minimum rooms needed. For example, meetings [[0,30], [5,10], [15,20]]: events [(0,+1), (5,+1), (10,-1), (15,+1), (20,-1), (30,-1)]. Sweep: 0→1, 5→2 (max), 10→1, 15→2 (max), 20→1, 30→0. Max is 2 rooms. This avoids heap, simpler to code, same O(n log n) complexity.',
    keyPoints: [
      'Two events per interval: start (+1), end (-1)',
      'Sort all events by time',
      'Sweep: maintain active count',
      'Track maximum count = max overlap',
      'Alternative to heap, same complexity',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through interval intersection. How do you find common parts of two sorted interval lists?',
    sampleAnswer:
      'Given two sorted interval lists, find all intersections. Use two pointers, one per list. At each step: check if current intervals from both lists overlap. If overlap, intersection is [max(start1, start2), min(end1, end2)] - add to result. Move pointer of interval that ends first (it cannot intersect with future intervals from other list). If no overlap, move pointer of interval that starts first. Continue until either pointer reaches end. For example, A=[[0,2],[5,10]], B=[[1,5],[8,12]]: compare [0,2] and [1,5], overlap [1,2]. Move A pointer (ends at 2). Compare [5,10] and [1,5], overlap [5,5]. Move B pointer (ends at 5). Compare [5,10] and [8,12], overlap [8,10]. Done. The two-pointer technique efficiently finds all intersections in O(n+m).',
    keyPoints: [
      'Two pointers, one per sorted list',
      'Check overlap: intersection is [max(starts), min(ends)]',
      'Move pointer of interval ending first',
      'Continue until either list exhausted',
      'O(n+m) using two pointers',
    ],
  },
];
