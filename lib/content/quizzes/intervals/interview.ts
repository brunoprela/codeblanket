/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize an interval problem? What keywords or patterns signal this?',
    sampleAnswer:
      'Keywords: "intervals", "ranges", "meetings", "events", "schedule", "overlapping", "merge", "insert". Problem types: merge overlapping ranges, schedule meetings, find free time, count conflicts, maximize selected intervals. Data format: array of [start, end] pairs. If you see pairs representing ranges and need to process relationships between them, likely interval problem. For example, "given meeting times, find minimum conference rooms" is intervals. "Merge overlapping time ranges" is intervals. Pattern: data is ranges, need to handle overlaps or sequential processing. Key question: does problem involve ranges with start/end that might overlap or need ordering? If yes, probably interval problem with sorting as first step.',
    keyPoints: [
      'Keywords: intervals, ranges, meetings, schedule, overlap',
      'Data: array of [start, end] pairs',
      'Types: merge, schedule, conflicts, maximize',
      'Pattern: ranges with potential overlaps',
      'First step usually: sort by start time',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your interval interview approach from recognition to implementation.',
    sampleAnswer:
      'First, recognize its interval problem from keywords. Second, clarify: are intervals closed/open, can they be negative, are they sorted, what to return. Third, explain approach: sort by start time (or end time for greedy), then iterate processing overlaps/merges. Fourth, state complexity: O(n log n) for sort dominates, O(n) for processing. Fifth, discuss implementation: use helper function for overlap check, consider heap vs events for counting. Sixth, draw example: show a few intervals, demonstrate sorting, show merging/processing. Seventh, code clearly with sort first, iterate with clear overlap logic. Test with edge cases: empty, single interval, all overlap, none overlap. Finally, discuss optimization: can we avoid sort? Is O(1) space possible? This systematic approach demonstrates depth.',
    keyPoints: [
      'Recognize keywords, clarify requirements',
      'Explain: sort by start, process overlaps',
      'Complexity: O(n log n) sort dominates',
      'Draw example, show sorting and processing',
      'Code: sort, iterate, overlap logic',
      'Test edges, discuss optimizations',
    ],
  },
  {
    id: 'q3',
    question: 'What are common interval mistakes and how do you avoid them?',
    sampleAnswer:
      'First: forgetting to sort - leads to missing overlaps. Second: wrong overlap condition - off-by-one with <= vs <. Third: not handling empty input. Fourth: modifying last interval incorrectly during merge (updating wrong end). Fifth: for heap solution, forgetting to pop freed rooms. Sixth: sorting by wrong field (start vs end). Seventh: inclusive vs exclusive interval ends not clarified. My strategy: always sort first, use overlap helper function, test with: empty, single, all overlap, none overlap, adjacent intervals ([1,2], [2,3]). Draw diagram to verify overlap logic. For merge, carefully track which interval is current. Heap problems: clearly understand what heap stores (end times). Most bugs come from overlap logic - nail that first.',
    keyPoints: [
      'Forget to sort → missing overlaps',
      'Wrong overlap condition (off-by-one)',
      'Modify wrong interval during merge',
      'Test: empty, single, all/none overlap, adjacent',
      'Use overlap helper, draw diagram',
    ],
  },
];
