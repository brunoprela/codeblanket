/**
 * Quiz questions for Interview Strategy section
 */

export const interviewstrategyQuiz = [
  {
    id: 'segment-interview-1',
    question: 'How do you recognize when a problem needs a Segment Tree?',
    hint: 'Look for patterns in the problem requirements.',
    sampleAnswer:
      'Key signals: "range queries" + "updates", operations like min/max/GCD that have no inverse, multiple queries on dynamic data, or explicit mention of intervals/segments. If the problem combines querying ranges and modifying data, and simpler approaches like prefix sums do not work, Segment Tree is likely needed.',
    keyPoints: [
      '"Range queries" + "updates"',
      'Min/max/GCD operations',
      'Dynamic data with multiple queries',
      'Intervals or segments mentioned',
    ],
  },
  {
    id: 'segment-interview-2',
    question:
      'What are the most common mistakes when implementing Segment Trees in interviews?',
    hint: 'Think about edge cases and implementation details.',
    sampleAnswer:
      'Common mistakes: 1) Off-by-one errors in range boundaries, 2) Forgetting to handle single-element ranges, 3) Wrong base case in recursion, 4) Not allocating enough space (use 4*N), 5) Incorrect merge logic for the specific operation. Always test with small examples and edge cases like N=1 or querying a single element.',
    keyPoints: [
      'Off-by-one errors in ranges',
      'Base case and single elements',
      'Space allocation (4*N)',
      'Correct merge operation',
    ],
  },
  {
    id: 'segment-interview-3',
    question:
      'Should you implement Segment Tree iteratively or recursively in an interview?',
    hint: 'Consider code clarity versus implementation speed.',
    sampleAnswer:
      'Use recursive implementation in interviews unless you are very comfortable with iterative. Recursive is more intuitive, easier to explain, and less error-prone. Iterative can be faster and saves stack space, but is harder to get right under pressure. Most interviewers prefer correct recursive code over buggy iterative code.',
    keyPoints: [
      'Recursive: more intuitive, easier to explain',
      'Iterative: faster, saves stack, harder to implement',
      'Choose based on confidence level',
    ],
  },
];
