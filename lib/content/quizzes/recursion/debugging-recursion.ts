/**
 * Quiz questions for Debugging & Visualizing Recursion section
 */

export const debuggingrecursionQuiz = [
  {
    id: 'q-debug1',
    question: 'What are the best techniques for debugging recursive functions?',
    sampleAnswer:
      "Key techniques: 1) **Add print statements** showing parameters, return values, and recursion depth at entry/exit, 2) **Start with small inputs** (n=0,1,2) to trace by hand, 3) **Use a debugger** with breakpoints to step through calls and inspect call stack, 4) **Draw recursion tree** to visualize call structure and returned values, 5) **Verify base cases** are correct and reachable. The print statement technique is most common for quickly understanding what's happening.",
    keyPoints: [
      'Print statements with parameters and depth',
      'Test with small inputs first',
      'Use debugger to step through calls',
      'Draw recursion tree on paper',
      'Verify base cases carefully',
      'Check that recursion progresses toward base case',
    ],
  },
  {
    id: 'q-debug2',
    question: 'How can you visualize and understand a recursive call stack?',
    sampleAnswer:
      'Methods to visualize: 1) **Draw recursion tree** - each node is a function call, show parameters and return values, 2) **Add indentation** to print statements based on recursion depth to show call hierarchy, 3) **Use Python Tutor** (pythontutor.com) to animate execution, 4) **Debugger call stack view** shows active function calls, 5) **Manual trace table** with columns for each call level. The recursion tree is most intuitive for understanding flow and overlapping subproblems.',
    keyPoints: [
      'Recursion tree diagram with parameters',
      'Indented print statements show depth',
      'Python Tutor for animation',
      'Debugger shows call stack',
      'Manual trace table',
      'Helps identify base case and overlapping work',
    ],
  },
  {
    id: 'q-debug3',
    question: 'What are common mistakes when writing base cases in recursion?',
    sampleAnswer:
      'Common mistakes: 1) **Missing edge cases** (empty input, n=0, null), 2) **Base case never reached** (wrong condition or not progressing toward it), 3) **Multiple base cases needed but only one handled**, 4) **Wrong return value** (e.g., returning None instead of [], 5) **Checking condition after recursive call** instead of before. Always test base cases separately and ensure every recursive path eventually hits one.',
    keyPoints: [
      'Missing edge cases (empty, zero, null)',
      'Base case unreachable - wrong condition',
      'Forgetting multiple base cases',
      'Incorrect return value for base case',
      'Checking condition too late',
      'Test base cases independently first',
    ],
  },
];
