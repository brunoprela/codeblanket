/**
 * Quiz questions for Testing & Debugging section
 */

export const testingdebuggingQuiz = [
  {
    id: 'q1',
    question:
      'Why is logging better than print statements for debugging production code?',
    sampleAnswer:
      'Logging provides: 1) Configurable levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) to filter messages, 2) Timestamps and context automatically, 3) Can write to files, not just console, 4) Can be disabled in production without changing code, 5) Structured format for parsing/analysis. Print statements clutter code, are hard to remove, provide no context, and go to stdout only. In production, you want logging for monitoring and troubleshooting without affecting performance or adding noise.',
    keyPoints: [
      'Logging has configurable levels (DEBUG to CRITICAL)',
      'Automatic timestamps and formatting',
      'Can write to files and multiple outputs',
      'Easy to disable without code changes',
      'Print statements clutter and are hard to manage',
    ],
  },
  {
    id: 'q2',
    question:
      'What edge cases should you always test when writing a function that processes a list?',
    sampleAnswer:
      'Always test: 1) Empty list [], 2) Single element list [x], 3) Two elements (minimum for comparison), 4) All same elements [5,5,5], 5) Already sorted vs unsorted (if relevant), 6) Negative numbers, 7) Duplicates, 8) Large lists (performance), 9) None as input, 10) Wrong type (non-list). These edge cases catch off-by-one errors, null pointer issues, and ensure robust error handling.',
    keyPoints: [
      'Empty list - most common edge case',
      'Single element - boundary condition',
      'All same elements - degenerate case',
      'Negative numbers and duplicates',
      'Type errors and None handling',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the difference between assertions and exceptions for error handling.',
    sampleAnswer:
      'Assertions (assert) are for debugging and catching programmer errors during development. They check assumptions that should never fail if code is correct, and can be disabled with python -O. Use for: preconditions, postconditions, invariants. Exceptions (try/except, raise) are for handling runtime errors that might legitimately occur: user input errors, network failures, file not found. They stay enabled in production. Rule: Use assert for "this should never happen if my code is correct", use exceptions for "this might happen due to external factors". Never use assert for input validation in production!',
    keyPoints: [
      'assert: debug-time, for programmer errors',
      'Assertions can be disabled (-O flag)',
      'Exceptions: runtime, for expected failures',
      'Exceptions always active in production',
      'Never use assert for production input validation',
    ],
  },
];
