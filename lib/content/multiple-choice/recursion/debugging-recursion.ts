/**
 * Multiple choice questions for Debugging & Visualizing Recursion section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const debuggingrecursionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-debug1',
    question:
      'What is the best first step when debugging a recursive function?',
    options: [
      'Rewrite it iteratively',
      'Test with small inputs (n=0, 1, 2)',
      'Add complex logging',
      'Optimize for performance',
    ],
    correctAnswer: 1,
    explanation:
      'Testing with small inputs lets you trace execution by hand and verify logic before dealing with complex cases. Start simple!',
  },
  {
    id: 'mc-debug2',
    question:
      'What visualization tool is most helpful for understanding recursion?',
    options: ['Flowchart', 'UML diagram', 'Recursion tree', 'State machine'],
    correctAnswer: 2,
    explanation:
      'A recursion tree shows each function call as a node with its parameters and return value, making it easy to trace execution flow and identify overlapping subproblems.',
  },
  {
    id: 'mc-debug3',
    question:
      'When adding print statements to debug recursion, what should you include?',
    options: [
      'Only the final result',
      'Just the parameters',
      'Parameters, depth, and return values',
      'Only error messages',
    ],
    correctAnswer: 2,
    explanation:
      "Include parameters (what's being processed), recursion depth (how deep), and return values (what each call produces) to fully understand execution flow.",
  },
  {
    id: 'mc-debug4',
    question: 'What indicates that a base case might be unreachable?',
    options: [
      'Function returns too quickly',
      'Stack overflow or infinite recursion',
      'Wrong output value',
      'Slow performance',
    ],
    correctAnswer: 1,
    explanation:
      "If base case is never reached, recursion continues indefinitely causing stack overflow. This means the recursive calls don't progress toward the base case condition.",
  },
  {
    id: 'mc-debug5',
    question: 'Why is it helpful to trace recursion with small inputs first?',
    options: [
      "It\'s faster to execute",
      'You can verify logic by hand before complex cases',
      'It uses less memory',
      "It\'s required by Python",
    ],
    correctAnswer: 1,
    explanation:
      'Small inputs (n=0,1,2) let you manually trace each step, verify base cases work, and understand the recursive pattern before tackling larger inputs.',
  },
];
