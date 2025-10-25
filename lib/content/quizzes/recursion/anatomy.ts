/**
 * Quiz questions for Anatomy of a Recursive Function section
 */

export const anatomyQuiz = [
  {
    id: 'q1',
    question:
      'Why is it important that the recursive case makes progress toward the base case?',
    sampleAnswer:
      "The recursive case must make progress toward the base case to ensure the recursion eventually terminates. If we don't move closer to the base case with each call, we'll have infinite recursion leading to a stack overflow. For example, calling factorial (n) instead of factorial (n-1) would never reach n <= 1. Progress typically means: decreasing a number, reducing array size, moving through a data structure, or simplifying a problem in some measurable way.",
    keyPoints: [
      'Ensures recursion terminates',
      'Prevents infinite recursion',
      'Prevents stack overflow',
      'Each call must be "simpler" than previous',
      'Progress can be: smaller n, smaller array, closer to target',
    ],
  },
  {
    id: 'q2',
    question:
      'Trace through the execution of factorial(4) step by step, showing both the "winding" (function calls) and "unwinding" (returns) phases.',
    sampleAnswer:
      'Winding phase (calls going down): factorial(4) calls factorial(3); factorial(3) calls factorial(2); factorial(2) calls factorial(1); factorial(1) returns 1 (base case). Unwinding phase (returns coming back): factorial(1) = 1; factorial(2) = 2 * 1 = 2; factorial(3) = 3 * 2 = 6; factorial(4) = 4 * 6 = 24. The key insight: we build up a chain of pending multiplications during winding, then compute them during unwinding. The call stack holds 4 frames at maximum depth, then shrinks as each function returns.',
    keyPoints: [
      'Winding: calls stack up until base case',
      'Unwinding: returns compute back up the stack',
      'Base case triggers the unwinding',
      'Maximum stack depth = recursion depth (4 frames)',
      'Each return combines current value with recursive result',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the "leap of faith" principle in recursive thinking and why it\'s important.',
    sampleAnswer:
      "The \"leap of faith\" means trusting that your recursive call correctly solves the smaller subproblem, without mentally tracing through all the calls. For factorial (n), assume factorial (n-1) gives you the right answer, then just multiply by n. Don't try to trace factorial (n-1) down to factorial(1) in your head - that's the computer's job. This is important because: (1) It simplifies your thinking - focus on one level at a time, (2) It makes recursion tractable for complex problems like trees where tracing would be overwhelming, (3) It matches the mathematical induction principle: prove the base case works, assume the recursive case works for n-1, prove it works for n. Without this mindset, recursion feels impossible.",
    keyPoints: [
      'Trust recursive call solves the smaller problem',
      "Don't mentally trace all calls",
      'Focus on one level: base case + one recursive step',
      'Mirrors mathematical induction',
      'Makes complex recursion tractable',
    ],
  },
];
