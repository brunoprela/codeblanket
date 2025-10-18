/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through your complete DP interview approach from problem statement to coded solution.',
    sampleAnswer:
      'First, recognize its DP: keywords like max/min/count, or recursive but slow. Second, clarify: constraints, edge cases, input size. Third, explain approach using 5 steps: state definition (what does dp[i] mean), recurrence (how to compute dp[i]), base cases, computation order, where answer is. Fourth, state complexity: number of states gives space, states times transition gives time. Fifth, draw small example: show first few values being computed. Sixth, code bottom-up solution with clear variable names and comments. Seventh, test with example: trace through computation. Eighth, discuss optimization: can we reduce space? Finally, mention related problems or variations. This systematic approach shows deep understanding and clear communication.',
    keyPoints: [
      'Recognize DP keywords and patterns',
      'Clarify constraints and edge cases',
      'Explain using 5-step framework',
      'State complexity with reasoning',
      'Draw example, trace computation',
      'Code clearly with comments, test, optimize',
    ],
  },
  {
    id: 'q2',
    question:
      'How do you debug DP solutions when they give wrong answers? What is your systematic debugging process?',
    sampleAnswer:
      'First, test with smallest possible input: n=0, n=1, n=2. Does base case work? Second, manually compute dp table for small example and compare with code output. Third, add print statements showing dp values at each step. Fourth, check recurrence: is it mathematically correct? Draw decision tree. Fifth, verify computation order: are we accessing values before computing them? Sixth, check bounds: off-by-one errors in loops or array indices? Seventh, verify initialization: are default values correct? Eighth, for 2D, print entire table to visualize. Common bugs: wrong base case, wrong recurrence, wrong loop bounds, wrong initialization. Systematic approach: start small, verify manually, compare with code, identify first divergence.',
    keyPoints: [
      'Test smallest inputs: n=0, 1, 2',
      'Manually compute and compare with code',
      'Print dp values at each step',
      'Verify recurrence mathematically',
      'Check: computation order, bounds, initialization',
      'Find first divergence between manual and code',
    ],
  },
  {
    id: 'q3',
    question:
      'What do you do when you cannot figure out the DP state or recurrence? What is your fallback strategy?',
    sampleAnswer:
      'First, solve smaller version manually (n=3, n=4) and look for pattern. How does answer for n=4 relate to n=3, n=2? Second, try writing brute force recursive solution. What parameters does recursion need? Those become DP state. What subproblems does it call? That hints at recurrence. Third, look for similar problems: is it like Fibonacci, Knapsack, LCS? Fourth, think about choices: at each step, what decisions can I make? Include/exclude, take/skip, etc. Fifth, define state as "answer for subproblem ending at i" or "answer using first i elements". Sixth, if stuck, mention what you know and ask for hint. Showing thought process is valuable even without complete solution.',
    keyPoints: [
      'Solve small manually, find pattern',
      'Write brute force, parameters → state',
      'Compare to known patterns',
      'Think about choices at each step',
      'State: answer for subproblem at/using i',
      'Show thought process, ask for hints',
    ],
  },
];
