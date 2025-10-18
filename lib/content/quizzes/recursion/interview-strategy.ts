/**
 * Quiz questions for Recursion in Interviews section
 */

export const interviewstrategyQuiz = [
  {
    id: 'q-interview1',
    question:
      'What is the framework for solving recursive problems in interviews?',
    sampleAnswer:
      'The 4-step framework: 1) **Define the subproblem** - what does the function solve for smaller input?, 2) **Find the base case** - simplest input that can be solved directly, 3) **Write the recursive case** - express solution in terms of subproblem(s), 4) **Analyze complexity** - time and space, consider memoization. Always start by clearly stating: "This function takes X and returns Y by recursing on...". Communicate your thinking process clearly.',
    keyPoints: [
      'Define subproblem: what function does',
      'Identify base case(s): simplest inputs',
      'Express recursive case: solution in terms of subproblems',
      'Analyze complexity: time/space',
      'Consider memoization for optimization',
      'Communicate thinking out loud',
    ],
  },
  {
    id: 'q-interview2',
    question:
      'How do you handle the interviewer asking "can you do this without recursion?"',
    sampleAnswer:
      'Strategy: 1) **Acknowledge tradeoff**: "Yes, I can convert to iteration. Recursion uses O(n) stack space, iteration would be O(1).", 2) **Explain approach**: "I\'d use an explicit stack to simulate the call stack", 3) **Ask clarification**: "Would you like me to implement the iterative version, or is understanding the approach sufficient?", 4) **If implementing**: convert methodically, 5) **Highlight when recursion is better**: "For tree traversal, recursion is cleaner and stack depth is usually manageable."',
    keyPoints: [
      'Acknowledge pros/cons of each approach',
      'Explain iterative conversion strategy',
      'Ask what level of detail they want',
      'Show you understand both paradigms',
      "Defend recursion when it's better",
      'Mention stack space as key difference',
    ],
  },
  {
    id: 'q-interview3',
    question:
      'What should you discuss about complexity for recursive solutions?',
    sampleAnswer:
      'Discuss: 1) **Time complexity**: Count recursive calls and work per call. Use recursion tree or recurrence relation. Mention if exponential without memoization, 2) **Space complexity**: Call stack depth + auxiliary space. Mention O(n) for call stack if depth is n, 3) **Optimization**: If exponential, propose memoization and analyze improved complexity, 4) **Comparison**: "Without memo: O(2^n), with memo: O(n)", 5) **Trade-off**: Acknowledge space cost of memoization. Show you can analyze recursion rigorously.',
    keyPoints: [
      'Time: recursive calls × work per call',
      'Space: call stack depth + auxiliary',
      'Mention if exponential (O(2^n))',
      'Propose memoization if needed',
      'Compare before/after optimization',
      'Show understanding of trade-offs',
    ],
  },
];
