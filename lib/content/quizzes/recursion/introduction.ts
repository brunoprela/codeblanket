/**
 * Quiz questions for Introduction to Recursion section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'What are the two essential components every recursive function must have?',
    sampleAnswer:
      'Every recursive function needs: (1) Base Case - a condition that stops the recursion and returns a value directly without further calls. This prevents infinite recursion. (2) Recursive Case - the part where the function calls itself with modified arguments that move towards the base case. For example, in factorial(n), the base case is n <= 1 returning 1, and the recursive case is n * factorial(n-1).',
    keyPoints: [
      'Base case: stops recursion',
      'Recursive case: calls itself with simpler input',
      'Arguments must progress toward base case',
      'Base case prevents infinite recursion',
      'Both components are mandatory',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain what happens on the call stack when a recursive function executes.',
    sampleAnswer:
      "When a recursive function calls itself, each call creates a new stack frame on the call stack. The stack frame stores the function's local variables and return address. These frames stack up until the base case is reached. Then, frames are popped off the stack in LIFO order as each call returns. For factorial(3): factorial(3) calls factorial(2), which calls factorial(1). When factorial(1) returns 1, then factorial(2) returns 2*1=2, then factorial(3) returns 3*2=6. The maximum stack depth equals the maximum recursion depth.",
    keyPoints: [
      'Each call creates a stack frame',
      'Frames stack up until base case',
      'Frames pop off in LIFO order',
      'Stack depth = recursion depth',
      'Risk of stack overflow if too deep',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is recursion considered "elegant" for certain problems, and what types of problems are naturally recursive?',
    sampleAnswer:
      'Recursion is elegant because it mirrors the mathematical definition of problems and often leads to shorter, more readable code compared to iterative solutions. Naturally recursive problems include: (1) Tree/Graph traversal - nodes have child nodes (same structure), (2) Divide-and-conquer algorithms - break problem into smaller subproblems of same type (merge sort, quicksort), (3) Problems with recursive definitions - factorial n! = n × (n-1)!, Fibonacci, (4) Backtracking - explore all paths/combinations (permutations, N-queens), (5) Nested structures - JSON parsing, file systems. The elegance comes from the direct translation: the code structure matches the problem structure. Compare: recursive tree traversal is 5 lines, iterative with explicit stack is 15+ lines.',
    keyPoints: [
      'Code structure mirrors problem structure',
      'Trees/graphs: nodes have same structure as subtrees',
      'Divide-and-conquer: break into subproblems of same type',
      'Backtracking: explore all recursive paths',
      'Shorter and more readable than iterative for these cases',
    ],
  },
];
