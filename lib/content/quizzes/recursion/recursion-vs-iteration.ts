/**
 * Quiz questions for Recursion vs Iteration section
 */

export const recursionvsiterationQuiz = [
  {
    id: 'q-reciter1',
    question: 'When would you choose recursion over iteration?',
    sampleAnswer:
      'Choose recursion when: 1) The problem has a natural recursive structure (tree traversal, divide-and-conquer), 2) The recursive solution is significantly clearer and more maintainable, 3) Stack depth is manageable (not too deep), 4) Performance difference is acceptable for the use case. Recursion shines for problems like tree traversal, backtracking, and divide-and-conquer where the recursive formulation maps directly to the problem structure.',
    keyPoints: [
      'Natural recursive structure (trees, graphs, divide-and-conquer)',
      'Clearer, more maintainable code',
      'Manageable stack depth',
      'Examples: tree traversal, backtracking, merge sort',
      'Consider when code clarity outweighs minor performance cost',
    ],
  },
  {
    id: 'q-reciter2',
    question: 'What are the main drawbacks of recursion compared to iteration?',
    sampleAnswer:
      'Main drawbacks: 1) **Stack overflow risk** - each call uses stack space, deep recursion can crash, 2) **Performance overhead** - function calls have overhead (saving registers, return addresses), 3) **Memory usage** - O(n) stack space vs O(1) for iteration, 4) **Harder to debug** - call stack can be complex. However, tail-call optimization (in some languages) and memoization can mitigate some issues.',
    keyPoints: [
      'Stack overflow with deep recursion',
      'Function call overhead',
      'O(n) stack space vs O(1) for iteration',
      'More complex debugging',
      'Not all languages optimize tail recursion',
    ],
  },
  {
    id: 'q-reciter3',
    question: 'How can you convert a recursive solution to iterative?',
    sampleAnswer:
      'Use an explicit stack to simulate the call stack: 1) Replace recursive calls with stack push/pop, 2) Use a while loop with stack.isEmpty() condition, 3) Track state that would be in function parameters. Example: recursive DFS becomes iterative with a stack of nodes. Sometimes use a queue (for BFS-like traversal). The key is manually managing what the language does automatically with recursion.',
    keyPoints: [
      'Use explicit stack data structure',
      'Replace recursive calls with push/pop',
      'While loop until stack empty',
      'Track state manually (function parameters)',
      'Example: DFS with stack, BFS with queue',
    ],
  },
];
