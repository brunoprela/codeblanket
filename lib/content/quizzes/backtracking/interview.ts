/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize that a problem needs backtracking? What keywords or patterns signal this?',
    sampleAnswer:
      'Several signals indicate backtracking. First, "all possible" or "find all" - generating all solutions. Second, "combinations", "permutations", "subsets" - explicit generation problems. Third, constraint satisfaction: "N-Queens", "Sudoku", "valid placements". Fourth, "can you find a path" or "does there exist" with complex constraints. Fifth, when greedy does not work and you need exhaustive search with pruning. The key question: do I need to explore all possibilities with ability to abandon invalid paths? If yes, backtracking. Examples: "generate all combinations of size k", "find all solutions to puzzle", "count ways to partition".',
    keyPoints: [
      'Keywords: all possible, find all',
      'Explicit: combinations, permutations, subsets',
      'Constraint satisfaction: N-Queens, Sudoku',
      'Path finding with constraints',
      'Need exhaustive search with pruning',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your approach to a backtracking problem in an interview, from identification to explaining complexity.',
    sampleAnswer:
      'First, I identify the pattern: "generate all subsets, so this is backtracking subset pattern". I clarify: duplicates in input? Empty subset allowed? Then I explain approach: "I will use recursive backtracking with start index. At each step, two choices: include current element or skip it. Base case: when start index reaches end, add current path to result". I discuss complexity: "2^n subsets, each takes O(n) to copy, so O(2^n × n) time. O(n) space for recursion depth". I draw decision tree for small example: [1,2] branches into include 1 or not, then include 2 or not, giving [], [2], [1], [1,2]. While coding, I explain choose-explore-unchoose pattern. Finally, I mention optimizations like early pruning if problem has constraints.',
    keyPoints: [
      'Identify pattern and explain why backtracking',
      'Clarify: duplicates, edge cases',
      'Explain approach with base case and choices',
      'State complexity with reasoning',
      'Draw decision tree for example',
      'Code with choose-explore-unchoose commentary',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most common mistakes in backtracking problems and how do you avoid them?',
    sampleAnswer:
      'First: forgetting to copy path when adding to result. Paths are references; without copy, all results point to same list that gets modified. Use path[:] or path.copy(). Second: forgetting to backtrack (unchoose). State must be restored for sibling branches. Third: wrong base case - not checking if solution complete. Fourth: not checking constraints before recursing, leading to wasted exploration. Fifth: modifying input accidentally. Sixth: off-by-one with indices, especially start parameter. My strategy: always use choose-explore-unchoose pattern explicitly, test with small examples, verify state restoration, add constraint checks before recursing. Drawing the recursion tree helps catch logic errors.',
    keyPoints: [
      'Copy path when adding to result',
      'Always backtrack (unchoose)',
      'Check correct base case',
      'Check constraints before recursing',
      'Do not modify input accidentally',
      'Test small examples, draw recursion tree',
    ],
  },
];
