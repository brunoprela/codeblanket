/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain why permutations are O(n! × n) while subsets are O(2^n × n). What causes the factorial vs exponential difference?',
    sampleAnswer:
      'Permutations generate all arrangements of n elements: n choices for first position, n-1 for second, n-2 for third, etc. This gives n × (n-1) × (n-2) × ... × 1 = n! permutations. Each takes O(n) to copy, giving O(n! × n). Subsets make binary include/exclude decision for each element: 2 choices per element for n elements gives 2^n subsets. Each takes O(n) to copy, giving O(2^n × n). The difference: permutations have shrinking choices at each level (n, n-1, n-2...), subsets have fixed 2 choices per level. For n=5: permutations = 120, subsets = 32. Factorial grows much faster than exponential for larger n.',
    keyPoints: [
      'Permutations: n choices, then n-1, n-2... = n!',
      'Subsets: 2 choices per element = 2^n',
      'Both: O(n) to copy each solution',
      'Factorial vs exponential: shrinking vs fixed choices',
      'n=5: 120 perms vs 32 subsets',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through N-Queens complexity. Why is it O(n!) rather than O(n^n)?',
    sampleAnswer:
      'N-Queens places one queen per row, trying each column. First row has n choices, but second row has fewer valid choices due to constraints (column and diagonal conflicts). On average, later rows have exponentially fewer valid choices. Without any pruning, it would be O(n^n) - n choices per row for n rows. With constraint checking, we prune invalid placements early, reducing to approximately O(n!) - similar to permutations but even less due to diagonal constraints. The exact complexity is hard to express but empirically closer to n! than n^n. For n=8, n^n would be 16 million, n! is 40 thousand, and actual is even less due to pruning. Constraint checking massively reduces search space.',
    keyPoints: [
      'Without pruning: O(n^n) - n choices per row',
      'With constraints: invalid placements pruned',
      'Reduces to approximately O(n!)',
      'Actually less than n! due to diagonal constraints',
      'n=8: n^n=16M, n!=40K, actual even less',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe early pruning optimization. How much does it improve backtracking performance?',
    sampleAnswer:
      'Early pruning checks constraints before recursing rather than after building complete solution. Bad approach: build entire solution, then check if valid. Good approach: check validity at each step, backtrack immediately if invalid. For example, in N-Queens, check column and diagonal conflicts before placing queen. If invalid, do not recurse to next row. This prevents exploring entire subtrees that cannot succeed. The improvement is exponential - instead of exploring b^d nodes (b = branching, d = depth), we might explore b^(d/2) or less. For N-Queens, without pruning, we explore n^n placements. With pruning, much less. Early pruning is the core of what makes backtracking practical.',
    keyPoints: [
      'Check constraints before recursing, not after',
      'Prevents exploring invalid subtrees',
      'Bad: build complete solution, then check',
      'Good: check at each step, backtrack early',
      'Improvement: exponential reduction in search space',
    ],
  },
];
