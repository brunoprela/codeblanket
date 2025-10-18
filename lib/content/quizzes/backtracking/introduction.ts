/**
 * Quiz questions for Introduction to Backtracking section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what backtracking is and how it differs from brute force. What makes it more efficient?',
    sampleAnswer:
      'Backtracking is a systematic way to explore all possible solutions by making choices incrementally and abandoning paths that cannot lead to valid solutions. It differs from brute force in that it prunes the search space - when we detect a path cannot succeed, we backtrack immediately rather than exploring all possibilities. For example, in N-Queens, if placing a queen creates a conflict, we backtrack without trying to place remaining queens on that board. Brute force would try all placements. The efficiency comes from pruning: we avoid exploring exponentially many invalid paths. Backtracking is essentially DFS with pruning - explore, check constraints, backtrack if invalid, continue if valid.',
    keyPoints: [
      'Incremental choice-making with pruning',
      'Abandon invalid paths early',
      'vs Brute force: explores all possibilities',
      'Prunes search space by checking constraints',
      'DFS with constraint checking',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the three steps of backtracking pattern. Why is the "undo" step crucial?',
    sampleAnswer:
      'The three steps are: choose (make a choice), explore (recurse with that choice), unchoose (undo the choice). The undo step is crucial because it restores the state for exploring other branches. Without undo, previous choices pollute the state for sibling branches. For example, in permutations, after exploring with element A in position 1, we must remove A before trying B in position 1. The undo ensures each branch starts from the same parent state. This is what enables systematic exploration of the entire solution space - each path is independent. The pattern: modify state, recurse, restore state.',
    keyPoints: [
      'Three steps: choose, explore, unchoose',
      'Undo restores state for other branches',
      'Without undo: state pollution',
      'Enables independent exploration of branches',
      'Pattern: modify, recurse, restore',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through when you would use backtracking vs other approaches like greedy or dynamic programming.',
    sampleAnswer:
      'Use backtracking when you need to find all solutions or explore all possibilities with constraints - problems where greedy does not work and DP does not apply. Greedy makes locally optimal choices and cannot backtrack - use when local optimum leads to global optimum. DP solves overlapping subproblems by memoization - use when problem has optimal substructure. Backtracking is for: generating all combinations/permutations, constraint satisfaction like N-Queens, finding all paths. For example, subset sum: backtracking finds all subsets that sum to target. DP finds if any subset exists. Backtracking explores decision trees with pruning when you need exhaustive search.',
    keyPoints: [
      'Backtracking: all solutions, constraints, exploration',
      'Greedy: local optimum, no backtracking',
      'DP: overlapping subproblems, memoization',
      'Backtracking for: all solutions, constraint satisfaction',
      'Use when exhaustive search needed with pruning',
    ],
  },
];
