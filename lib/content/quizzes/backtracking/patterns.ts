/**
 * Quiz questions for Backtracking Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Compare the subset and permutation patterns. What is the key difference in how they explore choices?',
    sampleAnswer:
      'Subsets explore include/exclude decisions for each element - binary choice at each step. Permutations explore which element to place at each position - n choices initially, n-1 next, etc. Key difference: subsets maintain element order (no rearrangement), permutations try all arrangements. In subsets, once we skip an element, we never go back to it. In permutations, we try each element at each position using a "used" array or swapping. Subsets generate 2^n results (each element in or out). Permutations generate n! results (all arrangements). The recursion tree shape differs: subsets are binary tree, permutations are n-ary tree with shrinking branches.',
    keyPoints: [
      'Subsets: include/exclude binary choice',
      'Permutations: which element at this position',
      'Subsets: maintain order, 2^n results',
      'Permutations: all arrangements, n! results',
      'Tree shape: binary vs n-ary',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the constraint satisfaction pattern with N-Queens. How do you check if a placement is valid?',
    sampleAnswer:
      'N-Queens places N queens on NxN board so none attack each other. We place one queen per row, trying each column. Before placing, check if valid: no queen in same column, no queen on same diagonal. Track used columns with set. For diagonals, use math: row-col identifies one diagonal direction (45 degrees), row+col identifies the other (135 degrees). These are unique per diagonal. So maintain sets for columns, diag1 (row-col), diag2 (row+col). Place queen if position not in any set, add to sets, recurse to next row, then remove from sets (backtrack). This constraint checking prunes invalid placements early. Without it, we would explore all N^N placements.',
    keyPoints: [
      'One queen per row, try each column',
      'Check: column, two diagonals',
      'Track: column set, diag1 (row-col), diag2 (row+col)',
      'Place, add to sets, recurse, remove from sets',
      'Pruning: N^N → much fewer valid placements',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the path-finding pattern for word search. Why do we need to mark visited cells?',
    sampleAnswer:
      'Word search finds if word exists in grid starting from each cell, moving adjacent (up/down/left/right). From each cell, try all 4 directions recursively if char matches next in word. We mark visited cells to prevent cycles - without marking, we could use same cell multiple times, which violates the problem. Mark current cell as visited before recursing, unmark after returning (backtrack). This ensures each path uses each cell at most once. The visited marking is temporary per path - when we backtrack and try a different direction, previous cells become available again. This is classic backtracking state management: modify (mark), recurse, restore (unmark).',
    keyPoints: [
      'Try starting from each cell',
      'Recurse in 4 directions if char matches',
      'Mark visited to prevent cycles',
      'Unmark after recursion (backtrack)',
      'Temporary marking per path',
    ],
  },
];
