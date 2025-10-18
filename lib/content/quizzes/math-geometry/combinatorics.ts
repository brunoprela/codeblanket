/**
 * Quiz questions for Combinatorics and Sequences section
 */

export const combinatoricsQuiz = [
  {
    id: 'q1',
    question: 'Explain permutations vs combinations. How do you compute each?',
    sampleAnswer:
      'Permutations: order matters. nPr = n!/(n-r)! ways to arrange r items from n. Combinations: order does not matter. nCr = n!/(r!(n-r)!) ways to choose r from n. For example, 3 items {A,B,C}, choose 2: permutations are AB, BA, AC, CA, BC, CB (6 = 3P2). Combinations are AB, AC, BC (3 = 3C2). Compute: use formula with factorials, but watch overflow. Better: nCr = nC(n-r) (symmetry), compute iteratively to avoid large factorials. Pascal triangle: nCr = (n-1)C(r-1) + (n-1)Cr. For example, 5C3 = 5!/(3!2!) = 120/(6×2) = 10. Applications: counting problems, probability, choosing teams, generating combinations.',
    keyPoints: [
      'Permutations: order matters, nPr = n!/(n-r)!',
      'Combinations: order irrelevant, nCr = n!/(r!(n-r)!)',
      'Compute iteratively to avoid overflow',
      'Pascal triangle: nCr = (n-1)C(r-1) + (n-1)Cr',
      'Uses: counting, probability, team selection',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe Fibonacci sequence. What are different ways to compute it?',
    sampleAnswer:
      'Fibonacci: F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1. Sequence: 0,1,1,2,3,5,8,13,... Approach 1: recursive F(n) = F(n-1) + F(n-2), O(2^n) time (exponential, slow). Approach 2: memoization (top-down DP), O(n) time and space. Approach 3: iteration (bottom-up DP), O(n) time, O(1) space. Approach 4: matrix exponentiation, O(log n) time. Approach 5: closed-form (Binet formula), O(1) but precision issues. For example, F(10): recursive does 177 calls, iteration does 10 steps. Matrix method: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n. Use fast power for O(log n). Best: iteration for moderate n, matrix for very large n.',
    keyPoints: [
      'F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1',
      'Recursive: O(2^n) - slow',
      'Iteration: O(n) time, O(1) space - good',
      'Matrix exponentiation: O(log n) - best for large n',
      'Binet formula: O(1) but precision issues',
    ],
  },
  {
    id: 'q3',
    question: 'Explain Catalan numbers. What problems do they solve?',
    sampleAnswer:
      'Catalan numbers: C(n) = (2n)! / ((n+1)!n!) = C(n-1) × 2(2n-1)/(n+1). Sequence: 1,1,2,5,14,42,... Count structures with recursive nesting. Problems: 1) Number of valid parenthesis sequences (n pairs). 2) Number of BSTs with n nodes. 3) Number of ways to triangulate polygon with n+2 sides. 4) Number of paths in n×n grid (not crossing diagonal). 5) Number of binary trees with n nodes. For example, C(3) = 5: valid parentheses are ((())), (()()), (())(), ()(()), ()()(). BSTs with 3 nodes: 5 different structures. Recurrence: C(n) = sum C(i)×C(n-1-i) for i=0 to n-1. Compute iteratively with formula. Catalan appears in many combinatorial problems with nested or recursive structure.',
    keyPoints: [
      'C(n) = (2n)! / ((n+1)!n!), recursive structures',
      'Sequence: 1,1,2,5,14,42,...',
      'Problems: valid parentheses, BSTs, triangulations',
      'Recurrence: C(n) = sum C(i)×C(n-1-i)',
      'Appears in: nested, recursive, binary structures',
    ],
  },
];
