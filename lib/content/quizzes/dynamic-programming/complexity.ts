/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain why 2D DP is often O(n × m). What do n and m typically represent?',
    sampleAnswer:
      'Two nested loops create O(n × m) time. Outer loop iterates first dimension (n), inner loop iterates second dimension (m). For Knapsack: n items, m weight limits, so O(n × m). For LCS: n characters in first string, m in second, so O(n × m). Each cell computed in O(1) using values from already-computed cells. Total cells = n × m, each O(1), giving O(n × m). The dimensions represent the two independent variables in the DP state. Space is also O(n × m) for full table, but can sometimes optimize to O(m) with rolling array if only previous row needed. Common pattern: two parameters in state definition leads to 2D table and O(n × m) complexity.',
    keyPoints: [
      'Two nested loops: outer n, inner m',
      'n and m: independent variables in state',
      'Knapsack: n items, m weights',
      'LCS: n and m are string lengths',
      'Time O(n × m), space can optimize to O(m)',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare time complexity of top-down vs bottom-up DP. Which is faster and why?',
    sampleAnswer:
      'Both have same asymptotic complexity (same number of subproblems), but bottom-up is faster in practice. Top-down has recursion overhead: function call stack, parameter passing, return values. Each recursive call costs cycles. Bottom-up uses simple loops with array access - much faster per operation. Top-down only computes needed subproblems which can help if many subproblems unused. Bottom-up computes all subproblems. For most interview problems, all subproblems needed anyway. Also bottom-up has better cache locality (sequential memory access) vs top-down (scattered recursive calls). In practice, bottom-up can be 2-3x faster. Use top-down when dependencies complex or not all subproblems needed. Use bottom-up for best performance.',
    keyPoints: [
      'Same asymptotic complexity',
      'Bottom-up faster: no recursion overhead',
      'Top-down: only needed subproblems',
      'Bottom-up: better cache locality',
      'Practice: bottom-up 2-3x faster',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the relationship between DP complexity and problem constraints. How do you estimate if DP will TLE (Time Limit Exceeded)?',
    sampleAnswer:
      'DP complexity determined by: number of states × transition cost. For n=1000, O(n^2) gives 1M operations - acceptable. O(n^3) gives 1B operations - might TLE depending on time limit. General rule: 10^8 operations per second, so 10^8 ops takes 1 second. For n=10^4, O(n^2)=10^8 is borderline. O(n^3)=10^12 will definitely TLE. Check problem constraints: if n <= 100, O(n^3) okay. If n <= 1000, need O(n^2). If n <= 10^5, need O(n log n) or better. DP states usually O(n) or O(n^2). Transition usually O(1) to O(n). Multiply to get total. If exceeds 10^8, consider optimizations or different approach.',
    keyPoints: [
      'Complexity: number of states × transition cost',
      'Rule: 10^8 operations per second',
      'n=10^4: O(n^2) borderline, O(n^3) TLE',
      'Check constraints: n <= 100 → O(n^3) okay',
      'Estimate: states × transitions, compare to 10^8',
    ],
  },
];
