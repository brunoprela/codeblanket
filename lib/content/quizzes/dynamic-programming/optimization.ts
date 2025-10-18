/**
 * Quiz questions for Space Optimization section
 */

export const optimizationQuiz = [
  {
    id: 'q1',
    question:
      'Explain the space optimization technique. When can you reduce from O(n) to O(1)?',
    sampleAnswer:
      'Space can be optimized when each dp[i] only depends on fixed number of previous states. For Fibonacci, dp[i] = dp[i-1] + dp[i-2] only needs last two values, not entire array. Use two variables instead of array: prev1 (dp[i-1]), prev2 (dp[i-2]). Each iteration: compute current = prev1 + prev2, then update prev2 = prev1, prev1 = current. This reduces O(n) space to O(1). Works for: Climbing Stairs, House Robber, any problem with fixed lookback distance. Cannot optimize if dependencies are variable or need entire history. For 2D DP, sometimes reduce to 1D by processing row by row if only previous row needed. Key: identify minimum state needed for next computation.',
    keyPoints: [
      'Optimize when dp[i] depends on fixed prev states',
      'Use variables instead of array',
      'Example: Fibonacci needs only prev two',
      'O(n) → O(1) for fixed lookback',
      '2D → 1D if only previous row needed',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe rolling array for 2D DP space optimization. When is it applicable?',
    sampleAnswer:
      'Rolling array reduces 2D DP from O(n×m) to O(m) space when computing row i only needs row i-1. Instead of full 2D table, maintain two 1D arrays: current row and previous row. Process row by row: compute current row using previous row, then swap (previous = current, reset current). For Knapsack: dp[i][w] only depends on dp[i-1][w] and dp[i-1][w-weight], so we only need previous row. Can further optimize to single array by iterating weights right to left (prevents overwriting needed values). This technique works for: Knapsack, LCS, Edit Distance, grid path problems. Key requirement: dependencies only on previous row, not arbitrary past rows.',
    keyPoints: [
      'For 2D where row i only needs row i-1',
      'Two 1D arrays: current and previous',
      'Process row by row, swap arrays',
      'Further: single array with right-to-left',
      'Works: Knapsack, LCS, Edit Distance, grid paths',
    ],
  },
  {
    id: 'q3',
    question:
      'What are common mistakes when optimizing DP space? How do you avoid them?',
    sampleAnswer:
      'First mistake: optimizing space before getting correct solution. Always solve with full space first, verify correctness, then optimize. Second: wrong iteration order after space optimization. For single array Knapsack, must iterate weights right to left to avoid overwriting needed values. Left to right would use already-updated values. Third: losing ability to reconstruct solution path. If problem asks for actual path, not just value, might need to keep full table. Fourth: incorrect variable updates - forgetting to save old values before overwriting. Fifth: optimizing when dependencies are not strictly local. My strategy: solve correctly first, identify dependencies, choose right optimization, test thoroughly, verify iteration order.',
    keyPoints: [
      'Always solve correctly with full space first',
      'Wrong iteration order (esp. single array)',
      'Knapsack: right-to-left to avoid overwrite',
      'Might lose path reconstruction ability',
      'Test thoroughly after optimization',
    ],
  },
];
