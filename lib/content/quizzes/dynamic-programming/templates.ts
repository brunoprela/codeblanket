/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the 1D DP bottom-up template. What are the key steps?',
    sampleAnswer:
      'The 1D template has five parts. First, initialize DP array of size n+1 with default values (often 0 or infinity). Second, set base cases explicitly (dp[0], dp[1], etc.). Third, iterate from base cases to n. Fourth, for each i, compute dp[i] using recurrence relation (typically dp[i-1], dp[i-2], etc.). Fifth, return dp[n] as final answer. For Climbing Stairs: dp = array of size n+1, dp[0]=1, dp[1]=1 (base), loop i from 2 to n, dp[i] = dp[i-1] + dp[i-2] (recurrence), return dp[n] (answer). This template works for: Fibonacci, House Robber, Min Cost Climbing, Jump Game - any 1D problem with dependencies on previous states.',
    keyPoints: [
      'Initialize dp array (size n+1)',
      'Set base cases explicitly',
      'Iterate from base to n',
      'Compute dp[i] using recurrence',
      'Return dp[n] as answer',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the 2D DP template for problems like Knapsack or LCS. How do you handle the nested loops?',
    sampleAnswer:
      'The 2D template creates table dp[n+1][m+1]. First, initialize entire table (often with 0 or infinity). Second, set base cases for row 0 and column 0. Third, nested loops: outer loop i from 1 to n (first dimension), inner loop j from 1 to m (second dimension). Fourth, for each cell dp[i][j], compute using values from dp[i-1][...] and dp[...][j-1] based on recurrence. Fifth, return dp[n][m]. For LCS: outer loop for first string, inner loop for second string. If characters match: dp[i][j] = 1 + dp[i-1][j-1]. Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1]). This template extends to: Knapsack, Edit Distance, Grid Paths.',
    keyPoints: [
      'Create 2D table dp[n+1][m+1]',
      'Base cases: row 0 and column 0',
      'Nested loops: i from 1 to n, j from 1 to m',
      'Compute dp[i][j] from neighbors',
      'Return dp[n][m]',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the top-down memoization template. How does it differ from bottom-up?',
    sampleAnswer:
      'Top-down uses recursion with caching. First, create memo dictionary or array. Second, write recursive helper with parameters representing state. Third, check if state in memo; if yes, return cached value. Fourth, if base case, return base value. Fifth, compute result using recursive calls to smaller subproblems. Sixth, store result in memo before returning. Seventh, call helper with initial state. For Fibonacci: helper(n) checks memo, if n <= 1 return n, else compute memo[n] = helper(n-1) + helper(n-2), return memo[n]. Differs from bottom-up: solves top-down (big to small), only computes needed states, has recursion overhead. Bottom-up: solves bottom-up (small to big), computes all states, uses iteration.',
    keyPoints: [
      'Create memo, write recursive helper',
      'Check memo first, return if cached',
      'Base case: return directly',
      'Recurse for smaller subproblems',
      'Store and return result',
      'vs Bottom-up: top-down solving, only needed states',
    ],
  },
];
