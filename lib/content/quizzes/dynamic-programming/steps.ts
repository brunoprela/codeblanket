/**
 * Quiz questions for The 5-Step DP Framework section
 */

export const stepsQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the 5-step DP framework with a concrete example like Climbing Stairs.',
    sampleAnswer:
      'Climbing Stairs: reach top of n stairs, can climb 1 or 2 steps. Step 1 (State): dp[i] = ways to reach stair i. Step 2 (Recurrence): dp[i] = dp[i-1] + dp[i-2] because we can reach stair i from i-1 (one step) or i-2 (two steps). Step 3 (Base case): dp[0] = 1 (one way to stay at ground), dp[1] = 1 (one way to reach first stair). Step 4 (Order): iterate i from 2 to n because dp[i] depends on dp[i-1] and dp[i-2]. Step 5 (Answer): dp[n] is ways to reach top. This framework forces systematic thinking: define what you are computing, how to compute it, what are starting values, in what order, and where final answer is.',
    keyPoints: [
      'Step 1: dp[i] = ways to reach stair i',
      'Step 2: dp[i] = dp[i-1] + dp[i-2]',
      'Step 3: dp[0] = 1, dp[1] = 1',
      'Step 4: iterate i from 2 to n',
      'Step 5: answer is dp[n]',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how to derive a recurrence relation. What questions do you ask yourself?',
    sampleAnswer:
      'To derive recurrence, ask: how can I solve dp[i] using smaller subproblems? What choices do I have at position i? For Fibonacci, to get F(n), I add F(n-1) and F(n-2). For Coin Change, to make amount i, I try each coin and take minimum of (1 + ways to make i-coin). For House Robber, at house i, I either rob it (take value + dp[i-2]) or skip it (take dp[i-1]). Pattern: express current state as combination of previous states based on problem constraints. Draw small examples (n=3, n=4) and see pattern. The recurrence captures the decision or combination logic of the problem. This is the creative step - finding the relationship between subproblems.',
    keyPoints: [
      'Ask: how to solve dp[i] from smaller subproblems?',
      'What choices at position i?',
      'Express current state from previous states',
      'Fibonacci: sum previous two',
      'House Robber: max of (rob + skip previous, skip)',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe base case initialization. What happens if you initialize incorrectly?',
    sampleAnswer:
      'Base cases are smallest subproblems that do not depend on others - directly solvable. For Fibonacci: F(0)=0, F(1)=1. For Climbing Stairs: ways[0]=1, ways[1]=1. For Coin Change: dp[0]=0 (zero coins for amount 0). Initialize wrong and entire solution fails because all other values build on base cases. Common mistake: forgetting edge cases like empty array or zero amount. For example, if you set ways[0]=0 for Climbing Stairs, all subsequent values will be wrong. Base cases seed the DP table - they must be correct. Test base cases independently before running full algorithm. They represent the termination condition of recursive formulation.',
    keyPoints: [
      'Base cases: smallest subproblems, directly solvable',
      'Do not depend on other subproblems',
      'Wrong base case → entire solution fails',
      'All other values build on base cases',
      'Test base cases independently',
    ],
  },
];
