/**
 * Quiz questions for DP Pattern Recognition & Decision Guide section
 */

export const patternrecognitionQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through how you would recognize that a problem needs DP versus greedy. Give specific examples.',
    hint: 'Think about making locally optimal choices vs needing to consider all possibilities.',
    sampleAnswer:
      'DP is needed when local choices do not guarantee global optimum - you need to explore different paths and combine subproblem solutions. Greedy works when local optimum always leads to global optimum. Example: Coin change with coins [1,3,4] for amount 6. Greedy picks largest first: 4+1+1=3 coins. But optimal is 3+3=2 coins. Greedy fails here, need DP. Conversely, activity selection problem: choosing maximum non-overlapping intervals. Greedy works - always pick earliest ending interval. The key test: does making locally optimal choice at each step guarantee globally optimal solution? If no, use DP. If yes, greedy suffices.',
    keyPoints: [
      'DP: local optimum ≠ global optimum',
      'Greedy: local optimum = global optimum',
      'Example: coin change needs DP',
      'Example: activity selection can use greedy',
      'Test: does greedy choice guarantee optimal solution?',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between counting DP and optimization DP. How does this affect your state definition?',
    hint: 'Think about what you are accumulating in your dp array.',
    sampleAnswer:
      'Counting DP counts number of ways to achieve something. State stores count, recurrence sums up counts from different paths. Example: "How many ways to climb n stairs?" dp[i] = dp[i-1] + dp[i-2] (sum). Optimization DP finds maximum or minimum. State stores best value, recurrence picks best option. Example: "Maximum money robbing houses?" dp[i] = max (dp[i-1], dp[i-2]+nums[i]) (max). The key difference: counting uses sum/addition to combine paths, optimization uses min/max to pick best path. This affects state definition - counting needs count type, optimization needs value type that you are optimizing.',
    keyPoints: [
      'Counting DP: accumulates number of ways (sum)',
      'Optimization DP: finds best value (min/max)',
      'Example: climbing stairs counts, house robber optimizes',
      'Counting combines paths, optimization chooses best path',
      'Different recurrence patterns: sum vs min/max',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is it important to solve DP top-down first before converting to bottom-up? What can go wrong if you skip this step?',
    hint: 'Think about understanding dependencies and iteration order.',
    sampleAnswer:
      'Top-down recursion with memoization makes dependencies explicit - you naturally call what you need. This helps you understand the problem structure and catch edge cases. When you jump straight to bottom-up, you might: 1) Get iteration order wrong (compute dp[i] before its dependencies), 2) Miss base cases, 3) Not understand why certain states are needed. Example: if dp[i] depends on dp[i-1] and dp[i-3], top-down makes this obvious through recursive calls. In bottom-up, you need to carefully iterate ensuring i-3 is computed before i. Starting recursive also helps in interviews - you can present working solution quickly, then optimize to iterative if time permits.',
    keyPoints: [
      'Top-down makes dependencies explicit',
      'Helps understand problem structure',
      'Catches edge cases naturally',
      'Risk of wrong iteration order in bottom-up',
      'Interview strategy: recursive first, optimize later',
    ],
  },
];
