/**
 * Quiz questions for Common DP Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the 0/1 Knapsack pattern. Why is it called 0/1 and how does the 2D DP table work?',
    sampleAnswer:
      'Called 0/1 because for each item, you have binary choice: include (1) or exclude (0) it. Cannot take partial items or multiple copies. The 2D table dp[i][w] means maximum value using first i items with weight limit w. Recurrence: dp[i][w] = max (skip item i which gives dp[i-1][w], or take item i which gives value[i] + dp[i-1][w-weight[i]]). Take option only valid if weight[i] <= w. Build table row by row: for each item, for each weight, decide take or skip. Final answer dp[n][W] is max value using all n items with weight limit W. This pattern extends to many problems: subset sum, partition, target sum - all involve include/exclude decisions.',
    keyPoints: [
      '0/1: binary choice (include or exclude)',
      'dp[i][w]: max value, first i items, weight limit w',
      'Recurrence: max (skip, take if weight allows)',
      'Build row by row: for each item, each weight',
      'Extends to: subset sum, partition, target sum',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the Longest Common Subsequence pattern. How do you fill the 2D table?',
    sampleAnswer:
      'LCS finds longest subsequence common to two strings. Not substring - elements need not be consecutive but must maintain order. Table dp[i][j] is LCS length of s1[0..i-1] and s2[0..j-1]. If s1[i-1] equals s2[j-1], characters match: dp[i][j] = 1 + dp[i-1][j-1]. If no match, take max of excluding either character: dp[i][j] = max (dp[i-1][j], dp[i][j-1]). Base case: dp[0][j] = dp[i][0] = 0 (empty string has LCS 0). Fill table row by row, left to right. For "abcde" and "ace", dp[5][3] = 3 (subsequence "ace"). This pattern extends to edit distance, diff algorithms, sequence alignment.',
    keyPoints: [
      'LCS: longest common subsequence (maintain order)',
      'dp[i][j]: LCS of s1[0..i-1] and s2[0..j-1]',
      'Match: 1 + dp[i-1][j-1]',
      'No match: max (dp[i-1][j], dp[i][j-1])',
      'Extends to: edit distance, diff, alignment',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the State Machine DP pattern with Buy/Sell Stock. Why do we need multiple states?',
    sampleAnswer:
      'Stock problems have constraints on transitions between states: can only sell after buying, might have cooldown, might limit transactions. Model as state machine with DP. For cooldown problem: three states per day - holding stock (just bought or continuing to hold), sold today (just sold), resting (waiting, cooldown or never bought). Each state depends on previous day states based on valid transitions. Holding today = max (continue holding, buy from rest). Sold today = sell from holding. Rest today = max (continue rest, cooldown after sold). Multiple states capture the constraints: cannot buy immediately after selling due to cooldown. Final answer is max of sold and rest on last day (cannot be holding). This pattern handles complex state transitions in sequential decision problems.',
    keyPoints: [
      'Model constraints as state machine',
      'Example: holding, sold, resting states',
      'Each state: DP based on valid prev transitions',
      'Captures complex constraints (cooldown)',
      'Answer: max of valid final states',
    ],
  },
];
