/**
 * Multiple choice questions for Complexity Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const complexityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the time complexity of typical 1D DP?',
    options: [
      'O(log N)',
      'O(N) - iterate through N states, each state O(1) or O(K) work',
      'O(N²)',
      'O(2^N)',
    ],
    correctAnswer: 1,
    explanation:
      '1D DP: N states, each computed once. If each state does O(1) work, total O(N). If trying K choices per state, O(N*K). Example: climbing stairs O(N), coin change O(N*coins).',
  },
  {
    id: 'mc2',
    question: 'What is the time complexity of typical 2D DP?',
    options: [
      'O(N)',
      'O(M*N) - M×N states, each state O(1) work',
      'O(log N)',
      'O(N³)',
    ],
    correctAnswer: 1,
    explanation:
      '2D DP: M×N states. If each state does O(1) work, total O(M*N). Examples: LCS O(M*N), edit distance O(M*N), grid paths O(M*N). With K choices per state: O(M*N*K).',
  },
  {
    id: 'mc3',
    question: 'What is the space complexity of bottom-up DP?',
    options: [
      'Always O(1)',
      'O(number of states) but can often optimize to O(N) or O(1) by keeping only needed previous states',
      'O(2^N)',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Bottom-up: space = DP table size. 1D: O(N), 2D: O(M*N). But often optimizable: if dp[i] only needs dp[i-1], keep O(1). If dp[i][j] needs previous row, keep O(N).',
  },
  {
    id: 'mc4',
    question: 'What is the space complexity of top-down DP?',
    options: [
      'O(1)',
      'O(states) for memo + O(depth) for recursion stack',
      'O(N³)',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down: memo size O(number of states) + recursion stack O(maximum depth). Often both O(N), so total O(N). Less optimizable than bottom-up due to recursion stack.',
  },
  {
    id: 'mc5',
    question: 'How does DP complexity compare to brute force?',
    options: [
      'Same',
      'DP avoids recomputation - often exponential O(2^N) → polynomial O(N²)',
      'DP slower',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'DP dramatically reduces complexity by caching. Fibonacci: O(2^N) → O(N). Knapsack: O(2^N) → O(N*W). DP polynomial vs brute force exponential. Trade-off: O(N) space.',
  },
];
