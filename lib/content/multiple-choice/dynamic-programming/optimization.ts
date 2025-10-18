/**
 * Multiple choice questions for Space Optimization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const optimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is space optimization in DP?',
    options: [
      'Delete variables',
      'Reduce space from O(N²) to O(N) or O(N) to O(1) by keeping only needed previous states',
      'Random',
      'Compress data',
    ],
    correctAnswer: 1,
    explanation:
      'Space optimization: observe that dp[i] often only depends on few previous values (dp[i-1], dp[i-2]). Keep only those instead of entire array. Example: Fibonacci O(N)→O(1) with two variables.',
  },
  {
    id: 'mc2',
    question: 'How do you optimize Fibonacci from O(N) space to O(1)?',
    options: [
      'Different algorithm',
      'Keep only prev2 and prev1 variables instead of entire dp array',
      'Cannot optimize',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Fibonacci only needs last 2 values. Instead of dp array, use prev2=0, prev1=1, curr=prev1+prev2. Update: prev2=prev1, prev1=curr. O(1) space vs O(N).',
  },
  {
    id: 'mc3',
    question: 'How do you optimize 2D DP from O(M*N) to O(N)?',
    options: [
      'Cannot',
      'Keep only current and previous row if dp[i][j] only depends on current and previous row',
      'Use hash map',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'For grid DP where dp[i][j] depends on dp[i-1][j] and dp[i][j-1], keep only 2 rows (prev and curr) instead of entire M×N table. Update row-by-row. O(N) space.',
  },
  {
    id: 'mc4',
    question: 'What is the trade-off between top-down and bottom-up space?',
    options: [
      'No difference',
      'Top-down: O(N) recursion stack + O(N) memo. Bottom-up: O(N) table only (can optimize further)',
      'Top-down always better',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Top-down uses recursion stack O(N) plus memo O(N). Bottom-up uses table O(N) and can be optimized to O(1) by keeping only needed states. Bottom-up generally more space-efficient.',
  },
  {
    id: 'mc5',
    question: 'What is state compression?',
    options: [
      'Data compression',
      'Use bitmask or compact representation for DP state - 2D→1D using encoding',
      'Random',
      'Delete states',
    ],
    correctAnswer: 1,
    explanation:
      'State compression: encode complex state compactly. Example: use bitmask for subset instead of array. Traveling salesman: dp[mask][i] where mask is visited cities as bits. Reduces dimensions.',
  },
];
