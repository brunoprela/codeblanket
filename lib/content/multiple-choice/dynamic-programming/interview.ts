/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a DP problem?',
    options: [
      'Sort, search',
      'Maximum/minimum, count ways, longest/shortest, optimal, can you reach',
      'Shortest path only',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'DP keywords: "maximum/minimum" (optimization), "count ways" (combinations), "longest/shortest" subsequence, "optimal", "can you reach/make". Suggests trying all possibilities optimally.',
  },
  {
    id: 'mc2',
    question: 'How do you approach a DP problem in an interview?',
    options: [
      'Code immediately',
      'Define state, find recurrence, identify base cases, implement (top-down or bottom-up), optimize',
      'Random',
      'Guess pattern',
    ],
    correctAnswer: 1,
    explanation:
      'DP approach: 1) Define state clearly (dp[i] means?), 2) Recurrence relation, 3) Base cases, 4) Implement (start top-down if easier), 5) Optimize space if needed. Communicate throughout.',
  },
  {
    id: 'mc3',
    question: 'What should you clarify in a DP interview?',
    options: [
      'Nothing',
      'Constraints (N size affects O(N²) feasibility), output format (value vs path), edge cases',
      'Random',
      'Language only',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: 1) Input constraints (N≤1000 allows O(N²), N≤10^6 needs O(N)), 2) Output (optimal value vs actual solution path), 3) Edge cases (empty, single element), 4) Multiple solutions or one.',
  },
  {
    id: 'mc4',
    question: 'What is a common DP mistake?',
    options: [
      'Using arrays',
      'Wrong state definition, incorrect base cases, wrong iteration order (dependencies)',
      'Good naming',
      'Comments',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistakes: 1) Vague state definition, 2) Missing/wrong base cases, 3) Computing dp[i] before dependencies ready, 4) Off-by-one errors in indices, 5) Not handling edge cases.',
  },
  {
    id: 'mc5',
    question: 'How should you communicate your DP solution?',
    options: [
      'Just code',
      'Explain state definition, recurrence relation, why it works, walk through example, complexity',
      'No explanation',
      'Random',
    ],
    correctAnswer: 1,
    explanation:
      'Communication: 1) Clear state definition (dp[i] represents...), 2) Recurrence with reasoning, 3) Base cases and why, 4) Walk through small example showing dp table, 5) Time O(?), space O(?), optimization possible.',
  },
];
