/**
 * Multiple choice questions for Interview Strategy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interviewMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What keywords signal a greedy problem?',
    options: [
      'Array, list, tree',
      'Maximum, minimum, optimal, scheduling, earliest/latest',
      'Count all ways',
      'Longest path',
    ],
    correctAnswer: 1,
    explanation:
      'Keywords like "maximum", "minimum", "optimal", "scheduling", "earliest", "latest", "best" suggest greedy. But verify greedy choice property before implementing!',
  },
  {
    id: 'mc2',
    question: 'What should you clarify first in a greedy interview problem?',
    options: [
      'Complexity only',
      'Can I sort? What defines "best"? Any constraints on choices?',
      'Language preference',
      'Nothing',
    ],
    correctAnswer: 1,
    explanation:
      'Clarify: Can you sort (modifies input)? What makes a choice "best"? Are there dependencies between choices? These affect whether greedy works.',
  },
  {
    id: 'mc3',
    question: 'What is a common mistake when using greedy?',
    options: [
      'Sorting correctly',
      'Assuming greedy works without proof - need exchange argument or stays-ahead',
      'Good variable names',
      'Using correct syntax',
    ],
    correctAnswer: 1,
    explanation:
      'Common mistake: assuming greedy works because it "looks right". Always verify with exchange argument, stays-ahead proof, or test counterexamples.',
  },
  {
    id: 'mc4',
    question: "What are red flags that greedy won't work?",
    options: [
      'Optimization problems',
      '"Longest path", "all possible ways", "need to reconsider past choices"',
      'Scheduling problems',
      'Interval problems',
    ],
    correctAnswer: 1,
    explanation:
      'Red flags: "longest path" (use DP), "all possible ways" (use DP/backtracking), "reconsider choices" (not greedy). Greedy makes irrevocable choices.',
  },
  {
    id: 'mc5',
    question: 'What is a good practice progression for greedy?',
    options: [
      'Start with hardest',
      'Week 1: Basics (Jump Game), Week 2: Intervals, Week 3: Advanced (Huffman)',
      'Random order',
      'Skip practice',
    ],
    correctAnswer: 1,
    explanation:
      'Progress: Week 1 basics (jump game, stock) → Week 2 intervals (meetings, activity selection) → Week 3 advanced (Huffman, knapsack). Build proof skills.',
  },
];
