/**
 * Multiple choice questions for Introduction to Greedy Algorithms section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the core principle of greedy algorithms?',
    options: [
      'Try all possibilities',
      'Make the locally optimal choice at each step without reconsidering',
      'Use memoization',
      'Divide and conquer',
    ],
    correctAnswer: 1,
    explanation:
      'Greedy algorithms make the locally optimal choice at each step (the choice that looks best right now) and never reconsider past decisions, hoping this leads to a global optimum.',
  },
  {
    id: 'mc2',
    question: 'What two properties must hold for a greedy algorithm to work?',
    options: [
      'Fast and simple',
      'Greedy choice property and optimal substructure',
      'Sorting and hashing',
      'Recursion and iteration',
    ],
    correctAnswer: 1,
    explanation:
      'Greedy works when: 1) Greedy choice property - locally optimal choices lead to global optimum, 2) Optimal substructure - optimal solution contains optimal solutions to subproblems.',
  },
  {
    id: 'mc3',
    question:
      'Why does greedy fail for coin change with arbitrary denominations?',
    options: [
      'Too slow',
      'Locally optimal choice (largest coin) may not lead to global optimum',
      'Uses too much memory',
      'Cannot handle coins',
    ],
    correctAnswer: 1,
    explanation:
      "For coins [25,20,5,1] and amount 40: greedy takes 25 first (locally best), needs 4 coins total. Optimal is 20+20=2 coins. The locally optimal choice doesn't guarantee global optimum.",
  },
  {
    id: 'mc4',
    question: 'How does greedy compare to dynamic programming?',
    options: [
      'Greedy is always better',
      'Greedy is faster but works on fewer problems; DP is slower but always finds optimal',
      'They are the same',
      'DP is always better',
    ],
    correctAnswer: 1,
    explanation:
      'Greedy makes one choice per step (faster) but only works when greedy choice property holds. DP tries all choices (slower) but always finds optimal solution for problems with optimal substructure.',
  },
  {
    id: 'mc5',
    question:
      'What is a common greedy strategy for activity/interval selection?',
    options: [
      'Select randomly',
      'Select activity finishing earliest (earliest deadline first)',
      'Select longest activity',
      'Select activity starting latest',
    ],
    correctAnswer: 1,
    explanation:
      'Activity selection: sort by end time, greedily select activities finishing earliest. This maximizes remaining time for future activities, giving optimal non-overlapping selection.',
  },
];
