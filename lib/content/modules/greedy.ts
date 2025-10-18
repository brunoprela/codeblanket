/**
 * Greedy Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/greedy/introduction';
import { patternsSection } from '../sections/greedy/patterns';
import { proofSection } from '../sections/greedy/proof';
import { complexitySection } from '../sections/greedy/complexity';
import { templatesSection } from '../sections/greedy/templates';
import { interviewSection } from '../sections/greedy/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/greedy/introduction';
import { patternsQuiz } from '../quizzes/greedy/patterns';
import { proofQuiz } from '../quizzes/greedy/proof';
import { complexityQuiz } from '../quizzes/greedy/complexity';
import { templatesQuiz } from '../quizzes/greedy/templates';
import { interviewQuiz } from '../quizzes/greedy/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/greedy/introduction';
import { patternsMultipleChoice } from '../multiple-choice/greedy/patterns';
import { proofMultipleChoice } from '../multiple-choice/greedy/proof';
import { complexityMultipleChoice } from '../multiple-choice/greedy/complexity';
import { templatesMultipleChoice } from '../multiple-choice/greedy/templates';
import { interviewMultipleChoice } from '../multiple-choice/greedy/interview';

export const greedyModule: Module = {
  id: 'greedy',
  title: 'Greedy',
  description:
    'Master greedy algorithms that make locally optimal choices to find global optima.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🎯',
  keyTakeaways: [
    'Greedy makes locally optimal choice without reconsidering, hoping for global optimum',
    'Works when greedy choice property + optimal substructure exist',
    'Prove correctness with exchange argument or "stays ahead" method',
    'Often O(n log n) due to sorting, but some greedy algorithms are O(n)',
    'Activity selection: sort by end time, select non-overlapping earliest finish',
    'Fractional knapsack: sort by value/weight ratio, greedy works; 0/1 knapsack needs DP',
    'Huffman/merge problems: always combine two smallest (use min heap)',
    'Jump game patterns: track maximum reachable index in single pass',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
    },
    {
      ...proofSection,
      quiz: proofQuiz,
      multipleChoice: proofMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
    },
    {
      ...templatesSection,
      quiz: templatesQuiz,
      multipleChoice: templatesMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
