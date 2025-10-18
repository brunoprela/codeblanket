/**
 * Dynamic Programming Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/dynamic-programming/introduction';
import { stepsSection } from '../sections/dynamic-programming/steps';
import { patternsSection } from '../sections/dynamic-programming/patterns';
import { optimizationSection } from '../sections/dynamic-programming/optimization';
import { complexitySection } from '../sections/dynamic-programming/complexity';
import { templatesSection } from '../sections/dynamic-programming/templates';
import { interviewSection } from '../sections/dynamic-programming/interview';
import { patternrecognitionSection } from '../sections/dynamic-programming/pattern-recognition';

// Import quizzes
import { introductionQuiz } from '../quizzes/dynamic-programming/introduction';
import { stepsQuiz } from '../quizzes/dynamic-programming/steps';
import { patternsQuiz } from '../quizzes/dynamic-programming/patterns';
import { optimizationQuiz } from '../quizzes/dynamic-programming/optimization';
import { complexityQuiz } from '../quizzes/dynamic-programming/complexity';
import { templatesQuiz } from '../quizzes/dynamic-programming/templates';
import { interviewQuiz } from '../quizzes/dynamic-programming/interview';
import { patternrecognitionQuiz } from '../quizzes/dynamic-programming/pattern-recognition';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/dynamic-programming/introduction';
import { stepsMultipleChoice } from '../multiple-choice/dynamic-programming/steps';
import { patternsMultipleChoice } from '../multiple-choice/dynamic-programming/patterns';
import { optimizationMultipleChoice } from '../multiple-choice/dynamic-programming/optimization';
import { complexityMultipleChoice } from '../multiple-choice/dynamic-programming/complexity';
import { templatesMultipleChoice } from '../multiple-choice/dynamic-programming/templates';
import { interviewMultipleChoice } from '../multiple-choice/dynamic-programming/interview';
import { patternrecognitionMultipleChoice } from '../multiple-choice/dynamic-programming/pattern-recognition';

export const dynamicProgrammingModule: Module = {
  id: 'dynamic-programming',
  title: 'Dynamic Programming',
  description:
    'Master the art of breaking problems into overlapping subproblems and building optimal solutions.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🧩',
  keyTakeaways: [
    'DP solves problems with optimal substructure and overlapping subproblems',
    'Two approaches: Top-down (memoization) and Bottom-up (tabulation)',
    '5-step framework: Define state, find recurrence, base cases, iteration order, compute answer',
    'Common patterns: Knapsack, LCS, palindromes, grid paths, decision making',
    'Typical complexity: O(n) for 1D, O(m*n) for 2D DP problems',
    'Space optimization: Often reduce O(n) to O(1) or O(m*n) to O(n)',
    'DP dramatically reduces exponential time to polynomial (2ⁿ → n²)',
    'Clear state definition is crucial - be precise about what dp[i] represents',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...stepsSection,
      quiz: stepsQuiz,
      multipleChoice: stepsMultipleChoice,
    },
    {
      ...patternsSection,
      quiz: patternsQuiz,
      multipleChoice: patternsMultipleChoice,
    },
    {
      ...optimizationSection,
      quiz: optimizationQuiz,
      multipleChoice: optimizationMultipleChoice,
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
    {
      ...patternrecognitionSection,
      quiz: patternrecognitionQuiz,
      multipleChoice: patternrecognitionMultipleChoice,
    },
  ],
};
