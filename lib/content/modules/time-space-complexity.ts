/**
 * Time & Space Complexity Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/time-space-complexity/introduction';
import { bigonotationSection } from '../sections/time-space-complexity/big-o-notation';
import { spacecomplexitySection } from '../sections/time-space-complexity/space-complexity';
import { analyzingcodeSection } from '../sections/time-space-complexity/analyzing-code';
import { bestaverageworstSection } from '../sections/time-space-complexity/best-average-worst';
import { optimizationSection } from '../sections/time-space-complexity/optimization';

// Import quizzes
import { introductionQuiz } from '../quizzes/time-space-complexity/introduction';
import { bigonotationQuiz } from '../quizzes/time-space-complexity/big-o-notation';
import { spacecomplexityQuiz } from '../quizzes/time-space-complexity/space-complexity';
import { analyzingcodeQuiz } from '../quizzes/time-space-complexity/analyzing-code';
import { bestaverageworstQuiz } from '../quizzes/time-space-complexity/best-average-worst';
import { optimizationQuiz } from '../quizzes/time-space-complexity/optimization';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/time-space-complexity/introduction';
import { bigonotationMultipleChoice } from '../multiple-choice/time-space-complexity/big-o-notation';
import { spacecomplexityMultipleChoice } from '../multiple-choice/time-space-complexity/space-complexity';
import { analyzingcodeMultipleChoice } from '../multiple-choice/time-space-complexity/analyzing-code';
import { bestaverageworstMultipleChoice } from '../multiple-choice/time-space-complexity/best-average-worst';
import { optimizationMultipleChoice } from '../multiple-choice/time-space-complexity/optimization';

export const timeSpaceComplexityModule: Module = {
  id: 'time-space-complexity',
  title: 'Time & Space Complexity',
  description:
    'Master the art of analyzing algorithm efficiency and understanding Big O notation for both time and space.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '⏱️',
  keyTakeaways: [
    'Big O measures how algorithms scale with input size, focusing on growth rate not exact counts',
    'Common complexities: O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)',
    'Space complexity measures auxiliary memory - includes data structures and recursive call stack',
    'Time-space tradeoffs are common - memoization trades space for time',
    'Analyze worst case by default for safety guarantees',
    'Nested loops often indicate quadratic complexity - look for optimizations',
    'Hash tables can turn O(n²) algorithms into O(n) with O(n) space',
    'Always consider both time and space complexity in your analysis',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...bigonotationSection,
      quiz: bigonotationQuiz,
      multipleChoice: bigonotationMultipleChoice,
    },
    {
      ...spacecomplexitySection,
      quiz: spacecomplexityQuiz,
      multipleChoice: spacecomplexityMultipleChoice,
    },
    {
      ...analyzingcodeSection,
      quiz: analyzingcodeQuiz,
      multipleChoice: analyzingcodeMultipleChoice,
    },
    {
      ...bestaverageworstSection,
      quiz: bestaverageworstQuiz,
      multipleChoice: bestaverageworstMultipleChoice,
    },
    {
      ...optimizationSection,
      quiz: optimizationQuiz,
      multipleChoice: optimizationMultipleChoice,
    },
  ],
};
