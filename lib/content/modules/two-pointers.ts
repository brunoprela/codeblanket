/**
 * Two Pointers Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/two-pointers/introduction';
import { patternsSection } from '../sections/two-pointers/patterns';
import { algorithmSection } from '../sections/two-pointers/algorithm';
import { complexitySection } from '../sections/two-pointers/complexity';
import { templatesSection } from '../sections/two-pointers/templates';
import { advancedSection } from '../sections/two-pointers/advanced';
import { strategySection } from '../sections/two-pointers/strategy';
import { whennottouseSection } from '../sections/two-pointers/when-not-to-use';

// Import quizzes
import { introductionQuiz } from '../quizzes/two-pointers/introduction';
import { patternsQuiz } from '../quizzes/two-pointers/patterns';
import { algorithmQuiz } from '../quizzes/two-pointers/algorithm';
import { complexityQuiz } from '../quizzes/two-pointers/complexity';
import { templatesQuiz } from '../quizzes/two-pointers/templates';
import { advancedQuiz } from '../quizzes/two-pointers/advanced';
import { strategyQuiz } from '../quizzes/two-pointers/strategy';
import { whennottouseQuiz } from '../quizzes/two-pointers/when-not-to-use';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/two-pointers/introduction';
import { patternsMultipleChoice } from '../multiple-choice/two-pointers/patterns';
import { algorithmMultipleChoice } from '../multiple-choice/two-pointers/algorithm';
import { complexityMultipleChoice } from '../multiple-choice/two-pointers/complexity';
import { templatesMultipleChoice } from '../multiple-choice/two-pointers/templates';
import { advancedMultipleChoice } from '../multiple-choice/two-pointers/advanced';
import { strategyMultipleChoice } from '../multiple-choice/two-pointers/strategy';
import { whennottouseMultipleChoice } from '../multiple-choice/two-pointers/when-not-to-use';

export const twoPointersModule: Module = {
  id: 'two-pointers',
  title: 'Two Pointers',
  description:
    'Learn the two-pointer technique for efficiently solving array and string problems.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '👉👈',
  keyTakeaways: [
    'Two pointers reduces O(n²) nested loops to O(n) by strategically moving pointers',
    'Three main patterns: opposite direction (converging), same direction (fast/slow), sliding window',
    'Opposite direction works great for sorted arrays and pair-finding problems',
    'Same direction pattern perfect for in-place modifications and partitioning',
    'Sliding window excels at subarray/substring problems with constraints',
    'Always O(1) space complexity - processes data in-place without extra structures',
    'Key skill: deciding which pointer to move based on current conditions',
    'Can extend to three or more pointers for problems like 3Sum and 4Sum',
    'Common mistakes: wrong initialization, infinite loops, off-by-one errors',
    'Recognition: look for pairs, in-place operations, or window-based problems',
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
      ...algorithmSection,
      quiz: algorithmQuiz,
      multipleChoice: algorithmMultipleChoice,
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
      ...advancedSection,
      quiz: advancedQuiz,
      multipleChoice: advancedMultipleChoice,
    },
    {
      ...strategySection,
      quiz: strategyQuiz,
      multipleChoice: strategyMultipleChoice,
    },
    {
      ...whennottouseSection,
      quiz: whennottouseQuiz,
      multipleChoice: whennottouseMultipleChoice,
    },
  ],
};
