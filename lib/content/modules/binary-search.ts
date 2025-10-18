/**
 * Binary Search Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/binary-search/introduction';
import { algorithmSection } from '../sections/binary-search/algorithm';
import { complexitySection } from '../sections/binary-search/complexity';
import { templatesSection } from '../sections/binary-search/templates';
import { commonmistakesSection } from '../sections/binary-search/common-mistakes';
import { variationsSection } from '../sections/binary-search/variations';
import { problemsolvingSection } from '../sections/binary-search/problem-solving';

// Import quizzes
import { introductionQuiz } from '../quizzes/binary-search/introduction';
import { algorithmQuiz } from '../quizzes/binary-search/algorithm';
import { complexityQuiz } from '../quizzes/binary-search/complexity';
import { templatesQuiz } from '../quizzes/binary-search/templates';
import { commonmistakesQuiz } from '../quizzes/binary-search/common-mistakes';
import { variationsQuiz } from '../quizzes/binary-search/variations';
import { problemsolvingQuiz } from '../quizzes/binary-search/problem-solving';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/binary-search/introduction';
import { algorithmMultipleChoice } from '../multiple-choice/binary-search/algorithm';
import { complexityMultipleChoice } from '../multiple-choice/binary-search/complexity';
import { templatesMultipleChoice } from '../multiple-choice/binary-search/templates';
import { commonmistakesMultipleChoice } from '../multiple-choice/binary-search/common-mistakes';
import { variationsMultipleChoice } from '../multiple-choice/binary-search/variations';
import { problemsolvingMultipleChoice } from '../multiple-choice/binary-search/problem-solving';

export const binarySearchModule: Module = {
  id: 'binary-search',
  title: 'Binary Search',
  description:
    'Master the art of efficiently searching in sorted arrays using the divide-and-conquer approach.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔍',
  keyTakeaways: [
    'Binary search reduces O(n) search to O(log n) by eliminating half the search space each iteration',
    'Only works on sorted (or monotonic) data - this is a strict requirement',
    'Use "left + (right - left) // 2" to avoid integer overflow',
    'Three main templates: exact match, find first, find last - master all three',
    'Common mistakes: wrong loop condition (use <=), off-by-one errors (use mid±1)',
    'Can be applied to many problems beyond simple array search - look for monotonic properties',
    'Time complexity: O(log n), Space: O(1) iterative, O(log n) recursive',
    'Always test edge cases: empty array, single element, duplicates, boundaries',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
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
      ...commonmistakesSection,
      quiz: commonmistakesQuiz,
      multipleChoice: commonmistakesMultipleChoice,
    },
    {
      ...variationsSection,
      quiz: variationsQuiz,
      multipleChoice: variationsMultipleChoice,
    },
    {
      ...problemsolvingSection,
      quiz: problemsolvingQuiz,
      multipleChoice: problemsolvingMultipleChoice,
    },
  ],
};
