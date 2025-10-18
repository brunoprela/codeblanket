/**
 * Linked List Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/linked-list/introduction';
import { patternsSection } from '../sections/linked-list/patterns';
import { complexitySection } from '../sections/linked-list/complexity';
import { templatesSection } from '../sections/linked-list/templates';
import { advancedSection } from '../sections/linked-list/advanced';
import { commonpitfallsSection } from '../sections/linked-list/common-pitfalls';
import { interviewSection } from '../sections/linked-list/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/linked-list/introduction';
import { patternsQuiz } from '../quizzes/linked-list/patterns';
import { complexityQuiz } from '../quizzes/linked-list/complexity';
import { templatesQuiz } from '../quizzes/linked-list/templates';
import { advancedQuiz } from '../quizzes/linked-list/advanced';
import { commonpitfallsQuiz } from '../quizzes/linked-list/common-pitfalls';
import { interviewQuiz } from '../quizzes/linked-list/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/linked-list/introduction';
import { patternsMultipleChoice } from '../multiple-choice/linked-list/patterns';
import { complexityMultipleChoice } from '../multiple-choice/linked-list/complexity';
import { templatesMultipleChoice } from '../multiple-choice/linked-list/templates';
import { advancedMultipleChoice } from '../multiple-choice/linked-list/advanced';
import { commonpitfallsMultipleChoice } from '../multiple-choice/linked-list/common-pitfalls';
import { interviewMultipleChoice } from '../multiple-choice/linked-list/interview';

export const linkedListModule: Module = {
  id: 'linked-list',
  title: 'Linked List',
  description:
    'Master linked list manipulation, pointer techniques, and common patterns for interview success.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔗',
  keyTakeaways: [
    'Linked lists provide O(1) insertion/deletion at known positions but O(N) random access',
    'Two pointers (fast & slow) solve cycle detection, middle finding, and kth from end problems',
    'Dummy nodes simplify edge cases by providing a stable reference before the head',
    'Reversal pattern: save next, reverse current link, move pointers forward',
    'Always save the next reference before modifying curr.next to avoid losing the list',
    "Floyd's cycle detection: fast and slow pointers meet inside cycle if one exists",
    'Runner technique: move first pointer k steps ahead to find kth from end',
    'Consider iterative (O(1) space) vs recursive (O(N) space) based on requirements',
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
      ...commonpitfallsSection,
      quiz: commonpitfallsQuiz,
      multipleChoice: commonpitfallsMultipleChoice,
    },
    {
      ...interviewSection,
      quiz: interviewQuiz,
      multipleChoice: interviewMultipleChoice,
    },
  ],
};
