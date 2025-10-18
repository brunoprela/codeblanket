/**
 * Stack Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/stack/introduction';
import { patternsSection } from '../sections/stack/patterns';
import { complexitySection } from '../sections/stack/complexity';
import { templatesSection } from '../sections/stack/templates';
import { advancedSection } from '../sections/stack/advanced';
import { commonpitfallsSection } from '../sections/stack/common-pitfalls';
import { interviewSection } from '../sections/stack/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/stack/introduction';
import { patternsQuiz } from '../quizzes/stack/patterns';
import { complexityQuiz } from '../quizzes/stack/complexity';
import { templatesQuiz } from '../quizzes/stack/templates';
import { advancedQuiz } from '../quizzes/stack/advanced';
import { commonpitfallsQuiz } from '../quizzes/stack/common-pitfalls';
import { interviewQuiz } from '../quizzes/stack/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/stack/introduction';
import { patternsMultipleChoice } from '../multiple-choice/stack/patterns';
import { complexityMultipleChoice } from '../multiple-choice/stack/complexity';
import { templatesMultipleChoice } from '../multiple-choice/stack/templates';
import { advancedMultipleChoice } from '../multiple-choice/stack/advanced';
import { commonpitfallsMultipleChoice } from '../multiple-choice/stack/common-pitfalls';
import { interviewMultipleChoice } from '../multiple-choice/stack/interview';

export const stackModule: Module = {
  id: 'stack',
  title: 'Stack',
  description:
    'Master the Last-In-First-Out (LIFO) data structure for parsing, backtracking, and monotonic patterns.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📚',
  keyTakeaways: [
    'Stacks follow LIFO (Last-In-First-Out) principle with O(1) push/pop operations',
    'Use stacks for matching pairs (parentheses validation) by pushing opening brackets and popping on closing',
    'Monotonic stacks maintain increasing/decreasing order to solve "next greater/smaller" problems in O(N)',
    'MinStack pattern: maintain parallel stack to track running minimum in O(1) per operation',
    'Stack-based DFS: replace recursion with explicit stack to avoid stack overflow',
    'Expression evaluation: use two stacks (operands and operators) with precedence rules',
    'Recognition: look for "recent", "nested", "backtrack", "undo", or "next greater/smaller" keywords',
    'Common pitfall: always check if stack is empty before popping',
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
