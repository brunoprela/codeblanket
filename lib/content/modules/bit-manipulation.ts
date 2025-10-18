/**
 * Bit Manipulation Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/bit-manipulation/introduction';
import { operatorsSection } from '../sections/bit-manipulation/operators';
import { commonpatternsSection } from '../sections/bit-manipulation/common-patterns';
import { xorpropertiesSection } from '../sections/bit-manipulation/xor-properties';
import { advancedtechniquesSection } from '../sections/bit-manipulation/advanced-techniques';
import { complexitySection } from '../sections/bit-manipulation/complexity';
import { interviewstrategySection } from '../sections/bit-manipulation/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/bit-manipulation/introduction';
import { operatorsQuiz } from '../quizzes/bit-manipulation/operators';
import { commonpatternsQuiz } from '../quizzes/bit-manipulation/common-patterns';
import { xorpropertiesQuiz } from '../quizzes/bit-manipulation/xor-properties';
import { advancedtechniquesQuiz } from '../quizzes/bit-manipulation/advanced-techniques';
import { complexityQuiz } from '../quizzes/bit-manipulation/complexity';
import { interviewstrategyQuiz } from '../quizzes/bit-manipulation/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/bit-manipulation/introduction';
import { operatorsMultipleChoice } from '../multiple-choice/bit-manipulation/operators';
import { commonpatternsMultipleChoice } from '../multiple-choice/bit-manipulation/common-patterns';
import { xorpropertiesMultipleChoice } from '../multiple-choice/bit-manipulation/xor-properties';
import { advancedtechniquesMultipleChoice } from '../multiple-choice/bit-manipulation/advanced-techniques';
import { complexityMultipleChoice } from '../multiple-choice/bit-manipulation/complexity';
import { interviewstrategyMultipleChoice } from '../multiple-choice/bit-manipulation/interview-strategy';

export const bitManipulationModule: Module = {
  id: 'bit-manipulation',
  title: 'Bit Manipulation',
  description:
    'Master bitwise operations and clever bit tricks for ultra-efficient problem solving.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '⚡',
  keyTakeaways: [
    'XOR is your best friend: a ^ a = 0, a ^ 0 = a, perfect for finding unique elements',
    'n & (n-1) removes the rightmost set bit - used for power of 2 check and counting bits',
    'Left shift (<<) multiplies by 2^n, right shift (>>) divides by 2^n',
    'Bit manipulation provides O(1) space solutions when others need O(n)',
    'Brian Kernighan algorithm efficiently counts set bits in O(k) time',
    'Use bit masking to represent sets and combinations efficiently',
    'Always draw binary representations to visualize the problem',
    'Bit manipulation is common in interviews - master XOR patterns especially',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...operatorsSection,
      quiz: operatorsQuiz,
      multipleChoice: operatorsMultipleChoice,
    },
    {
      ...commonpatternsSection,
      quiz: commonpatternsQuiz,
      multipleChoice: commonpatternsMultipleChoice,
    },
    {
      ...xorpropertiesSection,
      quiz: xorpropertiesQuiz,
      multipleChoice: xorpropertiesMultipleChoice,
    },
    {
      ...advancedtechniquesSection,
      quiz: advancedtechniquesQuiz,
      multipleChoice: advancedtechniquesMultipleChoice,
    },
    {
      ...complexitySection,
      quiz: complexityQuiz,
      multipleChoice: complexityMultipleChoice,
    },
    {
      ...interviewstrategySection,
      quiz: interviewstrategyQuiz,
      multipleChoice: interviewstrategyMultipleChoice,
    },
  ],
};
