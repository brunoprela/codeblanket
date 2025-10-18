/**
 * Sliding Window Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/sliding-window/introduction';
import { patternsSection } from '../sections/sliding-window/patterns';
import { complexitySection } from '../sections/sliding-window/complexity';
import { templatesSection } from '../sections/sliding-window/templates';
import { advancedSection } from '../sections/sliding-window/advanced';
import { commonpitfallsSection } from '../sections/sliding-window/common-pitfalls';
import { interviewSection } from '../sections/sliding-window/interview';

// Import quizzes
import { introductionQuiz } from '../quizzes/sliding-window/introduction';
import { patternsQuiz } from '../quizzes/sliding-window/patterns';
import { complexityQuiz } from '../quizzes/sliding-window/complexity';
import { templatesQuiz } from '../quizzes/sliding-window/templates';
import { advancedQuiz } from '../quizzes/sliding-window/advanced';
import { commonpitfallsQuiz } from '../quizzes/sliding-window/common-pitfalls';
import { interviewQuiz } from '../quizzes/sliding-window/interview';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/sliding-window/introduction';
import { patternsMultipleChoice } from '../multiple-choice/sliding-window/patterns';
import { complexityMultipleChoice } from '../multiple-choice/sliding-window/complexity';
import { templatesMultipleChoice } from '../multiple-choice/sliding-window/templates';
import { advancedMultipleChoice } from '../multiple-choice/sliding-window/advanced';
import { commonpitfallsMultipleChoice } from '../multiple-choice/sliding-window/common-pitfalls';
import { interviewMultipleChoice } from '../multiple-choice/sliding-window/interview';

export const slidingWindowModule: Module = {
  id: 'sliding-window',
  title: 'Sliding Window',
  description:
    'Master the sliding window technique for optimizing substring, subarray, and sequence problems.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🪟',
  keyTakeaways: [
    'Sliding window optimizes O(N²) brute force to O(N) for contiguous sequence problems',
    'Fixed-size window: maintain window size K by adding right, removing left at position (i-K)',
    'Variable-size window: expand right to add elements, shrink left when condition violated',
    'For maximum/longest: shrink when invalid, update result when valid',
    'For minimum/shortest: shrink while still valid, update result during shrinking',
    'Use hash map to track frequencies, set for uniqueness, deque for maximum/minimum',
    'Time complexity is O(N) because each element visited at most twice (once by each pointer)',
    'Window size formula: right - left + 1 (inclusive of both endpoints)',
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
