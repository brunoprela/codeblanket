/**
 * Math & Geometry Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { introductionSection } from '../sections/math-geometry/introduction';
import { numbertheorySection } from '../sections/math-geometry/number-theory';
import { matrixoperationsSection } from '../sections/math-geometry/matrix-operations';
import { geometrySection } from '../sections/math-geometry/geometry';
import { combinatoricsSection } from '../sections/math-geometry/combinatorics';
import { commonpatternsSection } from '../sections/math-geometry/common-patterns';
import { complexitySection } from '../sections/math-geometry/complexity';
import { interviewstrategySection } from '../sections/math-geometry/interview-strategy';

// Import quizzes
import { introductionQuiz } from '../quizzes/math-geometry/introduction';
import { numbertheoryQuiz } from '../quizzes/math-geometry/number-theory';
import { matrixoperationsQuiz } from '../quizzes/math-geometry/matrix-operations';
import { geometryQuiz } from '../quizzes/math-geometry/geometry';
import { combinatoricsQuiz } from '../quizzes/math-geometry/combinatorics';
import { commonpatternsQuiz } from '../quizzes/math-geometry/common-patterns';
import { complexityQuiz } from '../quizzes/math-geometry/complexity';
import { interviewstrategyQuiz } from '../quizzes/math-geometry/interview-strategy';

// Import multiple choice
import { introductionMultipleChoice } from '../multiple-choice/math-geometry/introduction';
import { numbertheoryMultipleChoice } from '../multiple-choice/math-geometry/number-theory';
import { matrixoperationsMultipleChoice } from '../multiple-choice/math-geometry/matrix-operations';
import { geometryMultipleChoice } from '../multiple-choice/math-geometry/geometry';
import { combinatoricsMultipleChoice } from '../multiple-choice/math-geometry/combinatorics';
import { commonpatternsMultipleChoice } from '../multiple-choice/math-geometry/common-patterns';
import { complexityMultipleChoice } from '../multiple-choice/math-geometry/complexity';
import { interviewstrategyMultipleChoice } from '../multiple-choice/math-geometry/interview-strategy';

export const mathGeometryModule: Module = {
  id: 'math-geometry',
  title: 'Math & Geometry',
  description:
    'Master mathematical algorithms and geometric problem-solving techniques.',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '📐',
  keyTakeaways: [
    'Matrix rotation in-place: transpose + reverse rows (90° clockwise)',
    'Fast exponentiation reduces O(n) to O(log n) using binary representation',
    'Check divisors only up to sqrt(n) for primality and factorization',
    'Use GCD for simplifying fractions and finding patterns',
    'Spiral/diagonal matrix traversal: track boundaries carefully',
    'Most math problems optimize to O(1) space with clever techniques',
    'Draw diagrams and work through small examples to find patterns',
    'Modular arithmetic prevents overflow and keeps numbers bounded',
  ],
  learningObjectives: [],
  sections: [
    {
      ...introductionSection,
      quiz: introductionQuiz,
      multipleChoice: introductionMultipleChoice,
    },
    {
      ...numbertheorySection,
      quiz: numbertheoryQuiz,
      multipleChoice: numbertheoryMultipleChoice,
    },
    {
      ...matrixoperationsSection,
      quiz: matrixoperationsQuiz,
      multipleChoice: matrixoperationsMultipleChoice,
    },
    {
      ...geometrySection,
      quiz: geometryQuiz,
      multipleChoice: geometryMultipleChoice,
    },
    {
      ...combinatoricsSection,
      quiz: combinatoricsQuiz,
      multipleChoice: combinatoricsMultipleChoice,
    },
    {
      ...commonpatternsSection,
      quiz: commonpatternsQuiz,
      multipleChoice: commonpatternsMultipleChoice,
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
