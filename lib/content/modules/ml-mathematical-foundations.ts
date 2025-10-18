/**
 * Mathematical Foundations Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { numbersystemsSection } from '../sections/ml-mathematical-foundations/number-systems';
import { algebraicexpressionsSection } from '../sections/ml-mathematical-foundations/algebraic-expressions';
import { functionsrelationsSection } from '../sections/ml-mathematical-foundations/functions-relations';
import { exponentslogarithmsSection } from '../sections/ml-mathematical-foundations/exponents-logarithms';
import { sequencesseriesSection } from '../sections/ml-mathematical-foundations/sequences-series';
import { settheorylogicSection } from '../sections/ml-mathematical-foundations/set-theory-logic';
import { combinatoricsbasicsSection } from '../sections/ml-mathematical-foundations/combinatorics-basics';
import { notationproofSection } from '../sections/ml-mathematical-foundations/notation-proof';

// Import quizzes
import { numbersystemsQuiz } from '../quizzes/ml-mathematical-foundations/number-systems';
import { algebraicexpressionsQuiz } from '../quizzes/ml-mathematical-foundations/algebraic-expressions';
import { functionsrelationsQuiz } from '../quizzes/ml-mathematical-foundations/functions-relations';
import { exponentslogarithmsQuiz } from '../quizzes/ml-mathematical-foundations/exponents-logarithms';
import { sequencesseriesQuiz } from '../quizzes/ml-mathematical-foundations/sequences-series';
import { settheorylogicQuiz } from '../quizzes/ml-mathematical-foundations/set-theory-logic';
import { combinatoricsbasicsQuiz } from '../quizzes/ml-mathematical-foundations/combinatorics-basics';
import { notationproofQuiz } from '../quizzes/ml-mathematical-foundations/notation-proof';

// Import multiple choice
import { numbersystemsMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/number-systems';
import { algebraicexpressionsMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/algebraic-expressions';
import { functionsrelationsMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/functions-relations';
import { exponentslogarithmsMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/exponents-logarithms';
import { sequencesseriesMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/sequences-series';
import { settheorylogicMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/set-theory-logic';
import { combinatoricsbasicsMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/combinatorics-basics';
import { notationproofMultipleChoice } from '../multiple-choice/ml-mathematical-foundations/notation-proof';

export const mlMathematicalFoundationsModule: Module = {
  id: 'ml-mathematical-foundations',
  title: 'Mathematical Foundations',
  description:
    'Master elementary mathematics, algebra, and functions essential for machine learning and AI',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: ['Basic arithmetic', 'Basic Python programming'],
  icon: '🔢',
  keyTakeaways: [
    'Different number systems (integers, rationals, reals, complex) have different properties and uses',
    'Floating-point arithmetic has precision limitations that affect ML algorithms',
    'Scientific notation and orders of magnitude are crucial for understanding scale in ML',
    'Properties like commutative, associative, and distributive guide mathematical operations',
    'Numerical stability techniques (log space, epsilon additions) prevent computational errors',
    'Understanding absolute values and inequalities is essential for loss functions and distances',
    'Complex numbers are fundamental in signal processing, Fourier transforms, and quantum computing',
    'Choosing appropriate data types (float32 vs float64) impacts memory, speed, and precision',
  ],
  learningObjectives: [
    'Understand different number systems and their computational representations',
    'Recognize floating-point precision limitations and their implications',
    'Apply numerical stability techniques in machine learning code',
    'Use scientific notation to reason about scales in ML models',
    'Choose appropriate data types for different ML tasks',
    'Implement numerically stable algorithms for real-world applications',
    'Understand how number theory impacts algorithm design and debugging',
  ],
  sections: [
    {
      ...numbersystemsSection,
      quiz: numbersystemsQuiz,
      multipleChoice: numbersystemsMultipleChoice,
    },
    {
      ...algebraicexpressionsSection,
      quiz: algebraicexpressionsQuiz,
      multipleChoice: algebraicexpressionsMultipleChoice,
    },
    {
      ...functionsrelationsSection,
      quiz: functionsrelationsQuiz,
      multipleChoice: functionsrelationsMultipleChoice,
    },
    {
      ...exponentslogarithmsSection,
      quiz: exponentslogarithmsQuiz,
      multipleChoice: exponentslogarithmsMultipleChoice,
    },
    {
      ...sequencesseriesSection,
      quiz: sequencesseriesQuiz,
      multipleChoice: sequencesseriesMultipleChoice,
    },
    {
      ...settheorylogicSection,
      quiz: settheorylogicQuiz,
      multipleChoice: settheorylogicMultipleChoice,
    },
    {
      ...combinatoricsbasicsSection,
      quiz: combinatoricsbasicsQuiz,
      multipleChoice: combinatoricsbasicsMultipleChoice,
    },
    {
      ...notationproofSection,
      quiz: notationproofQuiz,
      multipleChoice: notationproofMultipleChoice,
    },
  ],
};
