/**
 * Calculus Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { limitscontinuitySection } from '../sections/ml-calculus-fundamentals/limits-continuity';
import { derivativesfundamentalsSection } from '../sections/ml-calculus-fundamentals/derivatives-fundamentals';
import { differentiationrulesSection } from '../sections/ml-calculus-fundamentals/differentiation-rules';
import { applicationsderivativesSection } from '../sections/ml-calculus-fundamentals/applications-derivatives';
import { partialderivativesSection } from '../sections/ml-calculus-fundamentals/partial-derivatives';
import { gradientdirectionalderivativesSection } from '../sections/ml-calculus-fundamentals/gradient-directional-derivatives';
import { chainrulemultivariableSection } from '../sections/ml-calculus-fundamentals/chain-rule-multivariable';
import { integrationbasicsSection } from '../sections/ml-calculus-fundamentals/integration-basics';
import { multivariablecalculusSection } from '../sections/ml-calculus-fundamentals/multivariable-calculus';
import { convexoptimizationSection } from '../sections/ml-calculus-fundamentals/convex-optimization';
import { numericaloptimizationSection } from '../sections/ml-calculus-fundamentals/numerical-optimization';
import { stochasticcalculusSection } from '../sections/ml-calculus-fundamentals/stochastic-calculus';

// Import quizzes
import { limitscontinuityQuiz } from '../quizzes/ml-calculus-fundamentals/limits-continuity';
import { derivativesfundamentalsQuiz } from '../quizzes/ml-calculus-fundamentals/derivatives-fundamentals';
import { differentiationrulesQuiz } from '../quizzes/ml-calculus-fundamentals/differentiation-rules';
import { applicationsderivativesQuiz } from '../quizzes/ml-calculus-fundamentals/applications-derivatives';
import { partialderivativesQuiz } from '../quizzes/ml-calculus-fundamentals/partial-derivatives';
import { gradientdirectionalderivativesQuiz } from '../quizzes/ml-calculus-fundamentals/gradient-directional-derivatives';
import { chainrulemultivariableQuiz } from '../quizzes/ml-calculus-fundamentals/chain-rule-multivariable';
import { integrationbasicsQuiz } from '../quizzes/ml-calculus-fundamentals/integration-basics';
import { multivariablecalculusQuiz } from '../quizzes/ml-calculus-fundamentals/multivariable-calculus';
import { convexoptimizationQuiz } from '../quizzes/ml-calculus-fundamentals/convex-optimization';
import { numericaloptimizationQuiz } from '../quizzes/ml-calculus-fundamentals/numerical-optimization';
import { stochasticcalculusQuiz } from '../quizzes/ml-calculus-fundamentals/stochastic-calculus';

// Import multiple choice
import { limitscontinuityMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/limits-continuity';
import { derivativesfundamentalsMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/derivatives-fundamentals';
import { differentiationrulesMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/differentiation-rules';
import { applicationsderivativesMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/applications-derivatives';
import { partialderivativesMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/partial-derivatives';
import { gradientdirectionalderivativesMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/gradient-directional-derivatives';
import { chainrulemultivariableMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/chain-rule-multivariable';
import { integrationbasicsMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/integration-basics';
import { multivariablecalculusMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/multivariable-calculus';
import { convexoptimizationMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/convex-optimization';
import { numericaloptimizationMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/numerical-optimization';
import { stochasticcalculusMultipleChoice } from '../multiple-choice/ml-calculus-fundamentals/stochastic-calculus';

export const mlCalculusFundamentalsModule: Module = {
  id: 'ml-calculus-fundamentals',
  title: 'Calculus Fundamentals',
  description:
    'Master differential and integral calculus essential for understanding machine learning optimization and quantitative finance',
  category: 'undefined',
  difficulty: 'easy',
  estimatedTime: 'undefined',
  prerequisites: ['Module 1: Mathematical Foundations'],
  icon: '📈',
  keyTakeaways: [
    'Derivatives measure instantaneous rates of change, fundamental to optimization',
    'Gradient descent follows negative gradient direction to minimize loss functions',
    'Chain rule enables backpropagation in neural networks',
    'Partial derivatives and gradients extend calculus to multivariable functions',
    'Convex optimization has unique global minima, guaranteeing convergence',
    "Newton's method uses second-order information for quadratic convergence",
    'Stochastic gradient descent trades noise for computational efficiency',
    'Integration computes cumulative effects, expectations, and KL divergence',
    'Hessian matrix characterizes curvature and identifies saddle points',
    'KKT conditions provide optimality certificates for constrained optimization',
    'Adam optimizer combines momentum with adaptive learning rates',
    'Stochastic calculus underpins modern generative models like diffusion models',
  ],
  learningObjectives: [
    'Master differential calculus: limits, derivatives, and differentiation rules',
    'Understand gradient-based optimization for machine learning',
    'Apply chain rule to compute gradients in neural networks',
    'Use partial derivatives and gradients for multivariable optimization',
    'Identify and leverage convex functions in optimization problems',
    'Implement numerical optimization algorithms (GD, momentum, Adam, Newton)',
    'Understand integration for computing expectations and information measures',
    'Analyze critical points using Jacobian and Hessian matrices',
    'Apply KKT conditions to constrained optimization problems',
    'Understand stochastic differential equations and their ML applications',
    'Implement calculus-based algorithms in Python (NumPy, SymPy, SciPy)',
    'Connect calculus theory to practical deep learning optimization',
  ],
  sections: [
    {
      ...limitscontinuitySection,
      quiz: limitscontinuityQuiz,
      multipleChoice: limitscontinuityMultipleChoice,
    },
    {
      ...derivativesfundamentalsSection,
      quiz: derivativesfundamentalsQuiz,
      multipleChoice: derivativesfundamentalsMultipleChoice,
    },
    {
      ...differentiationrulesSection,
      quiz: differentiationrulesQuiz,
      multipleChoice: differentiationrulesMultipleChoice,
    },
    {
      ...applicationsderivativesSection,
      quiz: applicationsderivativesQuiz,
      multipleChoice: applicationsderivativesMultipleChoice,
    },
    {
      ...partialderivativesSection,
      quiz: partialderivativesQuiz,
      multipleChoice: partialderivativesMultipleChoice,
    },
    {
      ...gradientdirectionalderivativesSection,
      quiz: gradientdirectionalderivativesQuiz,
      multipleChoice: gradientdirectionalderivativesMultipleChoice,
    },
    {
      ...chainrulemultivariableSection,
      quiz: chainrulemultivariableQuiz,
      multipleChoice: chainrulemultivariableMultipleChoice,
    },
    {
      ...integrationbasicsSection,
      quiz: integrationbasicsQuiz,
      multipleChoice: integrationbasicsMultipleChoice,
    },
    {
      ...multivariablecalculusSection,
      quiz: multivariablecalculusQuiz,
      multipleChoice: multivariablecalculusMultipleChoice,
    },
    {
      ...convexoptimizationSection,
      quiz: convexoptimizationQuiz,
      multipleChoice: convexoptimizationMultipleChoice,
    },
    {
      ...numericaloptimizationSection,
      quiz: numericaloptimizationQuiz,
      multipleChoice: numericaloptimizationMultipleChoice,
    },
    {
      ...stochasticcalculusSection,
      quiz: stochasticcalculusQuiz,
      multipleChoice: stochasticcalculusMultipleChoice,
    },
  ],
};
