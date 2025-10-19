/**
 * Module 4: Probability Theory for ML & Finance
 *
 * Comprehensive coverage of probability theory with applications to
 * machine learning and quantitative finance.
 */

import { Module } from '../../types';

// Import sections
import { probabilityfundamentalsSection } from '../sections/ml-probability-theory/probability-fundamentals';
import { combinatoricscountingSection } from '../sections/ml-probability-theory/combinatorics-counting';
import { conditionalprobabilityindependenceSection } from '../sections/ml-probability-theory/conditional-probability-independence';
import { bayestheoremSection } from '../sections/ml-probability-theory/bayes-theorem';
import { randomvariablesSection } from '../sections/ml-probability-theory/random-variables';
import { commondiscretedistributionsSection } from '../sections/ml-probability-theory/common-discrete-distributions';
import { commoncontinuousdistributionsSection } from '../sections/ml-probability-theory/common-continuous-distributions';
import { normaldistributiondeepdiveSection } from '../sections/ml-probability-theory/normal-distribution-deep-dive';
import { jointmarginaldistributionsSection } from '../sections/ml-probability-theory/joint-marginal-distributions';
import { expectationvarianceSection } from '../sections/ml-probability-theory/expectation-variance';
import { lawnumberscltSection } from '../sections/ml-probability-theory/law-large-numbers-clt';
import { informationtheorybasicsSection } from '../sections/ml-probability-theory/information-theory-basics';
import { stochasticprocessesfinanceSection } from '../sections/ml-probability-theory/stochastic-processes-finance';

// Import quizzes
import { probabilityfundamentalsQuiz } from '../quizzes/ml-probability-theory/probability-fundamentals';
import { combinatoricscountingQuiz } from '../quizzes/ml-probability-theory/combinatorics-counting';
import { conditionalprobabilityindependenceQuiz } from '../quizzes/ml-probability-theory/conditional-probability-independence';
import { bayestheoremQuiz } from '../quizzes/ml-probability-theory/bayes-theorem';
import { randomvariablesQuiz } from '../quizzes/ml-probability-theory/random-variables';
import { commondiscretedistributionsQuiz } from '../quizzes/ml-probability-theory/common-discrete-distributions';
import { commoncontinuousdistributionsQuiz } from '../quizzes/ml-probability-theory/common-continuous-distributions';
import { normaldistributiondeepdiveQuiz } from '../quizzes/ml-probability-theory/normal-distribution-deep-dive';
import { jointmarginaldistributionsQuiz } from '../quizzes/ml-probability-theory/joint-marginal-distributions';
import { expectationvarianceQuiz } from '../quizzes/ml-probability-theory/expectation-variance';
import { lawnumberscltQuiz } from '../quizzes/ml-probability-theory/law-large-numbers-clt';
import { informationtheorybasicsQuiz } from '../quizzes/ml-probability-theory/information-theory-basics';
import { stochasticprocessesfinanceQuiz } from '../quizzes/ml-probability-theory/stochastic-processes-finance';

// Import multiple choice
import { probabilityfundamentalsMultipleChoice } from '../multiple-choice/ml-probability-theory/probability-fundamentals';
import { combinatoricscountingMultipleChoice } from '../multiple-choice/ml-probability-theory/combinatorics-counting';
import { conditionalprobabilityindependenceMultipleChoice } from '../multiple-choice/ml-probability-theory/conditional-probability-independence';
import { bayestheoremMultipleChoice } from '../multiple-choice/ml-probability-theory/bayes-theorem';
import { randomvariablesMultipleChoice } from '../multiple-choice/ml-probability-theory/random-variables';
import { commondiscretedistributionsMultipleChoice } from '../multiple-choice/ml-probability-theory/common-discrete-distributions';
import { commoncontinuousdistributionsMultipleChoice } from '../multiple-choice/ml-probability-theory/common-continuous-distributions';
import { normaldistributiondeepdiveMultipleChoice } from '../multiple-choice/ml-probability-theory/normal-distribution-deep-dive';
import { jointmarginaldistributionsMultipleChoice } from '../multiple-choice/ml-probability-theory/joint-marginal-distributions';
import { expectationvarianceMultipleChoice } from '../multiple-choice/ml-probability-theory/expectation-variance';
import { lawnumberscltMultipleChoice } from '../multiple-choice/ml-probability-theory/law-large-numbers-clt';
import { informationtheorybasicsMultipleChoice } from '../multiple-choice/ml-probability-theory/information-theory-basics';
import { stochasticprocessesfinanceMultipleChoice } from '../multiple-choice/ml-probability-theory/stochastic-processes-finance';

export const mlProbabilityTheoryModule: Module = {
  id: 'ml-probability-theory',
  title: 'Probability Theory for ML & Finance',
  description:
    'Master probability theory from fundamentals through advanced topics. Learn distributions, statistical inference, information theory, and stochastic processes with applications to machine learning and quantitative finance.',
  category: 'Quantitative Programming',
  difficulty: 'intermediate',
  estimatedTime: '16-20 hours',
  prerequisites: ['ml-calculus-fundamentals', 'ml-linear-algebra-foundations'],
  icon: '🎲',
  sections: [
    {
      ...probabilityfundamentalsSection,
      quiz: probabilityfundamentalsQuiz,
      multipleChoice: probabilityfundamentalsMultipleChoice,
    },
    {
      ...combinatoricscountingSection,
      quiz: combinatoricscountingQuiz,
      multipleChoice: combinatoricscountingMultipleChoice,
    },
    {
      ...conditionalprobabilityindependenceSection,
      quiz: conditionalprobabilityindependenceQuiz,
      multipleChoice: conditionalprobabilityindependenceMultipleChoice,
    },
    {
      ...bayestheoremSection,
      quiz: bayestheoremQuiz,
      multipleChoice: bayestheoremMultipleChoice,
    },
    {
      ...randomvariablesSection,
      quiz: randomvariablesQuiz,
      multipleChoice: randomvariablesMultipleChoice,
    },
    {
      ...commondiscretedistributionsSection,
      quiz: commondiscretedistributionsQuiz,
      multipleChoice: commondiscretedistributionsMultipleChoice,
    },
    {
      ...commoncontinuousdistributionsSection,
      quiz: commoncontinuousdistributionsQuiz,
      multipleChoice: commoncontinuousdistributionsMultipleChoice,
    },
    {
      ...normaldistributiondeepdiveSection,
      quiz: normaldistributiondeepdiveQuiz,
      multipleChoice: normaldistributiondeepdiveMultipleChoice,
    },
    {
      ...jointmarginaldistributionsSection,
      quiz: jointmarginaldistributionsQuiz,
      multipleChoice: jointmarginaldistributionsMultipleChoice,
    },
    {
      ...expectationvarianceSection,
      quiz: expectationvarianceQuiz,
      multipleChoice: expectationvarianceMultipleChoice,
    },
    {
      ...lawnumberscltSection,
      quiz: lawnumberscltQuiz,
      multipleChoice: lawnumberscltMultipleChoice,
    },
    {
      ...informationtheorybasicsSection,
      quiz: informationtheorybasicsQuiz,
      multipleChoice: informationtheorybasicsMultipleChoice,
    },
    {
      ...stochasticprocessesfinanceSection,
      quiz: stochasticprocessesfinanceQuiz,
      multipleChoice: stochasticprocessesfinanceMultipleChoice,
    },
  ],
  keyTakeaways: [
    'Probability axioms and fundamental rules for computing probabilities',
    'Combinatorics for counting outcomes in complex scenarios',
    "Conditional probability, independence, and Bayes' theorem",
    'Random variables and probability distributions (discrete and continuous)',
    'Common distributions: Bernoulli, Binomial, Poisson, Uniform, Exponential, Normal',
    'Joint, marginal, and conditional distributions for multiple variables',
    'Expectation, variance, and their properties',
    'Law of Large Numbers and Central Limit Theorem',
    'Information theory: entropy, cross-entropy, KL divergence',
    'Stochastic processes: Random Walk, GBM, mean reversion, Poisson',
    'Applications to ML: loss functions, model evaluation, feature selection',
    'Applications to finance: options pricing, risk management, trading strategies',
  ],
  learningObjectives: [
    'Apply probability axioms to solve complex probability problems',
    'Use combinatorics to count outcomes in sampling and ordering problems',
    "Calculate conditional probabilities and apply Bayes' theorem",
    'Work with random variables and compute expectations and variances',
    'Identify and use appropriate probability distributions for different scenarios',
    'Understand joint distributions, independence, and covariance',
    'Apply Law of Large Numbers and CLT to statistical inference',
    'Use information theory concepts in ML (cross-entropy loss, mutual information)',
    'Model financial time series with stochastic processes',
    'Implement probability-based algorithms in Python (simulation, estimation)',
    'Evaluate ML model performance using probabilistic metrics',
    'Design trading strategies based on stochastic process models',
  ],
};
