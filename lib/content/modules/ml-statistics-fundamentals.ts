/**
 * Statistics Fundamentals Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { descriptivestatisticsSection } from '../sections/ml-statistics-fundamentals/descriptive-statistics';
import { datavisualizationSection } from '../sections/ml-statistics-fundamentals/data-visualization';
import { statisticalinferenceSection } from '../sections/ml-statistics-fundamentals/statistical-inference';
import { hypothesistestingSection } from '../sections/ml-statistics-fundamentals/hypothesis-testing';
import { commonstatisticaltestsSection } from '../sections/ml-statistics-fundamentals/common-statistical-tests';
import { correlationassociationSection } from '../sections/ml-statistics-fundamentals/correlation-association';
import { simplelinearregressionSection } from '../sections/ml-statistics-fundamentals/simple-linear-regression';
import { multiplelinearregressionSection } from '../sections/ml-statistics-fundamentals/multiple-linear-regression';
import { regressiondiagnosticsSection } from '../sections/ml-statistics-fundamentals/regression-diagnostics';
import { maximumlikelihoodestimationSection } from '../sections/ml-statistics-fundamentals/maximum-likelihood-estimation';
import { bayesianstatisticsSection } from '../sections/ml-statistics-fundamentals/bayesian-statistics';
import { experimentaldesignSection } from '../sections/ml-statistics-fundamentals/experimental-design';
import { timeseriesstatisticsSection } from '../sections/ml-statistics-fundamentals/time-series-statistics';
import { robuststatisticsSection } from '../sections/ml-statistics-fundamentals/robust-statistics';

// Import quizzes
import { descriptivestatisticsQuiz } from '../quizzes/ml-statistics-fundamentals/descriptive-statistics';
import { datavisualizationQuiz } from '../quizzes/ml-statistics-fundamentals/data-visualization';
import { statisticalinferenceQuiz } from '../quizzes/ml-statistics-fundamentals/statistical-inference';
import { hypothesistestingQuiz } from '../quizzes/ml-statistics-fundamentals/hypothesis-testing';
import { commonstatisticaltestsQuiz } from '../quizzes/ml-statistics-fundamentals/common-statistical-tests';
import { correlationassociationQuiz } from '../quizzes/ml-statistics-fundamentals/correlation-association';
import { simplelinearregressionQuiz } from '../quizzes/ml-statistics-fundamentals/simple-linear-regression';
import { multiplelinearregressionQuiz } from '../quizzes/ml-statistics-fundamentals/multiple-linear-regression';
import { regressiondiagnosticsQuiz } from '../quizzes/ml-statistics-fundamentals/regression-diagnostics';
import { maximumlikelihoodestimationQuiz } from '../quizzes/ml-statistics-fundamentals/maximum-likelihood-estimation';
import { bayesianstatisticsQuiz } from '../quizzes/ml-statistics-fundamentals/bayesian-statistics';
import { experimentaldesignQuiz } from '../quizzes/ml-statistics-fundamentals/experimental-design';
import { timeseriesstatisticsQuiz } from '../quizzes/ml-statistics-fundamentals/time-series-statistics';
import { robuststatisticsQuiz } from '../quizzes/ml-statistics-fundamentals/robust-statistics';

// Import multiple choice
import { descriptivestatisticsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/descriptive-statistics';
import { datavisualizationMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/data-visualization';
import { statisticalinferenceMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/statistical-inference';
import { hypothesistestingMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/hypothesis-testing';
import { commonstatisticaltestsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/common-statistical-tests';
import { correlationassociationMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/correlation-association';
import { simplelinearregressionMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/simple-linear-regression';
import { multiplelinearregressionMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/multiple-linear-regression';
import { regressiondiagnosticsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/regression-diagnostics';
import { maximumlikelihoodestimationMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/maximum-likelihood-estimation';
import { bayesianstatisticsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/bayesian-statistics';
import { experimentaldesignMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/experimental-design';
import { timeseriesstatisticsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/time-series-statistics';
import { robuststatisticsMultipleChoice } from '../multiple-choice/ml-statistics-fundamentals/robust-statistics';

export const mlStatisticsFundamentalsModule: Module = {
  id: 'ml-statistics-fundamentals',
  title: 'Statistics Fundamentals',
  description:
    'Master statistical inference, hypothesis testing, and regression analysis essential for data science and machine learning',
  category: 'undefined',
  difficulty: 'easy',
  estimatedTime: 'undefined',
  prerequisites: ['Module 1: Mathematical Foundations'],
  icon: '📊',
  keyTakeaways: [
    'Descriptive statistics summarize data: mean, median, variance, skewness, kurtosis',
    'Visualization is critical for exploratory data analysis and understanding distributions',
    'Confidence intervals quantify uncertainty in parameter estimates',
    'Hypothesis testing provides framework for statistical decision-making',
    'P-values measure evidence against null hypothesis, not probability of truth',
    'Type I (false positive) and Type II (false negative) errors have different consequences',
    'Correlation measures association but does not imply causation',
    'Linear regression estimates relationships between variables via least squares',
    'R² measures proportion of variance explained by the model',
    'Multicollinearity inflates standard errors; detect with VIF',
    "Regression diagnostics check assumptions: residual plots, Q-Q plots, Cook's distance",
    'Maximum Likelihood Estimation connects to neural network training',
    'Bayesian statistics incorporates prior knowledge and provides credible intervals',
    'Randomization in experiments enables causal inference',
    'Time series data violates independence assumption; test for stationarity',
    'Robust methods handle outliers and heavy-tailed distributions',
  ],
  learningObjectives: [
    'Calculate and interpret descriptive statistics for data summarization',
    'Create effective visualizations for exploratory data analysis',
    'Construct and interpret confidence intervals for population parameters',
    'Conduct hypothesis tests and interpret p-values correctly',
    'Select appropriate statistical tests (t-test, ANOVA, chi-square) for different scenarios',
    'Distinguish between correlation and causation in data relationships',
    'Fit and interpret simple and multiple linear regression models',
    'Diagnose regression assumptions using residual analysis and statistical tests',
    'Apply maximum likelihood estimation to statistical models',
    'Understand Bayesian inference and compute posterior distributions',
    'Design A/B tests with proper randomization and sample size determination',
    'Test time series for stationarity and Granger causality',
    'Apply robust statistical methods to handle outliers and violations of assumptions',
    'Connect statistical theory to machine learning practice',
  ],
  sections: [
    {
      ...descriptivestatisticsSection,
      quiz: descriptivestatisticsQuiz,
      multipleChoice: descriptivestatisticsMultipleChoice,
    },
    {
      ...datavisualizationSection,
      quiz: datavisualizationQuiz,
      multipleChoice: datavisualizationMultipleChoice,
    },
    {
      ...statisticalinferenceSection,
      quiz: statisticalinferenceQuiz,
      multipleChoice: statisticalinferenceMultipleChoice,
    },
    {
      ...hypothesistestingSection,
      quiz: hypothesistestingQuiz,
      multipleChoice: hypothesistestingMultipleChoice,
    },
    {
      ...commonstatisticaltestsSection,
      quiz: commonstatisticaltestsQuiz,
      multipleChoice: commonstatisticaltestsMultipleChoice,
    },
    {
      ...correlationassociationSection,
      quiz: correlationassociationQuiz,
      multipleChoice: correlationassociationMultipleChoice,
    },
    {
      ...simplelinearregressionSection,
      quiz: simplelinearregressionQuiz,
      multipleChoice: simplelinearregressionMultipleChoice,
    },
    {
      ...multiplelinearregressionSection,
      quiz: multiplelinearregressionQuiz,
      multipleChoice: multiplelinearregressionMultipleChoice,
    },
    {
      ...regressiondiagnosticsSection,
      quiz: regressiondiagnosticsQuiz,
      multipleChoice: regressiondiagnosticsMultipleChoice,
    },
    {
      ...maximumlikelihoodestimationSection,
      quiz: maximumlikelihoodestimationQuiz,
      multipleChoice: maximumlikelihoodestimationMultipleChoice,
    },
    {
      ...bayesianstatisticsSection,
      quiz: bayesianstatisticsQuiz,
      multipleChoice: bayesianstatisticsMultipleChoice,
    },
    {
      ...experimentaldesignSection,
      quiz: experimentaldesignQuiz,
      multipleChoice: experimentaldesignMultipleChoice,
    },
    {
      ...timeseriesstatisticsSection,
      quiz: timeseriesstatisticsQuiz,
      multipleChoice: timeseriesstatisticsMultipleChoice,
    },
    {
      ...robuststatisticsSection,
      quiz: robuststatisticsQuiz,
      multipleChoice: robuststatisticsMultipleChoice,
    },
  ],
};
