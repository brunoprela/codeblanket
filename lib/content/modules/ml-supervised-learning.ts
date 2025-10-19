/**
 * Supervised Learning Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { mlfundamentalsSection } from '../sections/ml-supervised-learning/ml-fundamentals';
import { linearregressionSection } from '../sections/ml-supervised-learning/linear-regression';
import { polynomialregressionSection } from '../sections/ml-supervised-learning/polynomial-regression';
import { regularizationSection } from '../sections/ml-supervised-learning/regularization';
import { logisticregressionSection } from '../sections/ml-supervised-learning/logistic-regression';
import { knnSection } from '../sections/ml-supervised-learning/knn';
import { naivebayesSection } from '../sections/ml-supervised-learning/naive-bayes';
import { svmSection } from '../sections/ml-supervised-learning/svm';
import { decisiontreesSection } from '../sections/ml-supervised-learning/decision-trees';
import { randomforestsSection } from '../sections/ml-supervised-learning/random-forests';
import { gradientboostingSection } from '../sections/ml-supervised-learning/gradient-boosting';
import { ensemblemethodsSection } from '../sections/ml-supervised-learning/ensemble-methods';
import { featureselectionSection } from '../sections/ml-supervised-learning/feature-selection';
import { imbalanceddataSection } from '../sections/ml-supervised-learning/imbalanced-data';
import { timeseriesforecastingSection } from '../sections/ml-supervised-learning/timeseries-forecasting';

// Import quizzes
import { mlfundamentalsQuiz } from '../quizzes/ml-supervised-learning/ml-fundamentals';
import { linearregressionQuiz } from '../quizzes/ml-supervised-learning/linear-regression';
import { polynomialregressionQuiz } from '../quizzes/ml-supervised-learning/polynomial-regression';
import { regularizationQuiz } from '../quizzes/ml-supervised-learning/regularization';
import { logisticregressionQuiz } from '../quizzes/ml-supervised-learning/logistic-regression';
import { knnQuiz } from '../quizzes/ml-supervised-learning/knn';
import { naivebayesQuiz } from '../quizzes/ml-supervised-learning/naive-bayes';
import { svmQuiz } from '../quizzes/ml-supervised-learning/svm';
import { decisiontreesQuiz } from '../quizzes/ml-supervised-learning/decision-trees';
import { randomforestsQuiz } from '../quizzes/ml-supervised-learning/random-forests';
import { gradientboostingQuiz } from '../quizzes/ml-supervised-learning/gradient-boosting';
import { ensemblemethodsQuiz } from '../quizzes/ml-supervised-learning/ensemble-methods';
import { featureselectionQuiz } from '../quizzes/ml-supervised-learning/feature-selection';
import { imbalanceddataQuiz } from '../quizzes/ml-supervised-learning/imbalanced-data';
import { timeseriesforecastingQuiz } from '../quizzes/ml-supervised-learning/timeseries-forecasting';

// Import multiple choice
import { mlfundamentalsMultipleChoice } from '../multiple-choice/ml-supervised-learning/ml-fundamentals';
import { linearregressionMultipleChoice } from '../multiple-choice/ml-supervised-learning/linear-regression';
import { polynomialregressionMultipleChoice } from '../multiple-choice/ml-supervised-learning/polynomial-regression';
import { regularizationMultipleChoice } from '../multiple-choice/ml-supervised-learning/regularization';
import { logisticregressionMultipleChoice } from '../multiple-choice/ml-supervised-learning/logistic-regression';
import { knnMultipleChoice } from '../multiple-choice/ml-supervised-learning/knn';
import { naivebayesMultipleChoice } from '../multiple-choice/ml-supervised-learning/naive-bayes';
import { svmMultipleChoice } from '../multiple-choice/ml-supervised-learning/svm';
import { decisiontreesMultipleChoice } from '../multiple-choice/ml-supervised-learning/decision-trees';
import { randomforestsMultipleChoice } from '../multiple-choice/ml-supervised-learning/random-forests';
import { gradientboostingMultipleChoice } from '../multiple-choice/ml-supervised-learning/gradient-boosting';
import { ensemblemethodsMultipleChoice } from '../multiple-choice/ml-supervised-learning/ensemble-methods';
import { featureselectionMultipleChoice } from '../multiple-choice/ml-supervised-learning/feature-selection';
import { imbalanceddataMultipleChoice } from '../multiple-choice/ml-supervised-learning/imbalanced-data';
import { timeseriesforecastingMultipleChoice } from '../multiple-choice/ml-supervised-learning/timeseries-forecasting';

export const mlSupervisedLearningModule: Module = {
  id: 'ml-supervised-learning',
  title: 'Classical Machine Learning - Supervised Learning',
  description:
    'Master supervised learning algorithms from linear models to ensemble methods, including regression, classification, and time series forecasting',
  category: 'machine-learning',
  difficulty: 'intermediate',
  estimatedTime: '40 hours',
  prerequisites: [
    'Module 4: Statistics Fundamentals',
    'Module 5: Probability Theory',
  ],
  icon: '🤖',
  keyTakeaways: [
    'Linear regression minimizes squared errors to fit lines/hyperplanes to data',
    'Regularization (L1/L2) prevents overfitting by penalizing large coefficients',
    'Logistic regression models probability using sigmoid for binary classification',
    'k-NN classifies based on majority vote of k nearest neighbors',
    'Naive Bayes applies conditional independence for fast probabilistic classification',
    'SVM finds maximum-margin hyperplane using kernel trick for non-linear boundaries',
    'Decision trees partition feature space with interpretable if-then rules',
    'Random Forests average multiple trees to reduce variance and improve accuracy',
    'Gradient Boosting builds models sequentially, correcting previous errors',
    'Ensemble methods combine diverse models for superior performance',
    'Feature selection removes irrelevant features to improve generalization',
    'Imbalanced data requires special metrics (F1, PR AUC) and resampling techniques',
    'Time series forecasting uses ARIMA and exponential smoothing for temporal patterns',
    'Cross-validation estimates generalization error on unseen data',
    'Bias-variance tradeoff: simple models underfit, complex models overfit',
  ],
  learningObjectives: [
    'Implement linear and polynomial regression with gradient descent',
    'Apply L1/L2 regularization to prevent overfitting',
    'Build logistic regression classifiers for binary and multi-class problems',
    'Use k-NN for classification and understand curse of dimensionality',
    'Apply Naive Bayes for text classification and categorical data',
    'Train SVM with different kernels (linear, polynomial, RBF)',
    'Build and prune decision trees for classification and regression',
    'Train Random Forests and interpret feature importance',
    'Implement Gradient Boosting with XGBoost, LightGBM, and CatBoost',
    'Create ensemble models using voting, bagging, boosting, and stacking',
    'Perform feature selection using filter, wrapper, and embedded methods',
    'Handle imbalanced datasets with SMOTE, class weights, and threshold tuning',
    'Forecast time series using ARIMA, SARIMA, and exponential smoothing',
    'Evaluate models using appropriate metrics (accuracy, precision, recall, F1, AUC)',
    'Tune hyperparameters with grid search and cross-validation',
    'Build production ML pipelines with scikit-learn Pipeline',
  ],
  sections: [
    {
      ...mlfundamentalsSection,
      quiz: mlfundamentalsQuiz,
      multipleChoice: mlfundamentalsMultipleChoice,
    },
    {
      ...linearregressionSection,
      quiz: linearregressionQuiz,
      multipleChoice: linearregressionMultipleChoice,
    },
    {
      ...polynomialregressionSection,
      quiz: polynomialregressionQuiz,
      multipleChoice: polynomialregressionMultipleChoice,
    },
    {
      ...regularizationSection,
      quiz: regularizationQuiz,
      multipleChoice: regularizationMultipleChoice,
    },
    {
      ...logisticregressionSection,
      quiz: logisticregressionQuiz,
      multipleChoice: logisticregressionMultipleChoice,
    },
    {
      ...knnSection,
      quiz: knnQuiz,
      multipleChoice: knnMultipleChoice,
    },
    {
      ...naivebayesSection,
      quiz: naivebayesQuiz,
      multipleChoice: naivebayesMultipleChoice,
    },
    {
      ...svmSection,
      quiz: svmQuiz,
      multipleChoice: svmMultipleChoice,
    },
    {
      ...decisiontreesSection,
      quiz: decisiontreesQuiz,
      multipleChoice: decisiontreesMultipleChoice,
    },
    {
      ...randomforestsSection,
      quiz: randomforestsQuiz,
      multipleChoice: randomforestsMultipleChoice,
    },
    {
      ...gradientboostingSection,
      quiz: gradientboostingQuiz,
      multipleChoice: gradientboostingMultipleChoice,
    },
    {
      ...ensemblemethodsSection,
      quiz: ensemblemethodsQuiz,
      multipleChoice: ensemblemethodsMultipleChoice,
    },
    {
      ...featureselectionSection,
      quiz: featureselectionQuiz,
      multipleChoice: featureselectionMultipleChoice,
    },
    {
      ...imbalanceddataSection,
      quiz: imbalanceddataQuiz,
      multipleChoice: imbalanceddataMultipleChoice,
    },
    {
      ...timeseriesforecastingSection,
      quiz: timeseriesforecastingQuiz,
      multipleChoice: timeseriesforecastingMultipleChoice,
    },
  ],
};
