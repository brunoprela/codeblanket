/**
 * Model Evaluation & Optimization Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { trainTestSplitValidation } from '../sections/ml-model-evaluation-optimization/train-test-split-validation';
import { crossValidationTechniques } from '../sections/ml-model-evaluation-optimization/cross-validation-techniques';
import { regressionMetrics } from '../sections/ml-model-evaluation-optimization/regression-metrics';
import { classificationMetrics } from '../sections/ml-model-evaluation-optimization/classification-metrics';
import { multiClassMultiLabelMetrics } from '../sections/ml-model-evaluation-optimization/multi-class-multi-label-metrics';
import { biasVarianceTradeoff } from '../sections/ml-model-evaluation-optimization/bias-variance-tradeoff';
import { hyperparameterTuning } from '../sections/ml-model-evaluation-optimization/hyperparameter-tuning';
import { modelSelection } from '../sections/ml-model-evaluation-optimization/model-selection';
import { featureImportanceInterpretation } from '../sections/ml-model-evaluation-optimization/feature-importance-interpretation';
import { modelDebugging } from '../sections/ml-model-evaluation-optimization/model-debugging';

// Import quizzes
import { trainTestSplitValidationQuiz } from '../quizzes/ml-model-evaluation-optimization/train-test-split-validation';
import { crossValidationTechniquesQuiz } from '../quizzes/ml-model-evaluation-optimization/cross-validation-techniques';
import { regressionMetricsQuiz } from '../quizzes/ml-model-evaluation-optimization/regression-metrics';
import { classificationMetricsQuiz } from '../quizzes/ml-model-evaluation-optimization/classification-metrics';
import { multiClassMultiLabelMetricsQuiz } from '../quizzes/ml-model-evaluation-optimization/multi-class-multi-label-metrics';
import { biasVarianceTradeoffQuiz } from '../quizzes/ml-model-evaluation-optimization/bias-variance-tradeoff';
import { hyperparameterTuningQuiz } from '../quizzes/ml-model-evaluation-optimization/hyperparameter-tuning';
import { modelSelectionQuiz } from '../quizzes/ml-model-evaluation-optimization/model-selection';
import { featureImportanceInterpretationQuiz } from '../quizzes/ml-model-evaluation-optimization/feature-importance-interpretation';
import { modelDebuggingQuiz } from '../quizzes/ml-model-evaluation-optimization/model-debugging';

// Import multiple choice
import { trainTestSplitValidationMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/train-test-split-validation';
import { crossValidationTechniquesMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/cross-validation-techniques';
import { regressionMetricsMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/regression-metrics';
import { classificationMetricsMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/classification-metrics';
import { multiClassMultiLabelMetricsMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/multi-class-multi-label-metrics';
import { biasVarianceTradeoffMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/bias-variance-tradeoff';
import { hyperparameterTuningMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/hyperparameter-tuning';
import { modelSelectionMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/model-selection';
import { featureImportanceInterpretationMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/feature-importance-interpretation';
import { modelDebuggingMultipleChoice } from '../multiple-choice/ml-model-evaluation-optimization/model-debugging';

export const mlModelEvaluationOptimizationModule: Module = {
  id: 'ml-model-evaluation-optimization',
  title: 'Model Evaluation & Optimization',
  description:
    'Master comprehensive model evaluation techniques, metrics, hyperparameter tuning, and optimization strategies to build robust, high-performing machine learning systems',
  category: 'machine-learning',
  difficulty: 'intermediate',
  estimatedTime: '18 hours',
  prerequisites: [
    'Supervised Learning Fundamentals',
    'Data Preprocessing & Feature Engineering',
  ],
  icon: '📊',
  keyTakeaways: [
    'Proper train-test split prevents overfitting and provides unbiased performance estimates',
    'Cross-validation gives robust model performance estimates with confidence intervals',
    'Regression metrics (MAE, RMSE, R², MAPE) measure different aspects of prediction quality',
    'Classification metrics (Precision, Recall, F1, AUC) balance different types of errors',
    'Multi-class metrics use macro/micro/weighted averaging for different analysis goals',
    'Bias-variance tradeoff is fundamental: simpler models underfit, complex models overfit',
    'Hyperparameter tuning can improve performance 10-30% through grid/random search',
    'Model selection balances performance, speed, interpretability, and business constraints',
    'Feature importance and SHAP values explain predictions and build stakeholder trust',
    'Systematic debugging diagnoses issues: check baselines, learning curves, error patterns',
  ],
  learningObjectives: [
    'Implement proper train-test splits with stratification for unbiased evaluation',
    'Apply k-fold, stratified, and time-series cross-validation appropriately',
    'Select and interpret regression metrics based on business requirements',
    'Use classification metrics to balance precision/recall for different use cases',
    'Evaluate multi-class and multi-label models with appropriate averaging strategies',
    'Diagnose and address bias (underfitting) and variance (overfitting) issues',
    'Perform efficient hyperparameter tuning with grid search and random search',
    'Compare and select models considering performance, latency, and interpretability',
    'Explain model predictions using feature importance, permutation importance, and SHAP',
    'Debug models systematically using learning curves, error analysis, and data quality checks',
    'Implement nested cross-validation for unbiased hyperparameter selection',
    'Build production-ready evaluation pipelines with monitoring and retraining strategies',
  ],
  sections: [
    {
      ...trainTestSplitValidation,
      quiz: trainTestSplitValidationQuiz,
      multipleChoice: trainTestSplitValidationMultipleChoice,
    },
    {
      ...crossValidationTechniques,
      quiz: crossValidationTechniquesQuiz,
      multipleChoice: crossValidationTechniquesMultipleChoice,
    },
    {
      ...regressionMetrics,
      quiz: regressionMetricsQuiz,
      multipleChoice: regressionMetricsMultipleChoice,
    },
    {
      ...classificationMetrics,
      quiz: classificationMetricsQuiz,
      multipleChoice: classificationMetricsMultipleChoice,
    },
    {
      ...multiClassMultiLabelMetrics,
      quiz: multiClassMultiLabelMetricsQuiz,
      multipleChoice: multiClassMultiLabelMetricsMultipleChoice,
    },
    {
      ...biasVarianceTradeoff,
      quiz: biasVarianceTradeoffQuiz,
      multipleChoice: biasVarianceTradeoffMultipleChoice,
    },
    {
      ...hyperparameterTuning,
      quiz: hyperparameterTuningQuiz,
      multipleChoice: hyperparameterTuningMultipleChoice,
    },
    {
      ...modelSelection,
      quiz: modelSelectionQuiz,
      multipleChoice: modelSelectionMultipleChoice,
    },
    {
      ...featureImportanceInterpretation,
      quiz: featureImportanceInterpretationQuiz,
      multipleChoice: featureImportanceInterpretationMultipleChoice,
    },
    {
      ...modelDebugging,
      quiz: modelDebuggingQuiz,
      multipleChoice: modelDebuggingMultipleChoice,
    },
  ],
};
