/**
 * Multiple Choice Questions for Imbalanced Data
 */

import { MultipleChoiceQuestion } from '../../../types';

export const imbalanceddataMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'imbalanced-data-mc-1',
    question:
      'A fraud detection dataset has 99% legitimate transactions and 1% fraud. A model that predicts "legitimate" for all transactions achieves 99% accuracy. What is the problem?',
    options: [
      'The model is overfitting',
      'The model is not detecting any fraud (0% recall for fraud class)',
      'The model needs more training data',
      'Feature scaling was not applied',
    ],
    correctAnswer: 1,
    explanation:
      'This is the accuracy paradox. While 99% accuracy sounds good, the model is useless for fraud detection - it catches 0% of fraud (0% recall). For imbalanced data, accuracy is misleading. Use metrics like recall, precision, and F1-score that focus on the minority class performance.',
  },
  {
    id: 'imbalanced-data-mc-2',
    question: 'What does SMOTE do to handle class imbalance?',
    options: [
      'Removes majority class samples',
      'Duplicates minority class samples',
      'Creates synthetic minority samples by interpolating between existing ones',
      'Adjusts class weights in the loss function',
    ],
    correctAnswer: 2,
    explanation:
      'SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority class samples by interpolating between existing minority samples and their k-nearest neighbors. This is more effective than simple duplication because it generates diverse samples rather than exact copies.',
  },
  {
    id: 'imbalanced-data-mc-3',
    question:
      'For a medical screening test where missing a disease (false negative) is very costly, which metric should be prioritized?',
    options: ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity'],
    correctAnswer: 2,
    explanation:
      'Recall (sensitivity) measures the proportion of actual positives that are correctly identified. When false negatives are costly (missing a disease), you want to maximize recall - catch as many true cases as possible, even if it means more false positives. This typically requires lowering the classification threshold.',
  },
  {
    id: 'imbalanced-data-mc-4',
    question:
      'What is the effect of setting `class_weight="balanced"` in scikit-learn models?',
    options: [
      'It resamples the training data to balance classes',
      'It adjusts the loss function to penalize minority class errors more',
      'It always improves accuracy',
      'It removes majority class samples',
    ],
    correctAnswer: 1,
    explanation:
      'Setting class_weight="balanced" adjusts the loss function to weight classes inversely proportional to their frequencies. This makes the model penalize errors on the minority class more heavily during training, without actually resampling the data. It\'s computationally efficient and often effective for imbalanced data.',
  },
  {
    id: 'imbalanced-data-mc-5',
    question:
      'You have a severely imbalanced dataset (1:1000 ratio). You apply SMOTE and now have a balanced dataset, but test performance is worse than before. What is the most likely explanation?',
    options: [
      'SMOTE never works for classification',
      'The severe imbalance led SMOTE to create too many unrealistic synthetic samples, adding noise',
      'You need to use random oversampling instead',
      'Feature scaling was not applied',
    ],
    correctAnswer: 1,
    explanation:
      "For severely imbalanced data (>1:100), SMOTE can struggle. Creating so many synthetic samples may generate unrealistic examples that confuse the model, especially if minority class samples are sparse or noisy. For severe imbalance, consider: (1) Less aggressive oversampling (don't fully balance); (2) Ensemble methods like Balanced Random Forest; (3) Anomaly detection approaches; (4) Focus on threshold tuning rather than resampling.",
  },
];
