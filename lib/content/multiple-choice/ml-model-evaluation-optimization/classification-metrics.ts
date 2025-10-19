import { MultipleChoiceQuestion } from '../../../types';

export const classificationMetricsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'classification-metrics-mc-1',
    question:
      'A fraud detection model has the following confusion matrix for 10,000 transactions: TN=9,850, FP=50, FN=20, TP=80. What is the precision?',
    options: [
      '80 / (80 + 20) = 0.80 or 80%',
      '80 / (80 + 50) = 0.615 or 61.5%',
      '80 / (9,850 + 80) = 0.008 or 0.8%',
      '(80 + 9,850) / 10,000 = 0.993 or 99.3%',
    ],
    correctAnswer: 1,
    explanation:
      'Precision = TP / (TP + FP) = 80 / (80 + 50) = 80/130 = 0.615 or 61.5%. Precision asks "of all positive predictions, how many were correct?" We predicted 130 transactions as fraud (80+50), and 80 were actually fraud.',
  },
  {
    id: 'classification-metrics-mc-2',
    question:
      'For a cancer screening model, which metric should you prioritize to minimize the risk of missing cancer cases?',
    options: ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity'],
    correctAnswer: 2,
    explanation:
      'Recall (Sensitivity) measures the proportion of actual positive cases that were correctly identified: TP/(TP+FN). High recall means few false negatives (missed cancers). For cancer screening, missing a cancer case (false negative) is catastrophic, so recall should be prioritized even at the cost of more false positives.',
  },
  {
    id: 'classification-metrics-mc-3',
    question:
      'You have a model for detecting rare events (0.5% positive class). The model achieves 99% accuracy. What can you conclude?',
    options: [
      'The model is excellent and ready for production',
      'The model might be useless - need to check precision, recall, and F1',
      'The model has perfect balance between precision and recall',
      '99% accuracy means 99% of rare events are detected',
    ],
    correctAnswer: 1,
    explanation:
      'For highly imbalanced data (0.5% positive), a naive model that always predicts negative would achieve 99.5% accuracy. 99% accuracy could mean the model is barely better than random guessing on the minority class. You must check precision, recall, and F1-score to understand actual performance on the rare positive class. Accuracy alone is meaningless for imbalanced datasets.',
  },
  {
    id: 'classification-metrics-mc-4',
    question:
      'Two models have the following performance: Model A has Precision=0.9, Recall=0.4, F1=0.55. Model B has Precision=0.6, Recall=0.8, F1=0.69. Which statement is correct?',
    options: [
      'Model A is better because it has higher precision',
      'Model B is better because it has higher F1-score',
      'The models are equal since (0.9+0.4)/2 = (0.6+0.8)/2',
      'Model A is better because precision is more important than recall',
    ],
    correctAnswer: 1,
    explanation:
      'F1-score (harmonic mean) balances precision and recall. Model B has higher F1 (0.69 vs 0.55), indicating better overall performance. Model A has high precision but very low recall (missing 60% of positives). Model B has better balance. Note that arithmetic mean would give both 0.65, but F1\'s harmonic mean properly penalizes imbalance. Which is "better" ultimately depends on the application, but F1 provides the best single-number comparison.',
  },
  {
    id: 'classification-metrics-mc-5',
    question:
      'For a highly imbalanced dataset (1% positive class), which metric would be MOST appropriate for model evaluation?',
    options: ['Accuracy', 'ROC-AUC', 'Precision-Recall AUC', 'Specificity'],
    correctAnswer: 2,
    explanation:
      'Precision-Recall AUC is most appropriate for highly imbalanced datasets (<5% minority class). ROC-AUC can be overly optimistic because FPR includes the large true negative count, making it look artificially good. Precision-Recall AUC focuses only on the positive class performance without being influenced by the large negative class. Accuracy is meaningless for imbalanced data.',
  },
];
