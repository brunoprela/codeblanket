/**
 * Discussion Questions for Imbalanced Data
 */

import { QuizQuestion } from '../../../types';

export const imbalanceddataQuiz: QuizQuestion[] = [
  {
    id: 'imbalanced-data-q1',
    question:
      'Why is accuracy a misleading metric for imbalanced datasets? What metrics should be used instead and why?',
    hint: 'Think about what happens when a model always predicts the majority class.',
    sampleAnswer:
      'Accuracy is misleading because a naive model predicting only the majority class achieves high accuracy. Example: 99% non-fraud dataset - predicting "not fraud" always gives 99% accuracy but 0% fraud detection! This is the "accuracy paradox". Better metrics: (1) Precision: of predicted positives, how many correct? Critical when false positives costly (spam filter). (2) Recall/Sensitivity: of actual positives, how many found? Critical when false negatives costly (disease diagnosis). (3) F1-Score: harmonic mean balancing precision and recall. (4) ROC AUC: measures discrimination ability across thresholds. (5) PR AUC (Precision-Recall): better than ROC for imbalanced data because it focuses on positive class. Use PR AUC for imbalanced data since ROC can be optimistically biased. Choice depends on cost: FP expensive→optimize precision, FN expensive→optimize recall. For balanced tradeoff, use F1. Always report confusion matrix to show FP/FN tradeoffs explicitly.',
    keyPoints: [
      'Accuracy misleading: majority class prediction gives high accuracy',
      'Precision: of predicted positives, how many correct',
      'Recall: of actual positives, how many found',
      'F1: harmonic mean of precision and recall',
      'PR AUC better than ROC AUC for imbalanced data',
    ],
  },
  {
    id: 'imbalanced-data-q2',
    question:
      'Compare SMOTE with simple random oversampling. What are the advantages and potential pitfalls of SMOTE?',
    hint: 'Think about how synthetic samples are created and what this means for the decision boundary.',
    sampleAnswer:
      'Random oversampling duplicates existing minority samples. Simple but causes overfitting - model memorizes duplicates. SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic samples by interpolating between minority examples and their k nearest neighbors. For sample x, finds k neighbors, selects one randomly, generates point along line segment: x_new = x + λ(x_neighbor - x). Advantages: (1) Generates diverse samples, reducing overfitting; (2) Smooths decision boundary; (3) More generalizable than duplication. Pitfalls: (1) Can generate unrealistic samples if data has complex structure; (2) Increases overlap if classes already close; (3) Computationally more expensive than random; (4) Can worsen performance if noise present. Variants: ADASYN adapts based on local difficulty, Borderline-SMOTE only synthesizes near boundary. Best practice: SMOTE often improves over random, but not always - validate with CV. Works well for moderate imbalance (1:10 to 1:100), less effective for extreme imbalance.',
    keyPoints: [
      'Random oversampling duplicates, causes overfitting',
      'SMOTE interpolates between minority neighbors',
      'SMOTE generates diverse samples, smoother boundary',
      'Pitfalls: unrealistic samples, increased overlap',
      'Best for moderate imbalance (1:10 to 1:100)',
    ],
  },
  {
    id: 'imbalanced-data-q3',
    question:
      'Discuss the tradeoff between precision and recall for imbalanced classification. How do you choose the optimal threshold based on business costs?',
    hint: 'Consider the cost of false positives vs false negatives in different applications.',
    sampleAnswer:
      'Precision-recall tradeoff: lowering classification threshold increases recall (catch more positives) but decreases precision (more false positives). Raising threshold does opposite. Cannot maximize both simultaneously - must balance based on business costs. Framework: (1) Define cost: C_FP (cost of false positive), C_FN (cost of false negative). (2) Calculate expected cost at each threshold: Cost = C_FP × FP + C_FN × FN. (3) Choose threshold minimizing expected cost. Examples: Spam filter - FP very costly (legitimate email marked spam), prioritize precision, use high threshold (0.7-0.9). Disease screening - FN very costly (miss disease), prioritize recall, use low threshold (0.3-0.4). Fraud detection - balance needed, optimize F1 or use cost-sensitive threshold. Implementation: Plot precision-recall vs threshold, annotate with business costs, select optimal. If C_FN = 10 × C_FP, find threshold maximizing recall while keeping precision reasonable. Always validate on holdout set. Default 0.5 threshold is rarely optimal for imbalanced data!',
    keyPoints: [
      'Lower threshold: higher recall, lower precision',
      'Higher threshold: lower recall, higher precision',
      'Choose based on C_FP vs C_FN costs',
      'FP costly→high threshold, FN costly→low threshold',
      'Default 0.5 rarely optimal for imbalanced data',
    ],
  },
];
