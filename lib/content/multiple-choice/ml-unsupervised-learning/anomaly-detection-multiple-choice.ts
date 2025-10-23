/**
 * Multiple Choice Questions: Anomaly Detection
 * Module: Classical Machine Learning - Unsupervised Learning
 */

import { MultipleChoiceQuestion } from '../../../types';

export const anomaly_detectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'anomaly-detection-mc1',
    question: 'What is the main principle behind Isolation Forest?',
    options: [
      'Anomalies form separate clusters',
      'Anomalies are easier to isolate and require fewer splits in random trees',
      'Anomalies have higher density than normal points',
      'Anomalies are always at the edges of the feature space',
    ],
    correctAnswer: 1,
    explanation:
      "Isolation Forest is based on the principle that anomalies are 'few and different', making them easier to isolate with fewer random splits in a tree. Normal points in dense regions require more splits to isolate.",
  },
  {
    id: 'anomaly-detection-mc2',
    question: 'In Local Outlier Factor (LOF), an anomaly is characterized by:',
    options: [
      'Being far from all other points',
      'Having lower density than its neighbors',
      'Being in the smallest cluster',
      'Having the highest feature values',
    ],
    correctAnswer: 1,
    explanation:
      'LOF identifies anomalies based on local density. A point is an outlier if its density is significantly lower than the density of its neighbors, making it effective for detecting local anomalies in varying density data.',
  },
  {
    id: 'anomaly-detection-mc3',
    question:
      'What does the contamination parameter in anomaly detection specify?',
    options: [
      'The percentage of features that are corrupted',
      'The expected proportion of anomalies in the dataset',
      'The threshold distance for classifying outliers',
      'The number of anomaly detection methods to use',
    ],
    correctAnswer: 1,
    explanation:
      'The contamination parameter specifies the expected proportion (percentage) of anomalies in the dataset. It sets the threshold for classifying points as anomalies, typically ranging from 0.01 to 0.1 (1% to 10%).',
  },
  {
    id: 'anomaly-detection-mc4',
    question:
      'Why is accuracy NOT a good metric for evaluating anomaly detection?',
    options: [
      'Accuracy is too difficult to calculate',
      'Anomalies are rare, so high accuracy can be achieved by labeling everything as normal',
      'Accuracy only works for supervised learning',
      "Accuracy doesn't account for distance measures",
    ],
    correctAnswer: 1,
    explanation:
      "Due to class imbalance (anomalies are rare, often <1%), a naive classifier predicting 'normal' for everything achieves >99% accuracy but misses all anomalies. Use precision, recall, F1, or PR-AUC instead.",
  },
  {
    id: 'anomaly-detection-mc5',
    question:
      'You are monitoring network traffic for intrusions (anomalies). Your system has high recall (95%) but low precision (30%). What does this mean?',
    options: [
      'The system catches most attacks but generates many false alarms',
      'The system misses most attacks but rarely gives false alarms',
      'The system performs well on both detecting and avoiding false alarms',
      'The system is completely ineffective',
    ],
    correctAnswer: 0,
    explanation:
      'High recall (95%) means the system catches 95% of actual attacks (good at not missing anomalies). Low precision (30%) means only 30% of flagged items are actual attacks - 70% are false positives. This trade-off is common in anomaly detection: sensitive systems catch more anomalies but generate more false alarms. The choice depends on the cost of false positives vs false negatives.',
  },
];
