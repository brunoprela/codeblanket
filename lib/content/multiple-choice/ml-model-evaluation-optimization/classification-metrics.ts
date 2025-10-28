export const classificationMetricsMultipleChoice = {
  title: 'Classification Metrics - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'For fraud detection where fraudulent transactions are 0.1% of all transactions, why is accuracy a poor metric?',
      options: [
        'Accuracy is always a poor metric',
        "A model predicting 'not fraud' for everything achieves 99.9% accuracy without detecting any fraud",
        "Accuracy can't be calculated for fraud detection",
        'Fraud detection requires regression, not classification',
      ],
      correctAnswer: 1,
      explanation:
        "With 99.9% negative class, a naive 'always predict negative' model achieves 99.9% accuracy while catching 0% of fraud. For imbalanced data, use precision, recall, F1, or PR-AUC instead.",
      difficulty: 'intermediate' as const,
      category: 'Imbalanced Data',
    },
    {
      id: 2,
      question:
        'In medical diagnosis, which metric is typically most important?',
      options: [
        'Accuracy - overall correctness matters most',
        'Precision - avoid false positives at all costs',
        'Recall (Sensitivity) - must catch all positive cases to avoid missing diseases',
        'F1 score - always balance precision and recall equally',
      ],
      correctAnswer: 2,
      explanation:
        "In medical diagnosis, missing a disease (false negative) can be life-threatening, while false positives lead to additional testing (acceptable cost). High recall ensures we don't miss actual cases, even if it means more false alarms.",
      difficulty: 'easy' as const,
      category: 'Application',
    },
    {
      id: 3,
      question: 'What does an AUC-ROC score of 0.5 indicate?',
      options: [
        'Perfect classification',
        'The model is performing no better than random guessing',
        'The model is 50% accurate',
        'The model is excellent',
      ],
      correctAnswer: 1,
      explanation:
        "AUC-ROC of 0.5 means the model has no discrimination ability - it's equivalent to random guessing. AUC=1.0 is perfect, AUC=0.5 is random, and AUC<0.5 means the model is performing worse than random.",
      difficulty: 'easy' as const,
      category: 'ROC Analysis',
    },
    {
      id: 4,
      question:
        'For spam email classification, would you optimize for high precision or high recall?',
      options: [
        'High recall - catch all spam even if some legitimate emails are marked as spam',
        'High precision - avoid marking legitimate emails as spam, even if some spam gets through',
        'Neither - use accuracy only',
        'Both must be exactly equal',
      ],
      correctAnswer: 1,
      explanation:
        'For spam filtering, false positives (legitimate→spam) are worse than false negatives (spam→inbox). Missing important emails is critical, while spam in inbox is just annoying. Optimize for high precision to minimize false positives.',
      difficulty: 'intermediate' as const,
      category: 'Trade-offs',
    },
    {
      id: 5,
      question:
        'Why is PR-AUC (Precision-Recall AUC) preferred over ROC-AUC for highly imbalanced datasets?',
      options: [
        'PR-AUC is faster to compute',
        'ROC-AUC uses true negatives which can make performance look artificially good when negatives dominate',
        "PR-AUC works for regression while ROC-AUC doesn't",
        'PR-AUC is always higher than ROC-AUC',
      ],
      correctAnswer: 1,
      explanation:
        'ROC uses FPR (which includes TN in denominator). With millions of true negatives, FPR stays low even with poor precision, making ROC-AUC look good. PR-AUC ignores TN and focuses on precision/recall, better reflecting performance on the minority class.',
      difficulty: 'advanced' as const,
      category: 'Metrics',
    },
  ],
};
