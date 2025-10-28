export const multiClassMultiLabelMetricsMultipleChoice = {
  title: 'Multi-class & Multi-label Metrics - Multiple Choice Questions',
  questions: [
    {
      id: 1,
      question:
        'What is the key difference between macro-average and micro-average F1 score in multi-class classification?',
      options: [
        'Macro is always higher than micro',
        'Macro gives equal weight to each class; micro gives equal weight to each sample',
        'Micro is used for regression; macro for classification',
        'They are the same thing with different names',
      ],
      correctAnswer: 1,
      explanation:
        'Macro-average calculates metrics per class then averages (all classes equal weight). Micro-average aggregates all predictions globally then calculates (all samples equal weight). Macro highlights minority class performance; micro reflects overall accuracy.',
      difficulty: 'intermediate' as const,
      category: 'Averaging',
    },
    {
      id: 2,
      question:
        'For a multi-class problem with imbalanced classes where all classes are equally important for business, which averaging method should you use?',
      options: [
        'Micro-average - focuses on overall accuracy',
        'Macro-average - treats each class equally regardless of size',
        'Weighted-average - accounts for class imbalance',
        'No averaging - just use accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'When all classes are equally important (e.g., all disease types matter equally), use macro-average. It prevents frequent classes from dominating the metric and ensures good performance across all classes, including rare ones.',
      difficulty: 'intermediate' as const,
      category: 'Strategy',
    },
    {
      id: 3,
      question:
        'In multi-label classification, what does Hamming Loss measure?',
      options: [
        'The total number of classes',
        'The fraction of labels that are incorrectly predicted',
        'The accuracy on the most frequent label',
        'The time to make predictions',
      ],
      correctAnswer: 1,
      explanation:
        'Hamming Loss is the fraction of wrong labels (both false positives and false negatives) divided by total labels. For example, if you predict [1,0,1] but truth is [1,1,0], 2 out of 3 labels are wrong, Hamming Loss = 2/3 = 0.67.',
      difficulty: 'easy' as const,
      category: 'Multi-label',
    },
    {
      id: 4,
      question: 'What is the strictest metric for multi-label classification?',
      options: [
        'Hamming Loss',
        'Jaccard Score (IoU)',
        'Exact Match Ratio (Subset Accuracy)',
        'Macro F1',
      ],
      correctAnswer: 2,
      explanation:
        'Exact Match Ratio (Subset Accuracy) requires ALL labels to be correct for a prediction to count as correct. If ground truth is [1,1,0] and prediction is [1,1,1], this counts as wrong even though 2/3 labels are correct.',
      difficulty: 'advanced' as const,
      category: 'Multi-label',
    },
    {
      id: 5,
      question:
        'A multi-class classifier has Macro F1=0.60, Micro F1=0.85. What does this tell you?',
      options: [
        'The model is performing consistently across all classes',
        'The model performs well on frequent classes but poorly on minority classes',
        'The model is overfitting',
        "There\'s an error in the calculations",
      ],
      correctAnswer: 1,
      explanation:
        'Large gap between macro (0.60) and micro (0.85) indicates imbalanced performance. High micro F1 shows good overall accuracy (dominated by frequent classes), but low macro F1 shows poor performance on minority classes. The model is biased toward frequent classes.',
      difficulty: 'advanced' as const,
      category: 'Interpretation',
    },
  ],
};
