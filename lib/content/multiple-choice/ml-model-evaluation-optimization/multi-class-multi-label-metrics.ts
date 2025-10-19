import { MultipleChoiceQuestion } from '../../../types';

export const multiClassMultiLabelMetricsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'multi-class-multi-label-metrics-mc-1',
      question:
        'For a 5-class classification problem with per-class F1 scores of [0.9, 0.8, 0.7, 0.6, 0.5], what is the macro-averaged F1 score?',
      options: [
        '0.90 (the highest class score)',
        '0.70 (the arithmetic mean)',
        '0.75 (the weighted average)',
        '0.80 (the median)',
      ],
      correctAnswer: 1,
      explanation:
        'Macro-averaged F1 is the simple arithmetic mean of per-class F1 scores: (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5 = 3.5 / 5 = 0.70. All classes are weighted equally regardless of their frequency in the dataset.',
    },
    {
      id: 'multi-class-multi-label-metrics-mc-2',
      question:
        'In multi-label classification, a movie has true genres {Action, Comedy} and predicted genres {Action, Sci-Fi}. What is the Jaccard score for this prediction?',
      options: [
        '0.0 (completely wrong)',
        '0.33 (1 correct out of 3 total)',
        '0.50 (1 correct, 1 wrong)',
        '1.0 (perfect match)',
      ],
      correctAnswer: 1,
      explanation:
        'Jaccard score (Intersection over Union) = |True ∩ Pred| / |True ∪ Pred|. Intersection: {Action} (1 label). Union: {Action, Comedy, Sci-Fi} (3 labels). Jaccard = 1/3 = 0.33. It measures the overlap between predicted and true label sets.',
    },
    {
      id: 'multi-class-multi-label-metrics-mc-3',
      question:
        'For multi-class classification, what does it mean when micro-averaged F1 equals macro-averaged F1?',
      options: [
        'The model has perfect performance',
        'All classes have identical F1 scores',
        'The classes are perfectly balanced in the dataset',
        'The model makes no predictions',
      ],
      correctAnswer: 1,
      explanation:
        'Micro and macro F1 are equal when all classes have the same F1 score. Micro-average aggregates all TP/FP/FN globally, while macro-average is the mean of per-class scores. They converge when per-class performance is uniform. This does NOT require balanced classes or perfect performance.',
    },
    {
      id: 'multi-class-multi-label-metrics-mc-4',
      question:
        'In a 3-class problem with 80% Class A, 15% Class B, and 5% Class C, which averaging strategy would be MOST sensitive to poor performance on Class C?',
      options: [
        'Micro-average (treats all predictions equally)',
        'Weighted-average (accounts for class frequency)',
        'Macro-average (treats all classes equally)',
        'All strategies are equally sensitive',
      ],
      correctAnswer: 2,
      explanation:
        'Macro-average treats all classes equally (each contributes 33.3% to the final score), so poor Class C performance significantly impacts the metric. Micro and weighted averages are dominated by the majority class (80%), making them less sensitive to minority class performance.',
    },
    {
      id: 'multi-class-multi-label-metrics-mc-5',
      question:
        'What is the key difference between multi-class and multi-label classification evaluation?',
      options: [
        'Multi-class uses confusion matrices; multi-label does not',
        'Multi-class has one label per sample; multi-label allows multiple simultaneous labels',
        'Multi-class requires more training data than multi-label',
        'Multi-label always has better performance metrics',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-class classification assigns each sample to exactly ONE class (mutually exclusive). Multi-label classification allows each sample to have MULTIPLE labels simultaneously (non-exclusive). This fundamental difference requires different evaluation metrics: multi-label needs metrics like Hamming loss and Jaccard score that handle label sets.',
    },
  ];
