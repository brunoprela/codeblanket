/**
 * Discussion questions for Data Labeling & Annotation section
 */

export const dataLabelingAnnotationQuiz = [
  {
    id: 'data-label-q-1',
    question:
      'You need to label 50,000 customer support conversations as "Resolved" or "Unresolved". You estimate 2 minutes per conversation for single annotation = 1,667 hours. With $15/hour crowdworkers, that\'s $25,000 and 7 weeks with 10 workers. Design a more efficient annotation strategy that maintains quality while reducing cost and time by at least 50%.',
    hint: 'Consider: Can you use the AI model itself to pre-label? Can you focus human effort on uncertain/important examples? Can you use active learning? Can you bootstrap from simpler signals?',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Weak supervision: Bootstrap from existing signals (button clicks, timestamps, keywords)',
      'Active learning: Human-label uncertain cases, not random samples',
      'Clustering: Label cluster representatives, propagate to cluster members',
      'Model-assisted labeling: AI suggests, human verifies (3-4x faster)',
      'Early stopping: Monitor accuracy, stop collecting when plateaus',
      'Semi-supervised learning: Trust high-confidence model predictions',
      'Achieved 78% cost and time savings with minimal accuracy loss',
    ],
  },
  {
    id: 'data-label-q-2',
    question:
      "You're building a sentiment analysis model and hire crowdworkers to label tweets as Positive/Neutral/Negative. After 10,000 annotations, you discover that crowdworkers have a strong bias: they label 60% of tweets as Neutral (true distribution is 30%). This is causing your model to over-predict Neutral. How do you fix the existing labels and prevent this bias going forward?",
    hint: 'Consider calibration techniques, annotator training, and statistical adjustments. Can you detect which Neutral labels are likely wrong? Can you fix incentives or guidelines that cause the bias?',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Root cause: Ambiguity aversion, bad incentives, unclear guidelines',
      'Calibrate existing labels: Confusion matrix from expert sample, probabilistic correction',
      'Use soft labels: Model trains on label distributions, not hard categories',
      'Re-annotate uncertain cases: Where P(correct_label) < 0.6',
      'Prevent future bias: Better guidelines, quality-based payment, calibration tests',
      'Continuous monitoring: Hidden gold standards detect annotator drift',
      'Statistical adjustment: Class weights during training compensate for distribution bias',
    ],
  },
  {
    id: 'data-label-q-3',
    question:
      'You need to label medical images for cancer detection. Option A: Hire radiology residents at $50/hour (average accuracy: 88%). Option B: Hire board-certified radiologists at $200/hour (average accuracy: 94%). Your budget is $10,000 for labeling 5,000 images. Images take 2 minutes each (100 hours total for single annotation). Design an optimal strategy that maximizes label quality within budget.',
    hint: 'Consider: Can you use cheaper annotators for easier cases and experts for hard cases? Can you use ensemble methods? What about active learning to focus expert time on important examples?',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Tiered expertise: Residents for initial screening, radiologists for verification',
      'Risk-based allocation: Expert time on high-stakes cases (all positives, uncertain cases)',
      'Budget optimization: Single resident pass + selective expert review',
      'Majority vote: 3 residents (88% each) ≈ 1 radiologist (94%) via ensemble',
      'Safety first: In medical context, prioritize minimizing false negatives',
      'Expected result: ~92% accuracy vs 88% (residents only) or 94% (radiologists only, 50% coverage)',
    ],
  },
];
