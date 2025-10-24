import { MultipleChoiceQuestion } from '@/types/curriculum';

export const modelMonitoringQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'mm-mc-1',
    question:
      "Your deployed fraud detection model shows stable accuracy metrics but business stakeholders report increased fraud losses. Investigation reveals the model's precision remains high but recall has dropped significantly. What type of drift is most likely occurring?",
    options: [
      'Covariate shift: input feature distributions have changed',
      'Concept drift: relationship between features and target has changed',
      'Label drift: target variable distribution has changed',
      'Prediction drift: model output distribution has changed without performance degradation',
    ],
    correctAnswer: 1,
    explanation:
      "This scenario indicates concept drift: the patterns associated with fraud have evolved (fraudsters adapting tactics), changing the relationship between features and the fraud label. The model maintains high precision (low false positives) but misses new fraud patterns (low recall/high false negatives). Covariate shift (option A) would typically show in feature distributions but wouldn't necessarily explain the selective recall drop. Label drift (option C) refers to changing fraud rates, not detection difficulty. Prediction drift (option D) contradicts the performance degradation observed.",
    difficulty: 'advanced',
    topic: 'Model Monitoring',
  },
  {
    id: 'mm-mc-2',
    question:
      "You're monitoring a recommendation model in production using the Kolmogorov-Smirnov (KS) test to detect feature drift. The test shows significant drift (p < 0.01) in multiple features weekly, but model performance metrics remain stable. What is the most appropriate action?",
    options: [
      'Immediately retrain the model to address the detected drift',
      'Investigate whether the drift affects model performance using A/B testing or correlation analysis',
      'Adjust the KS test threshold to reduce false alarms',
      'Switch to a different drift detection method like Population Stability Index (PSI)',
    ],
    correctAnswer: 1,
    explanation:
      "Statistical significance doesn't always equal practical significance. The drift may be statistically detectable but not impactful to model performance. Investigate whether drifted features correlate with performance degradation through A/B testing (serve predictions from current vs. retrained model) or analyze performance on drifted segments. Option A (immediate retraining) wastes resources if drift is benign. Option C (adjusting threshold) hides potentially important drift. Option D (switching methods) doesn't address the core question of whether detected drift matters.",
    difficulty: 'advanced',
    topic: 'Model Monitoring',
  },
  {
    id: 'mm-mc-3',
    question:
      "Your production model's monitoring system tracks prediction confidence scores. You notice the average confidence has gradually increased from 0.75 to 0.90 over three months, but accuracy has remained constant. What does this most likely indicate?",
    options: [
      'The model is improving and becoming more certain about predictions',
      'The model is experiencing overconfidence, possibly due to data drift or calibration decay',
      'The input data has become easier to classify',
      'This is normal behavior and requires no action',
    ],
    correctAnswer: 1,
    explanation:
      "Increasing confidence without accuracy improvement suggests overconfidence or calibration decay. This can occur due to data drift (model encounters training-like patterns more often), changes in feature preprocessing, or population changes. Overconfident models are problematic for decision-making, especially when confidence thresholds are used. The model isn't actually improving (option A) since accuracy is flat. While data could be easier (option C), this should show in accuracy improvements. This pattern warrants investigation (option D is incorrect).",
    difficulty: 'advanced',
    topic: 'Model Monitoring',
  },
  {
    id: 'mm-mc-4',
    question:
      "You're implementing monitoring for a real-time bidding model that makes 100,000 predictions per second. Logging all predictions for monitoring would create significant storage and processing costs. What is the most effective sampling strategy?",
    options: [
      'Random sampling: log 1% of predictions uniformly',
      'Stratified sampling: ensure representation across key segments (time, user type, bid range)',
      'Reservoir sampling: maintain a fixed-size sample representing the stream',
      'Importance sampling: log predictions near decision boundaries or with extreme confidence',
    ],
    correctAnswer: 3,
    explanation:
      'Importance sampling focuses on predictions that are most informative for monitoring: borderline cases (near decision boundaries) are where drift or issues first appear, and extreme confidence cases may indicate overconfidence or outliers. This provides better drift detection per logged sample. Random sampling (option A) is simple but misses important cases. Stratified sampling (option B) helps but requires predefined strata and may miss boundary cases. Reservoir sampling (option C) maintains stream representation but treats all samples equally, missing critical cases.',
    difficulty: 'advanced',
    topic: 'Model Monitoring',
  },
  {
    id: 'mm-mc-5',
    question:
      'Your model monitoring detects a sudden 20% drop in prediction volume during business hours. Model latency metrics appear normal. Which monitoring approach would most quickly identify the root cause?',
    options: [
      'Check model performance metrics to see if accuracy dropped',
      'Investigate upstream data pipeline health and API error rates',
      'Review recent model deployments for potential bugs',
      'Analyze feature distributions for sudden drift',
    ],
    correctAnswer: 1,
    explanation:
      "A sudden drop in prediction volume with normal latency suggests an upstream issue: requests aren't reaching the model service. Check API gateway errors, data pipeline failures, service mesh issues, or client-side problems. This is the fastest way to identify the cause. Model performance (option A) is irrelevant if requests aren't arriving. Recent deployments (option C) could be the cause but checking upstream health is faster and more direct. Feature drift (option D) wouldn't cause volume drops—requests would still arrive.",
    difficulty: 'intermediate',
    topic: 'Model Monitoring',
  },
];
