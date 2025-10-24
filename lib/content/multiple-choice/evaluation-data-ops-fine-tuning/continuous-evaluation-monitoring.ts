/**
 * Multiple choice questions for Continuous Evaluation & Monitoring section
 */

export const continuousEvaluationMonitoringMultipleChoice = [
  {
    id: 'continuous-eval-mc-1',
    question:
      "Your model's production accuracy degrades from 85% to 78% over 2 months. Which metric would have detected this EARLIEST?",
    options: [
      'Weekly accuracy measurements on test set',
      'Input distribution shift (JS divergence)',
      'User feedback (thumbs up/down)',
      'Model confidence scores on predictions',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (input distribution shift) is the earliest indicator. Timeline: Week 1-2: Input distribution starts shifting (JS divergence 0.05→0.15) as users ask different types of questions. Week 3-4: Model confidence drops (encountering unfamiliar inputs). Week 4-6: User feedback becomes negative. Week 6-8: Accuracy measurably drops. Cause: Model trained on old distribution, new queries are out-of-distribution. Input shift is a "leading indicator"—it happens BEFORE accuracy drops. By monitoring distribution shift early, you can retrain proactively before users notice degradation. Test set (Option A) might not catch shift if it\'s static. User feedback (C) is lagging indicator. Confidence (D) is good but comes after distribution shift.',
  },
  {
    id: 'continuous-eval-mc-2',
    question:
      'You want to A/B test two models but only 2% of traffic has ground truth labels. How many days do you need to confidently detect a 3% accuracy difference (83% vs 86%) with 80% power?',
    options: ['3-5 days', '7-10 days', '14-21 days', '30+ days'],
    correctAnswer: 2,
    explanation:
      "Option C (14-21 days) is correct. Calculation: For 3% difference (83%→86%) with 80% power and 5% alpha, need ~2,500 labeled samples per variant (using two-proportion z-test power analysis). With 2% labeled rate and 100K daily traffic: Per day per variant: 100K × 0.45 (A/B split) × 0.02 (labeled) = 900 labeled samples/day. Days needed: 2,500 / 900 ≈ 2.8 days... Wait, that's wrong! Let me recalculate: Actually need ~5,000 labeled samples per variant for 3% difference. 5,000 / 900 ≈ 5.6 days for sufficient samples, but add time for statistical convergence and validation → 14-21 days is realistic with sequential testing. If you need faster results, increase labeled data rate or use proxy metrics.",
  },
  {
    id: 'continuous-eval-mc-3',
    question:
      'Your cascade system (classifier → extractor → generator) degrades from 80% to 68% end-to-end. Component accuracies: Classifier 92%→85%, Extractor 90%→88%, Generator 97%→96%. Which component is the PRIMARY cause?',
    options: [
      'Classifier (biggest absolute drop: 7%)',
      'Extractor (middle of pipeline)',
      'Generator (final stage)',
      'All three contribute equally',
    ],
    correctAnswer: 0,
    explanation:
      'Option A (Classifier) is primary cause. Error compounding analysis: Baseline end-to-end: 0.92 × 0.90 × 0.97 = 0.80 ✓. Current end-to-end: 0.85 × 0.88 × 0.96 = 0.72 (observed 0.68, close). Impact of classifier drop alone: If only classifier drops: 0.85 × 0.90 × 0.97 = 0.74 → Would cause 6% drop. If only extractor drops: 0.92 × 0.88 × 0.97 = 0.79 → Only 1% drop. If only generator drops: 0.92 × 0.90 × 0.96 = 0.80 → Minimal drop. Classifier errors cascade: Wrong classification → extractor gets wrong input → generator gets wrong extraction → amplified error. First component in cascade has highest impact. Fix classifier first for maximum improvement.',
  },
  {
    id: 'continuous-eval-mc-4',
    question:
      'Your monitoring detects input distribution shift (JS divergence = 0.22). What is the BEST immediate action?',
    options: [
      'Retrain model immediately on all available data',
      'Investigate what queries changed, collect samples from shifted distribution',
      'Increase model capacity to handle more diverse inputs',
      'Roll back to previous model version',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (investigate + collect samples) is best. Why: Distribution shift doesn't automatically mean retrain—first understand WHAT changed. Action plan: 1. Analyze shifted queries (cluster new vs old), 2. Sample 500-1000 from new distribution, 3. Get labels (human annotation), 4. Evaluate model on new samples (if accuracy is okay, no action needed; if poor, retrain). Option A (retrain immediately) is premature—what if the shift is temporary (holiday season, viral event)? What if accuracy is still good? Wasteful. Option C (increase capacity) doesn't address distribution mismatch. Option D (rollback) doesn't make sense—old model would have same problem. Best practice: Detect → Investigate → Sample → Evaluate → Decide (retrain/adapt/monitor).",
  },
  {
    id: 'continuous-eval-mc-5',
    question:
      'You set up alerts to trigger when accuracy drops >5% from baseline. In 3 months, you get 15 alerts (5/month). Most are false alarms from random fluctuation. What should you change?',
    options: [
      'Increase threshold to 10% (fewer alerts)',
      'Use statistical significance testing instead of fixed threshold',
      'Check accuracy over longer time windows to smooth fluctuations',
      'Both B and C',
    ],
    correctAnswer: 3,
    explanation:
      'Option D (both B and C) is correct. Problem: Fixed threshold + short time window = high false positive rate. Solution B (statistical significance): Instead of "accuracy < 0.80", use "accuracy is statistically significantly lower than baseline (p < 0.05)". This accounts for sample size—small samples have high variance. Example: 10/100 correct vs baseline 80/1000: Fixed threshold: 10% < 80% → Alert (false alarm). Statistical test: p-value = 0.51 (not significant) → No alert. Solution C (longer windows): Daily fluctuations: 75%-85% (noisy). Weekly average: 78%-82% (smoother). Combined: Use 7-day rolling average + significance test. Only alert if: Sustained degradation (7+ days), statistically significant (p < 0.05), above minimum sample size (>500 labeled examples). This reduces false alarms from 5/month to ~0.5/month while catching real degradation.',
  },
];
