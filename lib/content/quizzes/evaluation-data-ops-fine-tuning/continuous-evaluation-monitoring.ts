/**
 * Discussion questions for Continuous Evaluation & Monitoring section
 */

export const continuousEvaluationMonitoringQuiz = [
  {
    id: 'continuous-eval-q-1',
    question:
      'Your production AI system started at 88% accuracy (month 0) and has gradually degraded to 79% (month 6). Design a comprehensive monitoring system that: (1) Detects degradation early (within 1-2 weeks), (2) Diagnoses root causes automatically, (3) Triggers appropriate responses. What metrics would you track, what thresholds would trigger alerts, and what automated remediation would you implement?',
    hint: 'Consider model drift, data drift, infrastructure issues, and feedback loops. Think about leading indicators that degrade before accuracy drops.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Track multiple signals: Accuracy, latency, error rate, input/prediction drift, user feedback, confidence scores',
      'Alert thresholds: Accuracy drop >3%, latency increase >30%, input drift JS >0.15, negative feedback >10%',
      'Diagnose patterns: Accuracy↓ + input drift → data shift, latency↑ → infrastructure, feature drift → pipeline issue',
      'Automated remediation: Data shift → collect samples + retrain, infrastructure → autoscale, uncertainty → human-in-loop',
      'Early detection: Leading indicators (input drift, low confidence) appear weeks before accuracy drops',
      'Monitoring cycle: Collect (hourly) → Detect → Diagnose → Alert → Remediate → Log',
      'Expected improvement: Detect degradation in 2-3 weeks vs 6 months without monitoring',
    ],
  },
  {
    id: 'continuous-eval-q-2',
    question:
      "You want to implement A/B testing for your AI model: Model A (current, 84% accuracy) vs Model B (new, 87% accuracy on test set). However, you can only measure accuracy on 2% of production traffic (where you have ground truth labels). Design an A/B test that: (1) Confidently determines which model is better within 2 weeks, (2) Minimizes risk of deploying worse model, (3) Uses proxy metrics when labels aren't available.",
    hint: 'Consider statistical power calculations, sequential testing, proxy metrics (latency, user feedback, confidence scores), and early stopping rules.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Sample size: Need ~2,000 labeled samples per variant to detect 2% difference (84%→87%) with 80% power',
      'Timeline: With 2% labeled rate and 100K daily traffic, need ~11 days to reach significance',
      'Safety: Check daily, stop immediately if new model >5% worse (safety threshold)',
      'Traffic split: 45% Model A, 45% Model B, 10% holdout (safety)',
      'Sequential testing: Check for significance daily, stop early if reached (save time)',
      'Proxy metrics: Latency, confidence scores, user feedback when labels unavailable',
      'Decision rules: Deploy B if significant (p<0.05), rollback if worse, keep A if inconclusive after 14 days',
    ],
  },
  {
    id: 'continuous-eval-q-3',
    question:
      'Your AI system uses a cascade of models: (1) Intent classifier → (2) Entity extractor → (3) Response generator. Overall accuracy degrades from 82% to 71% over 3 months. How do you attribute the degradation to specific components, and what monitoring would prevent this?',
    hint: "Think about component-level evaluation, error propagation analysis, and independent monitoring of each stage. One component's errors can cascade.",
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Component-level evaluation: Measure intent, entity, response accuracy independently (not just end-to-end)',
      'Oracle evaluation: Test entity extractor with ground truth intent, response with ground truth entities',
      'Error attribution: 62% errors from intent, 25% from entity, 13% from response (identifies bottleneck)',
      'Error compounding: Intent 92%→84% (-8%), cascades to end-to-end 82%→71% (-11%)',
      'Diagnosis: Intent classifier primary bottleneck (degraded 8%), entity (2%), response (1%)',
      'Prevention: Monitor each component hourly, alert if drop >5%, use proxy metrics (confidence scores)',
      'Early detection: Week 2 (confidence drop) vs Month 3 (end-to-end degradation) without monitoring',
    ],
  },
];
