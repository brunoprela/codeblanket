/**
 * Quiz questions for Evaluation Datasets & Benchmarks section
 */

export const evaluationDatasetsBenchmarksQuiz = [
  {
    id: 'eval-datasets-q-1',
    question:
      'Your 1000-example test set shows your model achieves 88% accuracy. After deployment, users report many failures. Investigation reveals your test set has 80% "easy" examples but production queries are 60% "hard". Redesign your test set strategy to prevent this mismatch.',
    hint: 'Test set must reflect production distribution across difficulty levels.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Test set distribution must match production (not 80% easy when prod is 60% hard)',
      'Use stratified sampling to build representative test set',
      'Report metrics per difficulty level to identify weak spots',
      'Weight scores by production frequency for realistic overall metric',
    ],
  },
  {
    id: 'eval-datasets-q-2',
    question:
      'You need to create a 500-example evaluation dataset for a new task where no labeled data exists. You have budget for either: (A) 500 expert-labeled examples ($5000), or (B) 1000 crowd-labeled examples ($2500) with quality control. Which would you choose and why? Design a hybrid strategy if possible.',
    hint: 'Consider quality vs quantity trade-off and how to maximize value.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Hybrid strategy combining expert labeling and crowd labeling maximizes value',
      'Use small expert set as golden standard for quality control',
      'Crowd labor with quality control (3x redundancy + golden set validation)',
      'Expert review disagreements to improve quality where needed',
    ],
  },
  {
    id: 'eval-datasets-q-3',
    question:
      'You maintain a test set that was created 18 months ago. Model performance on this test set is stable at 85%, but production complaints are increasing. Explain why test sets become stale and design a maintenance strategy to keep your evaluation relevant.',
    hint: 'Test sets decay over time as (1) real distribution shifts, (2) models optimize for known test, (3) new edge cases emerge.',
    sampleAnswer:
      "**Why Test Sets Go Stale:** (1) **Distribution Drift:** Language/user behavior evolves (new slang, topics, use patterns), Test set frozen at t=0, production is t=now. (2) **Test Set Overfitting:** Model implicitly optimized for this specific test set over 18 months, May memorize patterns, doesn't reflect true capability. (3) ** New Edge Cases:** Users discover failure modes not in original test set, Test set missing recent edge cases. (4) ** Changing Requirements:** Product evolved(new features, use cases), Test set reflects old requirements. ** Test Set Maintenance Strategy:** (1) ** Continuous Monitoring(Monthly):** Sample 50 production examples monthly, Compare production distribution vs test set distribution(topics, difficulty, input types), Flag significant drift(> 15 % distribution change). (2) ** Regular Refresh(Quarterly):** Add 100 new examples from production(recent quarter), Remove 100 oldest / stalest examples, Keep core test set(gold standard examples) permanently. (3) ** Version Control:** Test Set v1.0(Jan 2023 - original), v1.1(Apr 2023 - added edge cases), v2.0(Oct 2023 - major refresh), Track model performance across versions. (4) ** Holdout Set Rotation:** Maintain 3 test sets: Active(used for dev), Holdout 1(quarterly validation), Holdout 2(final gate before deploy).Rotate: Every 6 months, Active becomes retired, Holdout 1 becomes Active, Create new Holdout 2.(5) ** Edge Case Mining:** Monitor production errors, Add failed examples to test set(after fixing), Ensure future models don't repeat mistakes. **Implementation Schedule:** Monthly: Distribution monitoring, Quarterly: Add 100 new + remove 100 old, Bi-annually: Holdout rotation, Annually: Major test set refresh (20-30% turnover). **Result:** Test set stays aligned with production, catches regressions early.",
    keyPoints: [
      'Test sets go stale due to distribution drift, implicit overfitting, and new edge cases',
      'Implement continuous monitoring of production vs test distribution',
      'Regular refresh: quarterly add new examples, remove stale ones',
      'Version control test sets and rotate holdout sets to prevent overfitting',
    ],
  },
];
