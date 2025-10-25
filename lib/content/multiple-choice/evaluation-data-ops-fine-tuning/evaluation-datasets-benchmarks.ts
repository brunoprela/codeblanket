/**
 * Multiple choice questions for Evaluation Datasets & Benchmarks section
 */

export const evaluationDatasetsBenchmarksMultipleChoice = [
  {
    id: 'eval-datasets-mc-1',
    question:
      'You have 5000 examples for your evaluation dataset. They consist of: 4000 common queries, 800 medium difficulty, 200 rare edge cases. What is the BEST sampling strategy to create a 500-example test set?',
    options: [
      'Random sampling: 500 examples chosen uniformly at random',
      'Stratified sampling: 400 common, 80 medium, 20 rare (maintaining proportions)',
      'Stratified sampling: 250 common, 150 medium, 100 rare (oversampling rare cases)',
      'Only use the 200 rare edge cases and add 300 synthetic hard examples',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (stratified with oversampling) is best. Pure proportional sampling (B) would give you only 20 rare cases—too few to reliably measure performance on edge cases that matter most. Random sampling (A) has the same problem. Oversampling rare cases ensures sufficient data to evaluate on all important scenarios. With 100 rare cases (vs 20), you can confidently measure edge case performance. You still have 250 common cases to measure general performance. Option D is extreme—you lose ability to measure common case performance. Best practice: Oversample underrepresented but important categories, then report per-category metrics separately and optionally compute a weighted average based on production frequency.',
  },
  {
    id: 'eval-datasets-mc-2',
    question:
      'Your model scores 0.72 on your internal test set but 0.85 on the public MMLU benchmark. What is the MOST likely explanation?',
    options: [
      'Your test set is too hard and should be made easier',
      'The model is overfitting to public benchmarks through indirect exposure during training',
      'MMLU is an easier benchmark than your test set',
      'There is a bug in your evaluation code',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (benchmark overfitting) is most likely. Public benchmarks like MMLU are widely known, and training data from the internet may contain information about these benchmarks, test-taking strategies, or similar questions. This leads to inflated performance on public benchmarks vs novel internal tests. This is why: (1) Leading AI labs use private evaluation sets, (2) Benchmark contamination is a known problem, (3) Internal tests on your specific domain may be genuinely harder. Option A is wrong—don't dumb down tests to match benchmarks. Option C could be true but less likely as primary cause. Option D is possible but check benchmark performance first. Best practice: Use both public benchmarks (for comparison) AND private test sets (for true capability assessment).",
  },
  {
    id: 'eval-datasets-mc-3',
    question:
      'You want to ensure your test set has no data leakage from training. Which validation method is MOST thorough?',
    options: [
      'Check that no exact string matches exist between train and test',
      'Use a hash-based deduplication to find near-duplicates',
      'Check for semantic similarity using embeddings and flag pairs above 0.9 cosine similarity',
      'All of the above: exact match + hash deduplication + semantic similarity checks',
    ],
    correctAnswer: 3,
    explanation:
      'Option D (all checks) is most thorough. Each method catches different types of leakage: (1) Exact match: Catches identical examples, (2) Hash-based (e.g., MinHash, SimHash): Catches near-duplicates with small modifications (typos, reordering), (3) Semantic similarity: Catches paraphrases and rewrites that mean the same thing. Example: "What is 2+2?" (test) and "Calculate two plus two" (train) won\'t match exactly or by hash, but have high semantic similarity. Multi-layered approach prevents all forms of leakage. In production: Run all three checks, manually review flagged pairs, remove test examples with any training similarity >0.85.',
  },
  {
    id: 'eval-datasets-mc-4',
    question:
      "Your test set was created by sampling production logs from 2022. It\'s now 2024 and model performance on this test set is stable, but production performance is degrading. What should you do?",
    options: [
      'The model is fine since test performance is stable; production issues must be from other factors',
      'Create a new test set sampled from recent 2024 production data to reflect current distribution',
      'Increase training data to improve robustness',
      'Lower your production performance targets since 2024 queries are harder',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. The 2022 test set no longer represents 2024 production distribution. Stable test performance with declining production performance indicates distribution drift. The test set has become stale and doesn't capture current user behavior, topics, or language patterns. Solution: (1) Sample new test set from 2024 production data, (2) Compare 2022 vs 2024 test performance to measure drift, (3) Update test set regularly (quarterly/annually). Option A ignores clear evidence of drift. Option C might help but doesn't address evaluation problem. Option D is wrong—users expect consistent quality regardless of year. Key lesson: Test sets decay and need maintenance.",
  },
  {
    id: 'eval-datasets-mc-5',
    question:
      "You have a test set with expected outputs written by one expert. Inter-annotator agreement with a second expert is only 70%. How should you interpret your model's 85% accuracy on this test set?",
    options: [
      '85% accuracy is excellent and deployment-ready',
      '85% accuracy may be near the ceiling since humans only agree 70% of the time; model performance is actually very good',
      'The test set is flawed and should be discarded',
      '85% accuracy is meaningless without more context',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. Human agreement is the ceiling for model performance. If two experts only agree 70% of the time, the task has inherent ambiguity. Your model at 85% might be: (1) Overfitting to first annotator's preferences, (2) Actually performing at ~60% true accuracy (85% × 70% human agreement), (3) Near maximum possible performance for this ambiguous task. To clarify: Get multiple annotations per example (e.g., 3 annotators), Use majority vote or aggregate scores, Measure inter-annotator agreement (Cohen's Kappa), Report model accuracy relative to human agreement. For tasks with low human agreement (<80%), consider: Is task well-defined? Can we improve guidelines? Should we use softer metrics (partial credit)? Option A is overconfident. Option C is wrong—test set is valuable but needs multiple annotations.",
  },
];
