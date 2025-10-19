/**
 * Quiz questions for Statistical Inference section
 */

export const statisticalinferenceQuiz = [
  {
    id: 'q1',
    question:
      'Explain why the standard error decreases as sample size increases, and why this relationship is √n rather than linear. What are the practical implications for machine learning projects?',
    hint: 'Think about how variance combines and the cost/benefit of collecting more data.',
    sampleAnswer:
      "The standard error formula SE = σ/√n shows that standard error decreases with the square root of sample size, not linearly. Why √n? When we average n independent observations, their variances add (not their standard deviations), and Var(mean) = σ²/n, so SE = √(σ²/n) = σ/√n. This has crucial ML implications: (1) **Diminishing returns**: To halve the standard error (double precision), you need 4x the data. To reduce it by 10x, you need 100x the data. (2) **Cost-benefit analysis**: Early data collection gives huge improvements, but eventually becomes expensive for small gains. (3) **Practical strategy**: Start with moderate sample (e.g., 1000 examples), evaluate performance CI. If CI is too wide, calculate how much more data needed. For example, if SE = 0.02 with n=1000, getting SE = 0.01 requires n=4000 (3000 more samples). (4) **Model selection**: With limited data, simpler models with fewer parameters will have tighter CIs. (5) **Active learning**: Instead of blind data collection, select most informative samples. The √n relationship means there's always a point where collecting more data isn't worth the cost - know when to stop!",
    keyPoints: [
      'SE = σ/√n because variances add, not standard deviations',
      'To halve SE, need 4x data (quadratic relationship)',
      'Diminishing returns: early data most valuable',
      'Calculate required sample size for target precision',
      'Balance data collection cost vs. performance gain',
      'Consider active learning for efficient data use',
    ],
  },
  {
    id: 'q2',
    question:
      'A 95% confidence interval for your model\'s accuracy is [0.78, 0.82]. Your stakeholder asks: "So the model is definitely between 78% and 82% accurate?" How do you correctly explain what this confidence interval means?',
    hint: 'The common misinterpretation is that the true value is "probably" in the interval.',
    sampleAnswer:
      'This is a critical misunderstanding to correct! The correct interpretation: "If we repeated this entire process (sampling data, training model, computing CI) many times, approximately 95% of the resulting CIs would contain the true accuracy." What it does NOT mean: (1) ✗ "The true accuracy has a 95% probability of being in [0.78, 0.82]" - The true accuracy is fixed (not random); the interval is what varies. (2) ✗ "There\'s a 95% chance our model\'s accuracy is in this range" - Again, implies the parameter is random. (3) ✗ "95% of predictions will fall in this range" - This confuses accuracy intervals with prediction intervals. Practical explanation for stakeholders: "Using our test data, we estimate 80% accuracy. However, because we tested on limited data (not all possible future cases), there\'s uncertainty. Based on our sample size, the true long-term accuracy is very likely between 78% and 82%. The 95% reflects our confidence level - we\'d be wrong about 1 in 20 times if we make many such statements." Better yet, use Bayesian interpretation with posterior probability if appropriate: "Given our data, we\'re 95% confident the true accuracy is in [0.78, 0.82]" - though this requires different methodology (Bayesian credible intervals, not frequentist CIs).',
    keyPoints: [
      'CI refers to the procedure, not this specific interval',
      '95% of CIs (from repeated sampling) contain true value',
      'True parameter is fixed, not random',
      'Do NOT say "95% probability the true value is in the interval"',
      'Explain as: "Very likely between 78-82% based on our sample"',
      'Bayesian credible intervals allow probability statements',
    ],
  },
  {
    id: 'q3',
    question:
      'You have two options: (A) Train on 1000 samples and evaluate on 200 samples, or (B) Train on 800 samples and evaluate on 400 samples. Which gives a better estimate of true model performance, and why? What factors should influence this decision?',
    hint: 'Consider both the model quality and the precision of the performance estimate.',
    sampleAnswer:
      'This involves a classic bias-variance tradeoff in evaluation: **Option A (1000/200)**: Pros: (1) Better model - trained on more data, likely higher true performance. (2) Less optimistic bias in performance estimate. Cons: (1) Wider CI for test accuracy due to smaller test set (SE = σ/√200 ≈ 0.07σ). (2) Less reliable estimate of true performance. **Option B (800/400)**: Pros: (1) Narrower CI for test accuracy (SE = σ/√400 = 0.05σ), 30% tighter. (2) More reliable performance estimate. Cons: (1) Slightly worse model due to less training data. (2) Small upward bias (model would be better with more training data). **Recommendation**: Choose based on priority: (1) If goal is final deployed model: Use A, train on as much as possible (1000+200=1200 after validation). (2) If goal is accurate performance estimate: Use B, narrower CI matters for decision-making. (3) Best practice: Use cross-validation! K-fold CV trains on K-1/K of data each fold, uses all data for both training and testing. With 5-fold CV on 1200 samples, train on 960, test on 240 each fold - best of both worlds. (4) Very large datasets: Use 80/20 split (plenty of data for both). (5) Small datasets (<1000): Use LOOCV or bootstrap for stable estimates. Real answer: Cross-validation dominates static splits for most ML applications, combining good training and reliable evaluation.',
    keyPoints: [
      'Larger training set → better model',
      'Larger test set → narrower CI, more reliable estimate',
      'SE reduces as √test_size',
      'Cross-validation uses all data for both purposes',
      'For small data, prefer cross-validation over single split',
      'For large data (>10K), 80/20 split is fine',
    ],
  },
];
