/**
 * Multiple choice questions for AI Evaluation Fundamentals section
 */

export const aiEvaluationFundamentalsMultipleChoice = [
  {
    id: 'eval-fund-mc-1',
    question:
      'Your company has a text classification model with 95% accuracy on the test set, but after deployment, users report it fails frequently. Investigation shows the test set was randomly sampled from training data collected 2 years ago. What is the MOST likely root cause?',
    options: [
      'The model is overfitting to the test set',
      'Data distribution has shifted since collection (concept drift)',
      'Users are intentionally trying to break the system',
      '95% accuracy is too low for production deployment',
    ],
    correctAnswer: 1,
    explanation:
      'Data distribution shift (concept drift) is the most likely cause. The test set is 2 years old and randomly sampled from training data, meaning both train and test come from the same old distribution. Production data represents current reality, which has likely shifted. For example, if classifying news articles, topics/language evolve over time. The model performs well on old data (95% test accuracy) but poorly on new distribution. Option A (overfitting) wouldn\'t cause the specific pattern of "good test, bad production." Option C is possible but unlikely as primary cause. Option D is wrong—95% is often acceptable; the issue is distribution mismatch, not accuracy level.',
  },
  {
    id: 'eval-fund-mc-2',
    question:
      'You want to evaluate a question-answering system that generates free-form text answers. Which evaluation approach would be MOST appropriate?',
    options: [
      'Exact string matching against reference answers',
      'Combination of semantic similarity, LLM-as-judge, and human evaluation on sample',
      "Only BLEU score since it's text generation",
      'Count how many words in the answer appear in the reference',
    ],
    correctAnswer: 1,
    explanation:
      "Combination approach (Option B) is best for free-form QA. Exact matching (A) fails because multiple valid phrasings exist for correct answers. BLEU (C) is designed for translation, not QA, and doesn't capture correctness. Word overlap (D) is too simplistic—answer could contain right words in wrong way. The hybrid approach works: (1) Semantic similarity catches paraphrases, (2) LLM-as-judge evaluates correctness/relevance at scale, (3) Human evaluation on sample validates automated metrics and catches edge cases. This balances thoroughness with cost/speed.",
  },
  {
    id: 'eval-fund-mc-3',
    question:
      'Your offline evaluation shows Model A (92% accuracy) outperforms Model B (89% accuracy). However, after A/B testing in production, users prefer Model B (4.2/5 vs 3.8/5 satisfaction). What should you do?',
    options: [
      'Deploy Model A because offline metrics are more reliable than subjective user ratings',
      'Investigate the discrepancy and consider deploying Model B since user satisfaction is the ultimate goal',
      'Average the scores: (0.92+3.8)/2 = 2.36 for A, (0.89+4.2)/2 = 2.55 for B, so deploy B',
      'Run more offline tests until they match user preferences',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. Offline metrics are proxies for user satisfaction, not the goal itself. When proxies disagree with real outcomes, trust real outcomes. Model A may achieve higher accuracy by being conservative (fewer errors but also less helpful), while Model B might take more risks that users appreciate. You should: (1) Deploy Model B (users prefer it), (2) Investigate why offline metrics disagree (perhaps accuracy doesn't capture helpfulness, engagement, or other factors users care about), (3) Update offline evaluation to better predict user satisfaction. Option A ignores the ultimate goal. Option C is nonsensical (can't average accuracy % with rating /5). Option D won't help—the issue is that offline tests measure the wrong thing.",
  },
  {
    id: 'eval-fund-mc-4',
    question:
      'You have 10,000 examples to evaluate your model. Running all evaluations would take 5 hours and cost $500. You need results quickly. What is the BEST strategy?',
    options: [
      'Skip evaluation entirely to save time and money',
      'Evaluate only the first 100 examples to get quick results',
      'Use stratified sampling to select 500 representative examples covering all categories',
      'Randomly select 500 examples and evaluate those',
    ],
    correctAnswer: 2,
    explanation:
      "Stratified sampling (Option C) is best. It ensures your sample represents all categories/scenarios in proportion to their frequency. For example, if your data has 70% easy, 20% medium, 10% hard cases, sample 350 easy, 100 medium, 50 hard from the 500 total. This gives accurate estimates of overall performance while being 20x faster/cheaper. Random sampling (D) works but might miss rare important categories by chance. First 100 (B) may not be representative if data is ordered. Skipping evaluation (A) is dangerous—you'd deploy blind. With stratified sampling, you get 95% confidence intervals on your metrics at 5% of the cost.",
  },
  {
    id: 'eval-fund-mc-5',
    question:
      'Your summarization model achieves ROUGE-1: 0.45, ROUGE-2: 0.23, ROUGE-L: 0.41 on your test set. A colleague says "these scores are terrible, we need 0.90+ to deploy." What is the MOST accurate response?',
    options: [
      'Agree—ROUGE scores below 0.9 indicate the model is not production-ready',
      'ROUGE scores are relative and depend on the task; need to compare to baselines and human performance',
      'ROUGE is deprecated, so these scores are meaningless',
      'The model needs exactly 0.75 ROUGE to be production-ready (industry standard)',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. ROUGE scores are not on an absolute scale like accuracy. What constitutes "good" depends on: (1) Task difficulty (news articles vs technical docs), (2) Reference summaries (how much variation exists in good summaries), (3) Baseline performance (extractive baselines often get 0.35-0.50), (4) Human agreement (humans summarizing the same text often have ROUGE ~0.55-0.65 with each other). ROUGE scores of 0.45/0.23/0.41 might be excellent if they beat baselines and approach human-level performance. Never evaluate metrics in isolation—always need context. Option A/D apply arbitrary thresholds without context. Option C is wrong—ROUGE is still widely used for summarization evaluation.',
  },
];
