/**
 * Quiz questions for AI Evaluation Fundamentals section
 */

export const aiEvaluationFundamentalsQuiz = [
  {
    id: 'eval-fund-q-1',
    question:
      'You have a chatbot deployed to production serving 100K users daily. Performance seems fine, but you have no formal evaluation process. Design a comprehensive evaluation strategy that balances cost, speed, and thoroughness. What mix of offline evaluation, online monitoring, and human evaluation would you implement?',
    hint: 'Consider the trade-offs between automated metrics (cheap, fast) and human evaluation (expensive, thorough). Think about what to measure continuously vs periodically.',
    sampleAnswer:
      'A comprehensive evaluation strategy would include: (1) Offline Evaluation before each deployment with automated test sets and metrics, (2) Continuous Online Monitoring tracking all production requests with sampled deep evaluation, and (3) Weekly Human Evaluation for quality assessment plus monthly deep dives for improvements. This balanced approach ensures comprehensive coverage while managing costs around $150/day.',
    keyPoints: [
      'Offline evaluation gates deployment (test set + automated metrics)',
      'Continuous monitoring tracks all requests with sample-based deep evaluation',
      'Human evaluation weekly for quality + monthly deep dives for improvements',
      'Balance cost (~$150/day) with comprehensive coverage across all evaluation types',
    ],
  },
  {
    id: 'eval-fund-q-2',
    question:
      'Your LLM-powered code generation tool has 85% accuracy on your test set, but users complain it "doesn\'t work well." Debugging reveals the test set focuses on simple functions, but users ask for complex multi-file changes. How do you fix your evaluation to better reflect real usage?',
    hint: "The test set doesn't match production distribution. Consider how to make evaluation more representative.",
    sampleAnswer:
      "**Root Cause:** Test set has distribution mismatch—optimizing for simple cases while users need complex scenarios. **Solution:** (1) **Collect Real Usage Data:** Sample 500 actual user requests from production logs. Analyze distribution: 15% simple functions, 40% multi-file changes, 25% refactoring, 20% debugging. (2) **Build Representative Test Set:** Create new test set matching real distribution: 75 simple (15%), 200 multi-file (40%), 125 refactoring (25%), 100 debugging (20%). Total: 500 examples. (3) **Update Metrics:** Add task-specific metrics: For simple functions: exact match, For multi-file: all files modified correctly, For refactoring: maintains behavior + improves code, For debugging: fixes bug without introducing new ones. (4) **Continuous Alignment:** Monthly: Compare test set distribution vs production, Add new edge cases found in production, Remove stale examples. (5) **Weighted Scoring:** Weight test set scores by production frequency: Final Score = 0.15 × simple + 0.40 × multi_file + 0.25 × refactoring + 0.20 × debugging. **Result:** Model scores drop to 62% on new test set (more realistic), but optimization now focuses on what users actually need. After improvements targeting multi-file changes, test score reaches 75% AND user satisfaction improves. **Lesson:** Your test set must mirror production distribution, or you'll optimize for the wrong thing.",
    keyPoints: [
      'Test set had distribution mismatch (simple vs complex tasks)',
      'Collect real usage data to understand actual user needs',
      'Build representative test set matching production distribution',
      'Weight metrics by task frequency to reflect real-world impact',
    ],
  },
  {
    id: 'eval-fund-q-3',
    question:
      "You're evaluating two summarization models: Model A scores 0.85 on ROUGE but users rate it 3.2/5. Model B scores 0.78 on ROUGE but users rate it 4.5/5. Model B is also 2x more expensive. Which should you deploy and why? How do you reconcile the metric-human disagreement?",
    hint: "Automated metrics don't always correlate with human preference. Consider what matters for your use case and investigate the disagreement.",
    sampleAnswer:
      "**Analysis:** Model B is better despite lower ROUGE score and higher cost. Here's why: **Investigating the Disagreement:** (1) ROUGE measures n-gram overlap with reference summaries. High ROUGE = close to reference. (2) But reference summaries may not be ideal! Users might prefer different styles, lengths, or focus. (3) Model A likely mimics reference style (high ROUGE) but isn't as helpful (low user rating). Model B likely: More concise/readable (users prefer), Better key point selection, More natural language. **Decision Framework:** (1) **Primary Metric = Business Goal:** Goal: User satisfaction → Use Model B. User ratings (4.5 vs 3.2) directly measure success. (2) **Cost Analysis:** Model B costs 2x but delivers 40% better satisfaction (4.5 vs 3.2). Calculate: Is 40% better UX worth 2x cost? For premium product: YES. For free product with thin margins: Maybe not. (3) **Hybrid Approach (Best Solution):** Deploy Model B for quality. Use Model B outputs to create better reference summaries. Retrain Model A to match Model B style (learn what humans prefer). Re-evaluate: Model A might achieve 0.82 ROUGE + 4.3 rating at 1x cost. **Recommendation:** (1) Short term: Deploy Model B (user satisfaction matters most). (2) Medium term: Use Model B to improve Model A. (3) Long term: Update evaluation to weight human feedback heavily. Add metrics that correlate with user satisfaction (readability, key point coverage). **Key Lesson:** Automated metrics are proxies. When they disagree with human judgment, trust humans and investigate why.",
    keyPoints: [
      'User satisfaction (4.5 vs 3.2) outweighs automated metric (ROUGE) when they disagree',
      'ROUGE measures overlap with references, but references may not be ideal',
      'Decision depends on business model (premium vs free, margins)',
      'Long-term: improve cheaper model by learning from expensive one, update metrics',
    ],
  },
];
