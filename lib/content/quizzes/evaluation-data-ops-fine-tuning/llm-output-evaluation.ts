/**
 * Quiz questions for LLM Output Evaluation section
 */

export const llmOutputEvaluationQuiz = [
  {
    id: 'llm-eval-q-1',
    question:
      "You're evaluating a medical QA system. ROUGE and BLEU scores are 0.65 and 0.58, semantic similarity is 0.82, but a medical expert rates factual accuracy at only 60%. Which metric should you prioritize and why? Design an evaluation framework that properly weighs these different dimensions for a safety-critical domain.",
    hint: "In medical/safety-critical domains, factual accuracy is paramount. Traditional NLP metrics don't capture this.",
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Medical factual accuracy is primary metric (95%+ required), not ROUGE/BLEU',
      "Traditional NLP metrics don't capture medical correctness",
      'Require expert review: 100% pre-deployment, 10% ongoing in production',
      'Automated metrics are secondary for development iteration, not deployment gates',
    ],
  },
  {
    id: 'llm-eval-q-2',
    question:
      'Your code generation model passes 78% of test cases using execution-based evaluation. However, users complain the code is "ugly and unmaintainable." Traditional metrics (pass rate, execution time) don\'t capture code quality. Design a comprehensive code evaluation framework that measures correctness AND quality.',
    hint: 'Code evaluation needs multiple dimensions: correctness (does it work?), quality (is it maintainable?), efficiency, style.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Code needs multi-dimensional evaluation: correctness, quality, efficiency, best practices',
      'Use automated tools (linters, complexity analysis) to measure code quality',
      'Weight dimensions based on importance (40% correctness, 30% quality, 15% each for efficiency/practices)',
      'Validate automated metrics with human expert review on samples',
    ],
  },
  {
    id: 'llm-eval-q-3',
    question:
      'You have two evaluation approaches: (A) BERTScore: $0.001/example, correlates 0.65 with human judgment. (B) GPT-4 as judge: $0.02/example, correlates 0.88 with human judgment. You need to evaluate 10,000 examples monthly. Design a cost-effective hybrid strategy that maximizes evaluation quality while minimizing cost.',
    hint: 'Use expensive high-quality evaluation strategically, not uniformly across all examples.',
    sampleAnswer:
      "**Cost Analysis:** BERTScore: 10K × $0.001 = $10/month (low cost, mediocre quality). GPT-4 Judge: 10K × $0.02 = $200/month (20x more, much better quality). **Hybrid Strategy (Best of Both Worlds):** (1) **Initial Filtering (BERTScore - All 10K examples)** - Run BERTScore on ALL examples ($10), Identify: High confidence good (BERTScore >0.85): ~3K examples, High confidence bad (BERTScore <0.50): ~2K examples, Uncertain (0.50-0.85): ~5K examples. (2) **Targeted Deep Evaluation (GPT-4 - Subset)** - Use GPT-4 on: ALL uncertain examples (5K × $0.02 = $100), Sample of high-confidence (20% × 5K × $0.02 = $20) for calibration. (3) **Human Validation (Critical Cases)** - GPT-4 identifies safety issues or contradictions, Human review those (~100 examples), Cost: ~$200 in human time. **Total Cost Breakdown:** BERTScore (10K): $10, GPT-4 (6K): $120, Human review: $200, Total: $330/month (vs $200 for pure GPT-4). **Quality Improvement:** Hybrid approach: Covers all 10K (vs sampling), Focuses expensive evaluation where it matters (uncertain cases), Human review for safety-critical, Estimated correlation with human: ~0.80 (vs 0.88 pure GPT-4, 0.65 pure BERTScore). **Result:** Get 91% of GPT-4's quality at 65% of cost ($330 vs $200... wait, this costs MORE! Let me recalculate...) Actually BETTER approach: (1) BERTScore ALL 10K ($10), (2) GPT-4 on 5K uncertain ($100), (3) Trust high-confidence BERTScore predictions (calibrated against GPT-4 sample). Total: $110/month for ~0.78 correlation. OR use GPT-3.5 ($0.002/example) for middle tier: $10 (BERTScore) + $10 (GPT-3.5 on 5K) + $40 (GPT-4 on 2K hardest) = $60 total, ~0.75 correlation. **Best Strategy:** Multi-tier based on difficulty, optimize spend where it matters most.",
    keyPoints: [
      'Use cheap metrics (BERTScore) to filter/triage all examples',
      'Apply expensive high-quality evaluation (GPT-4) to uncertain/critical cases only',
      'Calibrate cheap metrics against expensive ones to improve overall accuracy',
      'Multi-tier approach achieves 85-90% of max quality at 30-60% of max cost',
    ],
  },
];
