/**
 * Multiple choice questions for Human Evaluation & Feedback section
 */

export const humanEvaluationFeedbackMultipleChoice = [
  {
    id: 'human-eval-mc-1',
    question:
      'You have 1000 AI responses to evaluate. Annotators work at 20 responses/hour and cost $15/hour. You need each response rated by 3 annotators. What is the minimum cost and time (with 5 annotators working in parallel)?',
    options: [
      'Cost: $750, Time: 10 hours',
      'Cost: $2,250, Time: 30 hours',
      'Cost: $2,250, Time: 6 hours',
      'Cost: $1,125, Time: 15 hours',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. Calculation: 1000 responses × 3 annotations = 3000 total annotations needed. At 20 responses/hour per annotator, each annotator contributes 20 annotations/hour. With 5 annotators in parallel: 5 × 20 = 100 annotations/hour. Time = 3000 annotations ÷ 100 annotations/hour = 30 hours total. But with parallel work: 30 hours ÷ 5 annotators = 6 hours of wall-clock time. Cost = 3000 annotations ÷ 20 per hour × $15/hour = 150 hours × $15 = $2,250. Key insight: Parallelization reduces time but not cost. Triple annotation is expensive but necessary for quality and measuring inter-annotator agreement.',
  },
  {
    id: 'human-eval-mc-2',
    question:
      "You measure inter-annotator agreement using these methods on the same dataset: Percent Agreement = 78%, Cohen\'s Kappa = 0.52, Krippendorff's Alpha = 0.48. Which metric should you report and why?",
    options: [
      "Percent Agreement (78%) because it's highest and easiest to understand",
      "Cohen\'s Kappa (0.52) because it's most commonly used",
      "Krippendorff's Alpha (0.48) because it's most rigorous and handles missing data",
      'All three to show different perspectives',
    ],
    correctAnswer: 2,
    explanation:
      "Option C (Krippendorff\'s Alpha) is most rigorous. Here's why metrics differ: Percent Agreement (78%) is INFLATED—it counts random chance agreements as real agreement. Example: If two annotators randomly pick from 5 categories, they'll agree 20% by pure chance. Cohen\'s Kappa (0.52) corrects for chance but only works with 2 annotators and requires all annotators to rate all examples. Krippendorff's Alpha (0.48) is most conservative: handles ≥2 annotators, handles missing data (if some annotators skip examples), works with different data types (nominal, ordinal, interval, ratio). The big gap (78% → 48%) shows much of the agreement is just chance. 0.48 means only \"moderate\" agreement. For academic rigor and edge cases (missing data, >2 annotators), report Krippendorff's Alpha. For communication to non-experts, also show percent agreement but explain the chance correction.",
  },
  {
    id: 'human-eval-mc-3',
    question:
      'Your app shows a 5-star rating request after users interact with the AI. Users who rate are 2.3x more likely to be extreme experiences (1-star or 5-star) than average users. This is called:',
    options: [
      'Sample bias',
      'Response bias / Self-selection bias',
      'Confirmation bias',
      'Survivorship bias',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (Response bias / Self-selection bias) is correct. Response bias occurs when people who choose to respond differ systematically from those who don\'t. In this case: Users with extreme experiences (delighted or frustrated) are more motivated to give feedback. The "silent middle" (satisfied but not delighted) rarely responds. This creates a bimodal distribution that doesn\'t represent true satisfaction. Sample bias (A) is similar but typically refers to how you select people to ask, not who chooses to respond. Confirmation bias (C) is seeking information that confirms existing beliefs. Survivorship bias (D) is when you only analyze survivors (e.g., successful products) and ignore failures. Solution: Use active sampling where you explicitly ask a random subset of users, not just rely on voluntary responses. Aim for 30-40% response rate on actively sampled users.',
  },
  {
    id: 'human-eval-mc-4',
    question:
      'You use crowdworkers for annotation. After qualification tests, Worker A has 85% agreement with gold standard, Worker B has 92%. How should you use their annotations?',
    options: [
      'Only use Worker B, reject Worker A (below 90% threshold)',
      "Use both, but weight Worker B's annotations higher when aggregating",
      'Use both equally, individual noise averages out with multiple annotators',
      'Retrain Worker A until they reach 90%, then use both equally',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (weighted aggregation) is best. Here\'s why: Worker A at 85% is still usable—they're right most of the time. Rejecting them wastes time and money. Using both equally (Option C) works but suboptimal—why ignore information about annotator quality? Weighted aggregation is optimal: When annotations conflict, trust Worker B more. Example: Response gets ratings [3, 5, 4] from Workers [A, B, C with quality 85%, 92%, 88%]. Weighted average: (3×0.85 + 5×0.92 + 4×0.88) / (0.85+0.92+0.88) = 4.06 vs simple average of 4.0. Implementation: Use Bayesian inference or weighted voting. Weight by: calibration accuracy, Krippendorff Alpha with gold set, time to annotate (too fast = low quality). Option D (retrain) is good long-term but doesn't solve immediate needs. Best practice: Use all qualified annotators with performance-weighted aggregation.",
  },
  {
    id: 'human-eval-mc-5',
    question:
      'For a safety-critical medical AI, you need human evaluation. Which approach provides the HIGHEST quality annotations?',
    options: [
      'Crowdworkers with medical knowledge keyword filters',
      'Medical students with 1-day training',
      'Board-certified physicians in the relevant specialty',
      'Mix of all three for diversity of perspectives',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (board-certified physicians) is unambiguous for safety-critical medical AI. Here\'s the trade-off: Crowdworkers (A) are cheap and fast but lack expertise—dangerous for medical decisions. One wrong annotation could mean a model learns to give harmful advice. Medical students (B) have foundational knowledge but lack clinical experience—they might miss nuances. Board-certified physicians (C) have expertise, clinical experience, and understand real consequences. They\'re expensive ($200-400/hour) and slow, but medical AI MUST prioritize correctness over cost. Option D (mix) is wrong for safety-critical tasks—you need consistently high expertise, not "average" of expert+novice. Exception: For non-safety aspects (UI clarity, response time), use cheaper evaluators. Best practice for medical AI: Physicians annotate safety/correctness, general annotators evaluate UX. Always: Multiple physician annotations for high-risk outputs, Ethics review for evaluation protocol.',
  },
];
