/**
 * Multiple choice questions for Data Labeling & Annotation section
 */

export const dataLabelingAnnotationMultipleChoice = [
  {
    id: 'data-label-mc-1',
    question:
      'You need 10,000 labeled examples. Option A: Crowdworkers at $0.10/label, 85% accuracy. Option B: Experts at $0.50/label, 95% accuracy. Option C: Triple-annotation with crowdworkers at $0.30/label (3×$0.10), majority vote gives 93% accuracy. Which is most cost-effective?',
    options: [
      'Option A: $1,000, 85% accuracy = best cost/performance ratio',
      'Option B: $5,000, 95% accuracy = highest quality',
      'Option C: $3,000, 93% accuracy = best balance',
      'Option A, then manually review the 15% likely errors',
    ],
    correctAnswer: 2,
    explanation:
      "Option C is most cost-effective. Cost-performance analysis: A: $1,000 for 85% = $0.12 per correct label. B: $5,000 for 95% = $0.53 per correct label. C: $3,000 for 93% = $0.32 per correct label. Option C gets you to 93% accuracy (close to expert level) at 60% of expert cost. Triple annotation with majority vote is a proven technique to boost crowdworker quality. For the incremental 2% improvement (93%→95%), you'd pay an additional $2,000 ($5K - $3K). Usually not worth it unless your domain is extremely high-stakes. Option D sounds clever but: manually reviewing 1,500 examples at expert rates (~$0.40/review) adds $600, total $1,600 for ~90% accuracy. Still good, but Option C is more systematic and reaches higher accuracy.",
  },
  {
    id: 'data-label-mc-2',
    question:
      'Your annotation platform shows that Worker #147 completed 1,000 labels in 2 hours (30 seconds per label). Expected time is 2 minutes per label. Their accuracy on hidden gold standards is 55% (random is 50% for binary task). What should you do?',
    options: [
      'Pay them—they completed the work and beat random chance',
      "Reject all labels, don't pay, block the worker for low quality",
      'Investigate: Could be a bot or random clicking. Check for patterns.',
      'Pay half: They worked fast but quality is low',
    ],
    correctAnswer: 2,
    explanation:
      'Option C (investigate) is correct. Red flags: (1) 4x faster than expected (30s vs 2min), (2) Barely above random chance (55% vs 50%), (3) Suspiciously round numbers. This pattern suggests: Bot/script automation, Random clicking to maximize speed, Not reading instructions/examples. Investigation steps: Check for patterns (e.g., alternating labels, time between clicks too consistent), Review a sample manually, Check IP address for suspicious activity. If confirmed fraud: Reject all labels, don\'t pay, block worker, potentially report to platform. Option A is wrong—rewarding fraud encourages it. Option B might be too harsh without investigation—could be genuine misunderstanding. Option D is a weak compromise. Best practice: Clear quality thresholds in terms of service (e.g., "must maintain >80% accuracy on gold standards to get paid").',
  },
  {
    id: 'data-label-mc-3',
    question:
      'You use active learning: train model, it labels data, humans review uncertain cases, retrain. After 3 rounds, model accuracy is 91% but you notice it performs poorly on Class C (68% accuracy). You have 1,000 label budget. What should you prioritize?',
    options: [
      'Continue active learning—let model choose next 1,000 examples',
      'Random sample 1,000 examples from full dataset for unbiased coverage',
      'Manually select 1,000 examples from Class C to fix the weak spot',
      'Stratified sample: 500 Class C, 500 across others proportionally',
    ],
    correctAnswer: 3,
    explanation:
      'Option D (stratified with oversampling) is best. Here\'s why: Option A (continue active learning) will likely avoid Class C if model is confident (even though wrong). Active learning has a "cold start" problem—if model initially learned bad representation of Class C, it won\'t sample it. Option B (random sample) is unbiased but if Class C is rare (say 5% of data), you\'d only get 50 Class C examples—not enough to significantly improve. Option C (all Class C) swings too far—you\'d overtrain on Class C and potentially hurt performance on other classes. Also, you lose information about whether other classes still need work. Option D is optimal: 500 Class C examples will significantly improve weak class (68%→80%+ likely), 500 examples across others maintain overall performance, check if other classes have issues. After this round, re-evaluate and adjust strategy. This is called "stratified active learning" or "class-balanced active learning".',
  },
  {
    id: 'data-label-mc-4',
    question:
      'You discover that 20% of your training data has label noise (incorrect labels). Your current model accuracy is 84%. What is the MOST effective way to handle this?',
    options: [
      'Relabel all training data to fix the errors (most thorough)',
      'Train with label smoothing to make model robust to noise',
      'Use confident learning to automatically detect and fix/remove noisy labels',
      'Train multiple models and ensemble them to average out noise',
    ],
    correctAnswer: 2,
    explanation:
      "Option C (confident learning / noise detection) is most effective. Here's the comparison: Option A (relabel everything) is ideal but extremely expensive—if you have 100K examples and relabeling costs $0.50 each, that's $50K. Option C achieves similar results at fraction of cost. Label smoothing (Option B) helps a bit but doesn't fix the underlying data quality problem—model becomes more uncertain on everything, not just noisy labels. Ensembling (Option D) reduces variance but doesn't address bias from wrong labels. Confident Learning (Northcutt et al., 2021) method: Train model on noisy data, Use model predictions to identify likely label errors (examples where model strongly disagrees with label), Prioritize those for human review (~20K examples vs 100K full dataset), Fix or remove them, Retrain. Expected improvement: 84% → 89-91% accuracy. Cost: ~$10K to review 20K uncertain labels vs $50K to review all. Tools: cleanlab library implements this. Only relabel everything (Option A) if your budget allows and correctness is absolutely critical.",
  },
  {
    id: 'data-label-mc-5',
    question:
      'You need to create annotation guidelines for a subjective task (rating chatbot responses for "helpfulness"). Your annotators currently have 0.42 Krippendorff\'s Alpha. What is the FIRST step to improve?',
    options: [
      'Fire low-performing annotators and hire better ones',
      'Decompose "helpfulness" into multiple concrete sub-dimensions',
      'Switch from 5-point scale to 3-point scale to reduce disagreement',
      'Provide more training examples and require calibration tests',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (decompose into sub-dimensions) should be FIRST. Low inter-annotator agreement (0.42 = moderate at best) usually indicates the rubric itself is poorly defined, not that annotators are bad. "Helpfulness" is subjective—people interpret it differently. Decomposition approach: Break into concrete dimensions: Answers the question (binary), Factually accurate (binary), Appropriate length (3-point), Clear and understandable (3-point). Each dimension is easier to judge consistently. Combine into overall score using weights. Expected improvement: 0.42 → 0.68+ Alpha. Only after fixing the rubric should you: Train annotators on new rubric (Option D), Consider scale changes (Option C), Evaluate individual annotators (Option A). Option A (fire annotators) is often wrong—if EVERYONE has low agreement, it\'s a task definition problem, not a people problem. Exception: If one annotator has much lower agreement than others, that individual might need additional training or removal.',
  },
];
