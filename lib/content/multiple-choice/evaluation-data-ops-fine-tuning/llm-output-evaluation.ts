/**
 * Multiple choice questions for LLM Output Evaluation section
 */

export const llmOutputEvaluationMultipleChoice = [
  {
    id: 'llm-eval-mc-1',
    question:
      "You're comparing two summarization models using ROUGE scores. Model A: ROUGE-1=0.45, ROUGE-2=0.25, ROUGE-L=0.42. Model B: ROUGE-1=0.52, ROUGE-2=0.18, ROUGE-L=0.48. Which statement is MOST accurate?",
    options: [
      'Model B is definitively better because it has higher ROUGE-1 and ROUGE-L',
      'Model A is better because ROUGE-2 (bigram) is more important than unigrams',
      'The models have different strengths; Model B better captures overall content (unigrams) while Model A better captures phrases (bigrams)',
      'The models are identical in quality since average ROUGE scores are similar',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. Different ROUGE scores measure different aspects: ROUGE-1 (unigram overlap) measures overall content coverage, ROUGE-2 (bigram overlap) measures phrase-level matching, ROUGE-L (longest common subsequence) measures sentence structure preservation. Model B excels at content coverage (0.52 ROUGE-1) but is weaker at preserving exact phrases (0.18 ROUGE-2). Model A is more balanced with better phrase matching. Neither is objectively "better"—it depends on use case. For casual summaries, Model B might be preferable (covers more content). For professional summaries where exact terminology matters (legal, medical), Model A might be better (preserves phrases). Always look at multiple metrics and understand trade-offs.',
  },
  {
    id: 'llm-eval-mc-2',
    question:
      "Your QA system generates answers evaluated with BERTScore. You get BERTScore precision=0.82, recall=0.76, F1=0.79. What does the precision > recall pattern suggest about your model's behavior?",
    options: [
      'The model generates answers that are highly accurate but may miss some information from the reference',
      'The model generates longer answers than necessary',
      'The model has a bug in the tokenization',
      'These scores indicate the model is performing poorly overall',
    ],
    correctAnswer: 0,
    explanation:
      "Option A is correct. In BERTScore, precision measures how much of the generated answer matches the reference (how accurate is what it says), while recall measures how much of the reference is captured in the generation (how complete is it). Precision > recall (0.82 > 0.76) indicates the model is conservative: what it says is mostly correct (high precision) but it doesn't cover everything it should (lower recall). This is often preferable to the opposite (low precision, high recall) which would mean generating lots of content, some wrong. The model might be: leaving out details, being overly concise, or playing it safe. Depending on use case, you might want to prompt it to be more comprehensive. The scores themselves (F1=0.79) are actually quite good, not poor (Option D is wrong).",
  },
  {
    id: 'llm-eval-mc-3',
    question:
      'You want to detect hallucinations in RAG system outputs. Which evaluation approach would be MOST effective?',
    options: [
      'Compare output to retrieved documents using ROUGE score',
      'Use NLI model to check if each claim in the output is entailed by the retrieved context',
      'Count how many facts in the output also appear in the reference answer',
      'Measure semantic similarity between output and context',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (NLI for entailment checking) is most effective for hallucination detection. Here's why: NLI (Natural Language Inference) models are trained to determine if a hypothesis (claim) is entailed by, contradicted by, or neutral with respect to a premise (context). For each claim in the output, check if context entails it. If not entailed → potential hallucination. ROUGE (A) only measures word overlap, not factual correctness. Semantic similarity (D) measures meaning similarity, but similar doesn't mean factually grounded. Comparing to reference (C) only works if you have reference answers (often you don't in RAG). NLI-based faithfulness checking is the current best practice: extract claims from output, verify each against retrieved context, calculate % of supported claims as faithfulness score.",
  },
  {
    id: 'llm-eval-mc-4',
    question:
      'Your code generation model achieves 75% pass@1 (75% of generations pass tests on first try) and 88% pass@5 (88% pass if you generate 5 attempts). What does this suggest about improving the model?',
    options: [
      'The model needs complete retraining since 75% is too low',
      'The model already has the capability to generate correct code, but needs better sampling/selection strategies',
      'Pass@5 being higher than pass@1 indicates a bug in the test suite',
      'These metrics are unrelated and cannot be compared',
    ],
    correctAnswer: 1,
    explanation:
      "Option B is correct. The gap between pass@1 (75%) and pass@5 (88%) indicates the model CAN generate correct code (it's in the model's capability) but isn't consistently selecting the best option. Solutions: (1) Improve sampling (better temperature, top-p), (2) Implement ranking/selection (generate multiple candidates, select best via heuristics or another model), (3) Self-consistency (generate multiple solutions, pick most common), (4) Test-time compute (like AlphaCode: generate many, test all, return best). This is cheaper than retraining (Option A). The gap is actually a GOOD sign—means you can improve to 88% without model changes. Option C is wrong—pass@k being higher is expected and normal. This pattern shows low-hanging fruit for improvement through better inference strategies.",
  },
  {
    id: 'llm-eval-mc-5',
    question:
      'When using GPT-4 as a judge to evaluate other models, what is the MOST important consideration to ensure reliable results?',
    options: [
      'Always use temperature=0 for deterministic judgments',
      'Have GPT-4 use a rubric and provide structured evaluations rather than free-form judgments',
      'Only evaluate one output at a time to avoid comparison bias',
      'Use the longest context window possible',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. Providing a structured rubric with specific criteria ensures: (1) Consistency across evaluations, (2) Specific feedback on dimensions, (3) Reduced bias, (4) Interpretable scores. Example: Instead of "Rate this output," use "Rate 1-5 on: Accuracy (is information correct?), Completeness (covers all aspects?), Clarity (easy to understand?)". Temperature=0 (A) helps but isn\'t sufficient—even deterministic outputs can be biased without structure. Evaluating one at a time (C) is actually worse—pairwise comparisons are often more reliable (easier to judge "which is better" than absolute quality). Long context (D) is irrelevant to reliability. Best practices for LLM-as-judge: clear rubric, specific criteria, few-shot examples of good judgments, validate against human agreement on subset.',
  },
];
