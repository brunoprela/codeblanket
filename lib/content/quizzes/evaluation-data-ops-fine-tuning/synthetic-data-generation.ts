/**
 * Discussion questions for Synthetic Data Generation section
 */

export const syntheticDataGenerationQuiz = [
  {
    id: 'synthetic-data-q-1',
    question:
      'You need 10,000 training examples for a customer support chatbot but only have 500 real conversations. Design a synthetic data generation strategy that creates diverse, high-quality examples while avoiding common pitfalls (repetitive, unrealistic, biased). What techniques would you use and how would you validate quality?',
    hint: 'Consider using LLMs for generation, but with constraints and diversity mechanisms. Think about validation strategies to ensure synthetic data actually helps model performance.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Multi-strategy approach: Templates (fast), LLM generation (diverse), augmentation (realistic), edge cases (robust)',
      'LLM diversity: High temperature, presence/frequency penalties, cluster-based prompting',
      'Diversity check: Measure embedding similarity, reject near-duplicates above threshold',
      'Validation: Diversity (internal), realism (classifier test), coverage (topic distribution), utility (model performance)',
      'Allocate budget: 50% LLM, 20% templates, 20% augmentation, 10% edge cases',
      'Quality thresholds: Diversity >0.7, realism 0.4-0.6, coverage >0.9, utility >+5%',
    ],
  },
  {
    id: 'synthetic-data-q-2',
    question:
      "You generate 50,000 synthetic examples using GPT-4 to train a smaller model. After training, you discover the small model has inherited GPT-4's biases and failure modes (e.g., overconfidence, verbose responses, specific error patterns). How do you diagnose which synthetic examples are problematic and fix the dataset?",
    hint: 'Consider comparing model behavior on real vs synthetic data, using influence functions to find problematic examples, and filtering/correction strategies.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Diagnosis: Compare model behavior on real vs synthetic data (length, confidence, error patterns)',
      'Identify problematic examples: Influence functions (find harmful training points), outlier detection, GPT-4 artifact detection',
      'Fix strategies: Remove problematic (10-20%), correct/rewrite (remove hedging, verbosity), balance with real data (≥30% real)',
      'Retrain and validate: Measure accuracy, response length, calibration, failure modes',
      'Prevention: Mix real+synthetic, diverse generation methods, explicit prompts for desired style',
      'Expected improvement: +3-7% accuracy, -20-30% response length, better calibration',
    ],
  },
  {
    id: 'synthetic-data-q-3',
    question:
      'You want to generate synthetic data for a low-resource language (e.g., Swahili) where you only have 100 labeled examples and no access to large LLMs trained on that language. Design a practical synthetic data generation strategy using available tools and resources.',
    hint: 'Consider translation-based approaches, multilingual models, back-translation, and cross-lingual transfer. Think about validation when you have very limited gold standard data.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Translation-based: Translate English→Swahili, filter with back-translation (60% of data)',
      'Multilingual LLMs: GPT-4 with few-shot Swahili prompts (20% of data, expensive)',
      'Augmentation: Synonym replacement, paraphrasing seed examples (15% of data)',
      'Template-based: Extract patterns from seed, fill with Swahili vocabulary (5% of data)',
      'Validation with limited gold: Perplexity, native speaker sample, back-translation consistency, downstream task improvement',
      'Expected: 5K synthetic from 100 seed, +15-25% accuracy, $170 total cost ($0.034/example)',
    ],
  },
];
