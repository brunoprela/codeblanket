export const llmEvaluationSafetyMC = {
  title: 'LLM Evaluation & Safety Quiz',
  id: 'llm-evaluation-safety-mc',
  sectionId: 'llm-evaluation-safety',
  questions: [
    {
      id: 1,
      question: 'What does MMLU benchmark evaluate?',
      options: [
        'Model latency and memory usage',
        'Multi-task language understanding across 57 subjects',
        'Multilingual understanding',
        'Model math and logic only',
      ],
      correctAnswer: 1,
      explanation:
        "MMLU (Massive Multitask Language Understanding) tests knowledge across 57 diverse subjects from elementary to professional level: STEM, humanities, social sciences. It's become a key benchmark for evaluating general capability of LLMs.",
    },
    {
      id: 2,
      question:
        'Why is perplexity a poor metric for evaluating modern LLM quality for end-user applications?',
      options: [
        "It's too computationally expensive",
        'It measures token prediction ability but not usefulness, helpfulness, or safety',
        'It only works on English',
        'It requires labeled data',
      ],
      correctAnswer: 1,
      explanation:
        "Perplexity measures how well a model predicts held-out text tokens—useful during pretraining but doesn't capture what users care about: helpfulness, harmlessness, accuracy, following instructions. A model can have low perplexity but give poor answers.",
    },
    {
      id: 3,
      question: 'What is "red teaming" in the context of LLM safety?',
      options: [
        'Training models on red-colored data',
        'Adversarial testing to find model vulnerabilities and unsafe behaviors',
        'Using red flags to mark unsafe outputs',
        'Emergency shutdown procedures',
      ],
      correctAnswer: 1,
      explanation:
        'Red teaming involves deliberately trying to make the model produce harmful, biased, or incorrect outputs. This adversarial testing helps discover vulnerabilities before deployment, informing safety improvements and content filtering strategies.',
    },
    {
      id: 4,
      question: 'What approach does "LLM-as-judge" use for evaluation?',
      options: [
        'Having judges manually evaluate every output',
        'Using a strong LLM to evaluate outputs from other models',
        'Automated grammar checking',
        'Crowdsourced voting',
      ],
      correctAnswer: 1,
      explanation:
        'LLM-as-judge uses a powerful model (like GPT-4) to evaluate outputs from other models, rating quality, helpfulness, accuracy. This scales better than human evaluation and correlates well with human judgment, though it has limitations (biases, consistency).',
    },
    {
      id: 5,
      question: 'What is benchmark contamination?',
      options: [
        'Benchmarks having incorrect labels',
        'Test data accidentally included in training data, inflating scores',
        'Benchmarks being too easy',
        'Multiple models using the same benchmark',
      ],
      correctAnswer: 1,
      explanation:
        "Contamination occurs when test/evaluation data leaks into training data (especially with web-scraped datasets). Models then memorize answers rather than truly learning, leading to inflated benchmark scores that don't reflect real capability. This is a major concern.",
    },
  ],
};
