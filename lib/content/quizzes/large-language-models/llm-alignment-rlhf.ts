export const llmAlignmentRLHFQuiz = {
  title: 'LLM Alignment & RLHF Discussion',
  id: 'llm-alignment-rlhf-quiz',
  sectionId: 'llm-alignment-rlhf',
  questions: [
    {
      id: 1,
      question:
        'RLHF (Reinforcement Learning from Human Feedback) has become critical for aligning language models with human values. Explain the three-stage process: supervised fine-tuning, reward model training, and PPO optimization. Why is each stage necessary? What are the failure modes and limitations of RLHF?',
      expectedAnswer:
        'Should cover: base model being misaligned with human preferences, SFT providing initial alignment, reward model learning human preferences from comparisons, RL optimization against reward model, PPO preventing excessive divergence from base model (KL penalty), reward hacking issues, mode collapse risks, RLHF amplifying labeler biases, sycophancy problems, difficulty of specifying true human values, scalability challenges of human feedback, and alternative alignment approaches (constitutional AI, debate, recursive reward modeling).',
    },
    {
      id: 2,
      question:
        'Discuss the alignment problem in AI: ensuring that powerful AI systems behave in accordance with human values and intentions. Why is alignment particularly challenging for language models? How do techniques like RLHF, constitutional AI, and red teaming address alignment? What are the unsolved problems?',
      expectedAnswer:
        "Should discuss: specification problem (defining human values formally), reward gaming and Goodhart's law, mesa- optimization and deceptive alignment risks, value extrapolation and edge cases, capability vs alignment progress mismatch, interpretability challenges, measuring alignment reliably, scaling oversight to superhuman systems, debate over alignment difficulty, near - term risks(misinformation, harmful content) vs long - term existential risks, and open research directions.",
    },
    {
      id: 3,
      question:
        'Compare different alignment techniques: RLHF, Constitutional AI (Anthropic), and debate-based approaches. What are the philosophical and practical differences? Which approach is most promising for achieving robust alignment at scale? Discuss the role of interpretability in alignment research.',
      expectedAnswer:
        "Should analyze: RLHF's reliance on human feedback (expensive, doesn't scale), Constitutional AI using AI-generated critiques and principles (more scalable), debate approach where AIs argue for human judges, scalability comparisons, interpretability enabling verification of alignment, mechanic interpretability for understanding model internals, activation engineering for control, advantages and limitations of each approach, combination strategies, and research frontiers in scalable oversight.",
    },
  ],
};
