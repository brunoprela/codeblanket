export const llmFundamentalsQuiz = {
  title: 'LLM Fundamentals Discussion',
  id: 'llm-fundamentals-quiz',
  sectionId: 'llm-fundamentals',
  questions: [
    {
      id: 1,
      question:
        'How have large language models fundamentally changed the software development paradigm compared to traditional programming approaches? Discuss the shift from explicit rule-based programming to prompt-based programming, and analyze both the opportunities and risks this transformation presents for software engineering practices.',
      expectedAnswer:
        'Should discuss: the transition from deterministic to probabilistic programming, emergence of prompt engineering as new skill, reduced need for extensive labeled data, challenges in testing/validation, unpredictability vs traditional control flow, opportunities for rapid prototyping, risks of hallucinations and reliability, new debugging paradigms, and implications for software architecture patterns.',
    },
    {
      id: 2,
      question:
        'Compare and contrast the emergent capabilities that appear in large language models at scale (100B+ parameters) versus smaller models. Why do capabilities like chain-of-thought reasoning, arithmetic, and multi-step problem solving suddenly appear at certain scales? Discuss the implications of these scaling laws for future model development and the feasibility of achieving AGI.',
      expectedAnswer:
        'Should cover: the concept of emergence in neural networks, phase transitions in capability as scale increases, grokking phenomena, relationship between parameter count and capability, compression theory perspective, implications of scaling laws (Kaplan et al., Chinchilla), diminishing returns at extreme scales, cost-benefit analysis of scale, alternative approaches to capability (architecture innovations, better training), and philosophical questions about whether intelligence is fundamentally about scale.',
    },
    {
      id: 3,
      question:
        "Analyze the architectural decisions behind the success of decoder-only models (like GPT) versus encoder-only (like BERT) versus encoder-decoder models (like T5). For what types of tasks is each architecture optimal, and why? How do these architectural choices affect the model's ability to perform different types of reasoning and generation tasks?",
      expectedAnswer:
        'Should explain: causal vs bidirectional attention tradeoffs, autoregressive generation in decoder-only models, masked language modeling in encoders, when bidirectional context is crucial (NLU tasks), when causal context is necessary (generation), hybrid approaches and their use cases, efficiency considerations, fine-tuning implications, zero-shot vs few-shot performance differences, and the trend toward unified decoder-only architectures despite apparent limitations.',
    },
  ],
};
