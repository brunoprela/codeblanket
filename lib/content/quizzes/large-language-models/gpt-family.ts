export const gptFamilyQuiz = {
  title: 'GPT Family Discussion',
  id: 'gpt-family-quiz',
  sectionId: 'gpt-family',
  questions: [
    {
      id: 1,
      question:
        "Trace the evolution from GPT-2 to GPT-4, analyzing the key innovations at each stage. Beyond simply scaling parameters, what architectural, training, and alignment improvements were critical to GPT-4's superior performance? Discuss the role of RLHF, instruction tuning, and potential multimodal training.",
      expectedAnswer:
        "Should cover: GPT-2's unsupervised pretraining demonstration, GPT-3's few-shot learning emergence, instruction tuning in InstructGPT, RLHF integration, safety improvements across versions, hypothesized MoE architecture in GPT-4, multimodal capabilities, improved reasoning abilities, reduced hallucinations, cost-performance tradeoffs, and the shift from pure scaling to training efficiency.",
    },
    {
      id: 2,
      question:
        'GPT models are trained with a next-token prediction objective, yet they demonstrate capabilities far beyond simple text completion (reasoning, math, coding, translation). How does this simple objective lead to such diverse emergent abilities? Discuss the relationship between language modeling and world modeling.',
      expectedAnswer:
        'Should discuss: compression requires understanding, world knowledge embedded in training data, language as proxy for reasoning, multi-task learning through diverse internet text, emergent phenomena from scale, limitations of next-token prediction (planning, factual updates), relationship between perplexity and capabilities, debate over whether LLMs truly "understand," and implications for AI safety and alignment.',
    },
    {
      id: 3,
      question:
        'Compare the practical deployment tradeoffs between GPT-3.5-Turbo and GPT-4 for production applications. Under what circumstances would you choose each model? How do factors like cost, latency, context window, and capability affect these decisions? Discuss strategies for optimizing cost while maintaining quality.',
      expectedAnswer:
        'Should analyze: 10-20x cost difference implications, latency requirements for different applications, when GPT-3.5 is sufficient (simple classification, straightforward QA), when GPT-4 is necessary (complex reasoning, coding, nuanced understanding), cascading strategies (try 3.5 first), context window considerations, prompt optimization to reduce costs, caching strategies, batch processing benefits, and monitoring quality degradation when downgrading models.',
    },
  ],
};
