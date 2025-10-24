export const promptEngineeringQuiz = {
  title: 'Prompt Engineering Discussion',
  id: 'prompt-engineering-quiz',
  sectionId: 'prompt-engineering',
  questions: [
    {
      id: 1,
      question:
        'Chain-of-thought (CoT) prompting dramatically improves reasoning capabilities in large language models. Explain why explicitly showing reasoning steps helps models solve complex problems. Does CoT represent true reasoning or sophisticated pattern matching? Discuss the implications for AGI and the limits of current LLMs.',
      expectedAnswer:
        'Should cover: CoT as intermediate computation space, breaking down complex problems, attention mechanism benefits from step-by-step text, emergent capability appearing at sufficient scale, few-shot CoT vs zero-shot CoT differences, self-consistency improvements via multiple paths, debate over genuine reasoning vs pattern matching, comparison to human reasoning processes, limitations on truly novel problems, and philosophical implications for understanding and consciousness in LLMs.',
    },
    {
      id: 2,
      question:
        'Prompt engineering often feels like an art rather than a science—small wording changes can dramatically affect output quality. Why are language models so sensitive to prompt formulation? Discuss techniques to make prompting more robust and systematic: structured formats, role prompting, few-shot examples, and prompt templates. How can we move toward more reliable prompting?',
      expectedAnswer:
        'Should discuss: sensitivity to pretraining data distribution, reward model biases from RLHF, role prompts activating different behaviors, few-shot examples as implicit task specification, format consistency importance, prompt engineering as inverse problem, optimization approaches (automatic prompt engineering), prompt evaluation and testing strategies, version control for prompts, A/B testing prompts, and emerging tools for systematic prompt development.',
    },
    {
      id: 3,
      question:
        'Compare zero-shot, one-shot, and few-shot prompting strategies. How does the number and quality of examples affect model performance? Discuss the tradeoffs: context window usage, cost, latency, and performance. When should you invest in few-shot examples vs fine-tuning the model?',
      expectedAnswer:
        'Should analyze: zero-shot relying entirely on pretraining, task specification through instructions alone, one-shot as minimal demonstration, few-shot as implicit gradient descent in context, example selection importance (representativeness, diversity), optimal number of examples (typically 3-10), diminishing returns beyond certain count, context window consumption, cost implications of longer prompts, when few-shot suffices vs requiring fine-tuning (high volume, latency sensitive, need determinism), and hybrid approaches using both techniques.',
    },
  ],
};
