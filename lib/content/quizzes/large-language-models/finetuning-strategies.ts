export const finetuningStrategiesQuiz = {
  title: 'Fine-tuning Strategies Discussion',
  id: 'finetuning-strategies-quiz',
  sectionId: 'finetuning-strategies',
  questions: [
    {
      id: 1,
      question:
        'Compare full fine-tuning, LoRA, and prompt tuning for adapting a large language model to a specific task. Analyze the tradeoffs in terms of: trainable parameters, memory requirements, training time, final performance, and ease of deployment. Under what circumstances would you choose each approach?',
      expectedAnswer:
        "Should cover: full fine-tuning requires updating all parameters, catastrophic forgetting risks, LoRA's low- rank adaptation of weights reducing parameters by 100 - 1000x, minimal performance degradation with LoRA, prompt tuning's parameter efficiency (0.01% parameters) but performance limitations, memory and compute requirements for each, deployment complexity (multiple LoRA adapters vs full models), when to use full fine-tuning (major distribution shift, small models), when LoRA suffices (large models, moderate adaptation), and prompt tuning for resource-constrained scenarios.",
    },
    {
      id: 2,
      question:
        'Discuss the phenomenon of catastrophic forgetting in fine-tuned language models. Why does fine-tuning on a narrow domain often degrade general capabilities? What techniques can mitigate this: continual learning, elastic weight consolidation, data mixing, or adapter methods? How do you balance specialization and generalization?',
      expectedAnswer:
        'Should explain: neural network plasticity-stability tradeoff, overwriting of pretrained weights during fine-tuning, distribution shift between pretraining and fine-tuning data, mixing pretraining data during fine-tuning to maintain generality, parameter-efficient methods (LoRA) avoiding catastrophic forgetting by not modifying original weights, continual learning techniques, evaluation importance for detecting forgetting, when forgetting is acceptable (highly specialized models), and strategies for production models needing both domain expertise and general capability.',
    },
    {
      id: 3,
      question:
        'LoRA (Low-Rank Adaptation) has become extremely popular for fine-tuning large models efficiently. Explain the mathematical intuition behind why low-rank updates are sufficient for adaptation. What is the relationship between the rank hyperparameter, model capacity, and final performance? How does LoRA enable practical multi-tenant model serving?',
      expectedAnswer:
        'Should discuss: weight updates having low intrinsic dimensionality, most information in top singular values, rank as control of expressiveness (typical r=8-64), higher rank = more capacity but diminishing returns, memory savings (rank 8 on 7B model = 8MB adapter vs 14GB full model), inference optimizations by merging LoRA weights, multi-tenant serving with adapter swapping, per-user/per-task adapters, composing multiple LoRA adapters, and practical deployment architectures for LoRA-based systems.',
    },
  ],
};
