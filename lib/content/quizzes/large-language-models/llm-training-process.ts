export const llmTrainingProcessQuiz = {
  title: 'LLM Training Process Discussion',
  id: 'llm-training-process-quiz',
  sectionId: 'llm-training-process',
  questions: [
    {
      id: 1,
      question:
        'Training large language models requires massive computational resources (thousands of GPUs, millions of dollars). Discuss the key techniques that make this feasible: data parallelism, model parallelism (tensor, pipeline, sequence), mixed precision training, and gradient checkpointing. How do these techniques work together to enable efficient training at scale?',
      expectedAnswer:
        'Should explain: data parallelism limitations with large models, splitting model across GPUs in tensor parallelism, pipeline parallelism for layer-wise distribution, bubble problem in pipeline parallelism and solutions, mixed precision using FP16/BF16 to reduce memory, gradient checkpointing trading computation for memory, ZeRO optimizer stages, CPU offloading strategies, communication bottlenecks, and how to combine techniques based on hardware and model size.',
    },
    {
      id: 2,
      question:
        'The quality and composition of pretraining data profoundly affect model capabilities and biases. Discuss the challenges in curating trillion-token datasets: data quality vs quantity tradeoffs, deduplication, filtering toxic content, balancing domains, and handling multilingual data. How do these decisions affect downstream model behavior?',
      expectedAnswer:
        'Should cover: Common Crawl as primary source and its issues, deduplication importance for preventing memorization, quality filtering using classifiers, domain mixing ratios affecting capabilities, code data improving reasoning, multilingual data tradeoffs, bias amplification from internet data, data contamination in benchmarks, the "data diet" debate (quality vs quantity), data ordering effects, curriculum learning during pretraining, and transparency challenges in dataset disclosure.',
    },
    {
      id: 3,
      question:
        'Explain the scaling laws discovered for language models (Kaplan et al., Chinchilla). How do model size, dataset size, and compute budget interact? Given a fixed compute budget, should you train a larger model on less data or a smaller model on more data? How have these insights changed the way models are trained?',
      expectedAnswer:
        'Should discuss: original scaling laws (Kaplan): focus on large models, Chinchilla revision: most models are undertrained, optimal compute allocation between parameters and tokens, 20:1 token to parameter ratio, implications for model design (Chinchilla, LLaMA vs GPT-3), inference cost vs training cost tradeoffs, Chinchilla-optimal training for deployment efficiency, continued training benefits, and future predictions about scaling limits.',
    },
  ],
};
