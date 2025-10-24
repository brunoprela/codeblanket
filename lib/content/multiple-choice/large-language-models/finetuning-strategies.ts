export const finetuningStrategiesMC = {
  title: 'Fine-tuning Strategies Quiz',
  id: 'finetuning-strategies-mc',
  sectionId: 'finetuning-strategies',
  questions: [
    {
      id: 1,
      question:
        'What is the primary advantage of LoRA (Low-Rank Adaptation) over full fine-tuning?',
      options: [
        'Better final performance',
        'Faster convergence',
        'Drastically fewer trainable parameters (0.1-1% of model)',
        'Works with smaller datasets',
      ],
      correctAnswer: 2,
      explanation:
        'LoRA only trains low-rank decomposition matrices injected into attention layers, resulting in 0.1-1% trainable parameters compared to full fine-tuning. A rank-8 LoRA on LLaMA-7B requires only 8MB of adapters vs 14GB for full weights.',
    },
    {
      id: 2,
      question:
        'What is catastrophic forgetting in the context of fine-tuning?',
      options: [
        'The model forgets the fine-tuning data after training',
        'The model loses general capabilities while learning task-specific knowledge',
        'The model cannot remember long contexts',
        'Training loss suddenly increases',
      ],
      correctAnswer: 1,
      explanation:
        'Catastrophic forgetting occurs when fine-tuning on a narrow domain causes the model to lose its general capabilities learned during pretraining. The model overwrites pretrained weights with task-specific knowledge.',
    },
    {
      id: 3,
      question: 'In LoRA, what does the rank hyperparameter (r) control?',
      options: [
        'The number of training epochs',
        'The expressiveness of the adaptation (higher rank = more capacity)',
        'The layer where adaptation is applied',
        'The learning rate schedule',
      ],
      correctAnswer: 1,
      explanation:
        'The rank (r) controls the size of the low-rank matrices (e.g., r=8, r=16). Higher rank means more parameters and expressiveness but diminishing returns. Typical values are r=8-64, finding sweet spot between efficiency and capability.',
    },
    {
      id: 4,
      question:
        'Which fine-tuning approach has the fewest trainable parameters?',
      options: ['Full fine-tuning', 'LoRA', 'Prefix tuning', 'Adapter layers'],
      correctAnswer: 2,
      explanation:
        'Prefix tuning (or prompt tuning) trains only continuous prompt embeddings prepended to inputs, often just 0.01-0.1% of parameters. However, LoRA typically achieves better performance with slightly more parameters (~0.5-1%).',
    },
    {
      id: 5,
      question:
        'When is full fine-tuning preferred over parameter-efficient methods like LoRA?',
      options: [
        'Always - it always performs better',
        'When the model is very large (>70B parameters)',
        'When there is significant distribution shift or the model is relatively small',
        'When you need to deploy multiple task-specific models',
      ],
      correctAnswer: 2,
      explanation:
        "Full fine-tuning is preferred when: (1) there's major distribution shift from pretraining (e.g., domain-specific language), (2) the model is small enough to fully fine-tune easily (<7B), or (3) you need maximum performance on a single task.",
    },
  ],
};
