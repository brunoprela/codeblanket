/**
 * Discussion questions for Model Fine-Tuning Fundamentals section
 */

export const modelFineTuningFundamentalsQuiz = [
  {
    id: 'fine-tune-fundamentals-q-1',
    question:
      'You have a base GPT-3.5 model and 5,000 labeled examples for a customer support task. Your options: (A) Full fine-tuning (update all parameters), (B) LoRA (Low-Rank Adaptation), (C) Prompt engineering only. Compare the trade-offs in terms of cost, performance, and maintenance. Which would you choose and why?',
    hint: 'Consider: training cost, inference cost, model size, performance gains, flexibility for updates, and how many examples you have.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Full fine-tuning: Best performance but expensive ($500-2000), slow iteration, risk of catastrophic forgetting',
      'LoRA: 90-95% of full performance, 10-20x cheaper ($50-200), fast iteration, tiny adapter (20MB vs 13GB)',
      'Prompt engineering: Zero cost, instant iteration, but 5-15% accuracy gap',
      'Recommendation for 5K examples: LoRA—best cost/performance, fast iteration, preserves base model',
      'Hybrid approach: Start with prompts, add LoRA when data available, upgrade to full if needed',
      'LoRA advantages: Multiple adapters for different tasks, no catastrophic forgetting, stackable',
    ],
  },
  {
    id: 'fine-tune-fundamentals-q-2',
    question:
      'After fine-tuning your model on 10,000 domain-specific examples, you notice it performs great on your task (92% accuracy) but has "forgotten" basic general knowledge (e.g., "What is the capital of France?" → incorrect or refuses). This is catastrophic forgetting. How do you fix this while maintaining your task performance?',
    hint: 'Consider regularization techniques, data mixing strategies, and adapter methods that preserve base model capabilities.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Catastrophic forgetting: Fine-tuning overwrites general knowledge, model becomes overly specialized',
      'Solution 1 - Mix general data: Include 10-20% general examples during fine-tuning, preserves 85% general knowledge',
      'Solution 2 - EWC: Penalize changes to important weights using Fisher Information Matrix',
      'Solution 3 - LoRA/Adapters: Freeze base model, train small adapters, 100% knowledge preservation (best solution)',
      'Solution 4 - Multi-task: Fine-tune on multiple tasks simultaneously with task prefixes',
      'Solution 5 - Lower learning rate + early stopping: Gentle fine-tuning, monitor general knowledge',
      'Best approach: LoRA adapters + 15% general data mixing + multi-task learning',
    ],
  },
  {
    id: 'fine-tune-fundamentals-q-3',
    question:
      "You're deciding between fine-tuning a smaller model (7B parameters) vs prompt engineering a larger model (GPT-4, 175B+). For a sentiment analysis task with 2,000 labeled examples, compare: (1) Cost, (2) Performance, (3) Latency, (4) Maintenance. Which would you choose for a production system with 10M requests/month?",
    hint: 'Calculate end-to-end costs including training, inference, and iteration. Consider ongoing costs, not just initial setup.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Cost: Fine-tuned 7B ($3K/year) vs GPT-4 ($583K/year) at 10M requests/month → 195x cheaper',
      'Performance: Fine-tuned 91% vs GPT-4 85% → +6% accuracy with task-specific training',
      'Latency: Fine-tuned 50ms vs GPT-4 800ms → 16x faster response time',
      'Maintenance: Fine-tuned needs more upfront (2.5 weeks) and ongoing (8 hrs/month) vs GPT-4 (1 week, 2 hrs/month)',
      'Break-even: ~13K requests/month; above this, fine-tuned is economically better',
      'Hybrid approach: 95% fine-tuned + 5% GPT-4 for edge cases = best cost/performance balance',
      'Recommendation: Fine-tune 7B for production at scale, use GPT-4 for prototyping or low-volume',
    ],
  },
];
