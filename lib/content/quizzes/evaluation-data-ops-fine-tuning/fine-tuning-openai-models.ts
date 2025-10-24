/**
 * Discussion questions for Fine-Tuning OpenAI Models section
 */

export const fineTuningOpenAIModelsQuiz = [
  {
    id: 'openai-fine-tune-q-1',
    question:
      'You want to fine-tune GPT-3.5-turbo for a customer support chatbot. You have 10,000 conversations in your database. Walk through the complete process: (1) Data preparation and formatting, (2) Uploading and starting the fine-tuning job, (3) Evaluation and deployment, (4) Cost estimation. What are the critical considerations at each step?',
    hint: 'OpenAI fine-tuning requires JSONL format with specific structure. Consider data quality, validation splits, hyperparameters, and ongoing costs.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Data prep: JSONL format with system/user/assistant messages, validate quality (no PII, reasonable length)',
      'Upload: Use OpenAI API to upload train/val files, create fine-tuning job with hyperparameters',
      'Hyperparameters: 3 epochs typical, auto learning rate, takes 20-60 minutes for 10K examples',
      'Evaluation: Compare to baseline on held-out test set, use human eval or LLM-as-judge',
      'Cost: ~$0.008/1K tokens training ($48 for 10K examples), 2x inference cost vs base model',
      'Deployment: Simple API swap to fine-tuned model ID, monitor performance',
      'ROI: Worth it if performance gain justifies 2x inference cost or reduces support costs',
    ],
  },
  {
    id: 'openai-fine-tune-q-2',
    question:
      'Your OpenAI fine-tuned model shows training loss decreasing to 0.5 but validation loss plateaus at 1.2 after epoch 2. What does this indicate, and what hyperparameter adjustments would you make?',
    hint: 'Gap between training and validation loss indicates overfitting. Consider adjusting epochs, learning rate, or data augmentation strategies.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Large train/val loss gap (0.5 vs 1.2) indicates overfitting—memorizing training data',
      'Solution 1 (easiest): Reduce epochs from 3 to 2, stop when validation is best',
      'Solution 2: Lower learning rate (0.5-1.0 multiplier), slower learning reduces overfitting',
      'Solution 3: Collect more data (5K-10K examples), harder to memorize larger dataset',
      'Solution 4: Data augmentation (paraphrasing, back-translation) to create variations',
      'Solution 5: Mix 10-15% general data to improve generalization',
      'Target: Train/val gap < 0.3 is healthy, > 0.5 is problematic',
    ],
  },
  {
    id: 'openai-fine-tune-q-3',
    question:
      'You fine-tuned GPT-3.5 successfully 6 months ago (92% accuracy). Now, OpenAI deprecated that model version and released a new base GPT-3.5. Your fine-tuned model is no longer available. How do you handle model versioning and deprecation for production systems?',
    hint: 'Consider model lifecycle management, backup strategies, migration planning, and avoiding vendor lock-in.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Model deprecation: When OpenAI retires base model, your fine-tuned model stops working',
      'Immediate response: Backup training data, download all JSONL files and metadata',
      'Migration: Fine-tune on new base model with same hyperparameters and data',
      'Evaluation: Compare old vs new model performance on held-out test set',
      'Gradual rollout: 10% → 25% → 50% → 100% traffic over 3-4 weeks',
      'Prevention: Version control, automatic backups, track deprecation dates, multi-vendor strategy',
      'Self-hosted backup: Fine-tune open-source model (Llama/Mistral) as fallback',
    ],
  },
];
