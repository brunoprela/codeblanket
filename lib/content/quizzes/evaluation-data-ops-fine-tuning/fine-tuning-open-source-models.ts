/**
 * Discussion questions for Fine-Tuning Open-Source Models section
 */

export const fineTuningOpenSourceModelsQuiz = [
  {
    id: 'open-source-fine-tune-q-1',
    question:
      'Compare fine-tuning Llama 2 7B locally vs using OpenAI fine-tuning for GPT-3.5. Consider: (1) Setup complexity, (2) Training cost, (3) Inference cost at 1M req/month, (4) Control and customization, (5) Data privacy. Which would you choose for a healthcare application with strict privacy requirements and moderate budget (\$5K/month)?',
    hint: 'Healthcare requires data privacy (HIPAA compliance). Consider whether cloud APIs are acceptable and total cost of ownership including infrastructure.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Healthcare requires HIPAA compliance—data privacy is critical, self-hosted model essential',
      'Cost: Llama 2 $44.9K first year (\$14.9K ongoing) vs GPT-3.5 $11.2K (\$7.2K ongoing)',
      'Setup: Llama needs 2-3 weeks engineering vs GPT-3.5 needs 3-4 days',
      'Inference: Llama $740/month (A10G GPU) vs GPT-3.5 $600/month at 1M req/month',
      'Privacy: Llama full control, GPT-3.5 requires BAA and still external data sharing',
      'Customization: Llama full model access vs GPT-3.5 limited to prompts',
      'Recommendation: Llama 2 7B for healthcare—privacy premium worth $50K over 5 years',
    ],
  },
  {
    id: 'open-source-fine-tune-q-2',
    question:
      'You\'re fine-tuning Llama 2 70B but your A100 GPU (80GB) runs out of memory during training. You get "CUDA out of memory" error. What are your options, ranked by effectiveness, and what are the trade-offs of each approach?',
    hint: 'Consider quantization, gradient checkpointing, LoRA, DeepSpeed ZeRO, and multi-GPU strategies. Think about memory vs speed vs accuracy trade-offs.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'OOM error: Llama 2 70B needs 140GB (FP16) but A100 only has 80GB',
      'Solution 1 - LoRA: Freeze base, train adapters (79GB total, 0.16% trainable params)',
      'Solution 2 - QLoRA: 4-bit quantization + LoRA (44GB total, best option)',
      'Solution 3 - Gradient checkpointing: Recompute activations, saves 22GB, 20-30% slower',
      'Solution 4 - DeepSpeed ZeRO-3: Partition across GPUs + CPU offload, needs multiple GPUs',
      'Solution 5 - Smaller batch: batch_size=1 + grad accumulation, saves 26GB, proportionally slower',
      'Best combination: QLoRA + gradient checkpointing + batch_size=1 = 30-35GB (fits easily)',
    ],
  },
  {
    id: 'open-source-fine-tune-q-3',
    question:
      'After fine-tuning Llama 2 7B with LoRA, you want to deploy it to production. Compare three deployment options: (1) Merge LoRA and deploy full model, (2) Keep LoRA separate and load at runtime, (3) Distill fine-tuned model to smaller model. What are the trade-offs for latency, memory, and flexibility?',
    hint: 'Consider inference performance, ability to switch between models, and resource requirements. Think about serving multiple fine-tuned versions.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Option 1 - Merged: 13GB per model, fastest (50ms), least flexible, use for single model',
      'Option 2 - Separate LoRA: 13GB base + 20MB per adapter, minimal overhead (52ms), most flexible, best for multiple variants',
      'Option 3 - Distilled: 2.6GB (5x smaller), fastest (10ms), 5-10% accuracy loss, best for scale',
      'Storage: 3 variants = Merged 39GB vs LoRA 13.06GB vs Distilled 7.8GB',
      'Flexibility: LoRA allows runtime adapter switching, merged and distilled need separate models',
      'Recommendation: Start with LoRA (flexible, easy updates), distill for production scale (5x faster, 1/5 cost)',
      'Hybrid: Use LoRA in dev/staging for iteration, distilled in production for performance',
    ],
  },
];
