/**
 * Multiple choice questions for Fine-Tuning Open-Source Models section
 */

export const fineTuningOpenSourceModelsMultipleChoice = [
  {
    id: 'open-source-fine-tune-mc-1',
    question:
      'You want to fine-tune Llama 2 7B on a single A100 80GB GPU. Which combination will fit in memory and train efficiently?',
    options: [
      'Full fine-tuning in FP32',
      'Full fine-tuning in FP16 with gradient checkpointing',
      'LoRA (r=16) in FP16',
      'QLoRA (4-bit) with r=64',
    ],
    correctAnswer: 3,
    explanation:
      "Option D (QLoRA with r=64) is best. Memory analysis: Option A (FP32): 7B × 4 bytes = 28GB model + 28GB gradients + 56GB optimizer = 112GB → won't fit. Option B (FP16): 7B × 2 = 14GB model + 14GB grad + 28GB opt + 20GB activations = 76GB → barely fits, risky. Option C (LoRA FP16): 14GB model + small adapters → fits easily but r=16 is conservative. Option D (QLoRA): 7B × 0.5 bytes = 3.5GB model + adapters + optimizer = ~12GB → plenty of room, can use larger rank (r=64) for better performance. QLoRA gets best of both worlds: fits comfortably and high rank for performance.",
  },
  {
    id: 'open-source-fine-tune-mc-2',
    question:
      'After fine-tuning Llama 2 with LoRA (r=8), your validation accuracy is 85%. Full fine-tuning achieves 91%. What should you try FIRST to close the gap?',
    options: [
      'Increase LoRA rank to r=32 or r=64',
      'Switch to full fine-tuning (accept the cost)',
      'Collect more training data',
      'Train for more epochs',
    ],
    correctAnswer: 0,
    explanation:
      "Option A (increase LoRA rank) is the first thing to try. LoRA rank controls adapter capacity: r=8 is very small (only 8-dimensional adaptation). Increasing to r=32 or r=64 gives adapters much more expressiveness to capture task patterns. Expected improvement: 85% → 88-90% (close to full fine-tuning's 91%). Cost: Minimal—still much cheaper than full fine-tuning. Rank guidelines: r=8: Very small, for simple tasks, r=16: Standard, good balance, r=32-64: High capacity, complex tasks, r=128+: Approaching full fine-tuning capacity. Only switch to Option B (full) if increasing rank doesn't help. Options C and D are generic and don't address the LoRA capacity issue.",
  },
  {
    id: 'open-source-fine-tune-mc-3',
    question:
      'You deploy a fine-tuned Llama 2 7B model and notice inference is slow (500ms per request). Your GPU is only 30% utilized. What is the likely bottleneck?',
    options: [
      'GPU is too slow',
      'Small batch size (not utilizing GPU parallelism)',
      'Model is too large',
      'Network latency',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (small batch size) is the bottleneck. 30% GPU utilization means GPU is idle most of the time—not fully fed with work. Likely: Serving batch_size=1 (one request at a time), GPU processes quickly then waits for next request, most time spent idle. Solution: Batch requests together: Collect 8-32 concurrent requests, process as batch, dramatically improves throughput. Example: batch_size=1: 500ms per request, 30% util, 2 req/sec. batch_size=16: 150ms per request, 85% util, 100+ req/sec. Implementation: Use dynamic batching (vLLM, TensorRT-LLM, text-generation-inference). Option A is wrong—GPU is underutilized, not slow. Option C is wrong—7B is not too large. Option D is unlikely with only 30% util.',
  },
  {
    id: 'open-source-fine-tune-mc-4',
    question:
      'You fine-tuned Mistral 7B and want to quantize it to 4-bit for deployment. What is the correct approach?',
    options: [
      'Train with QLoRA (4-bit) from the start, deploy the 4-bit model',
      'Fine-tune in FP16, then use GPTQ or AWQ to quantize to 4-bit post-training',
      'Fine-tune in FP16, then use simple rounding to 4-bit',
      "Quantization doesn't work with fine-tuned models",
    ],
    correctAnswer: 1,
    explanation:
      'Option B (fine-tune FP16, then post-training quantization) is correct. Two approaches: Approach 1 - QLoRA: Train with 4-bit base model, LoRA adapters in FP16, merge and deploy. Result: 4-bit base, but LoRA merge happens in FP16 (may lose quantization). Approach 2 - Post-training quantization (BETTER): Fine-tune in FP16 (full precision training), quantize after training using GPTQ/AWQ, these methods calibrate quantization to minimize accuracy loss. Example: AutoGPTQ or AutoAWQ libraries. Typical accuracy: FP16 → 4-bit: 1-2% accuracy drop with GPTQ/AWQ, 5-10% drop with naive rounding (Option C). Option D is wrong—quantization works fine with fine-tuned models. Best practice: Fine-tune in FP16 for best quality, then quantize with GPTQ/AWQ for deployment.',
  },
  {
    id: 'open-source-fine-tune-mc-5',
    question:
      'You have 3 separate LoRA adapters for Llama 2 7B (customer support, sales, technical). What is the BEST deployment strategy?',
    options: [
      'Merge each adapter into base model, deploy 3 separate 13GB models',
      'Load base model once (13GB), dynamically switch adapters at runtime (20MB each)',
      'Distill each to 3 separate 1.3B models',
      'Fine-tune a single model to handle all three tasks',
    ],
    correctAnswer: 1,
    explanation:
      'Option B (dynamic adapter switching) is best for multiple adapters. Storage comparison: Option A: 3 × 13GB = 39GB, Option B: 13GB + 3 × 20MB = 13.06GB (33x smaller!), Option C: 3 × 2.6GB = 7.8GB (smaller but need distillation), Option D: Could work but loses separation of concerns. Benefits of Option B: Load base model once (13GB GPU RAM), all 3 adapters fit in 13.15GB total, switch adapters in <1 second, easy to update (swap 20MB file), minimal overhead (2-5% slower than merged). Implementation: Use PEFT library with adapter switching, serve from FastAPI/vLLM with multi-adapter support. Option A wastes storage and memory. Option C requires distillation work. Option D loses flexibility. Clear winner: Option B.',
  },
];
