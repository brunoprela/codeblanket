import { MultipleChoiceQuestion } from '@/lib/types';

export const modelTrainingPipelineQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'mtp-mc-1',
    question:
      "You're building a distributed training pipeline for a large computer vision model across 8 GPU nodes. The training occasionally hangs when synchronizing gradients across nodes. Which strategy would most effectively diagnose and resolve this issue?",
    options: [
      'Switch from synchronous to asynchronous gradient updates to avoid blocking',
      'Implement gradient compression and add timeout mechanisms with fallback to checkpoint recovery',
      'Reduce the batch size per GPU to minimize communication overhead',
      'Use model parallelism instead of data parallelism to reduce inter-node communication',
    ],
    correctAnswer: 1,
    explanation:
      "Implementing gradient compression (reducing communication volume) combined with timeout mechanisms and checkpoint recovery provides a robust solution. Timeouts detect hanging nodes, while compression reduces the likelihood of network bottlenecks. Checkpoint recovery ensures training can resume without starting over. Asynchronous updates (option A) can lead to convergence issues and stale gradients. Reducing batch size (option C) may hurt convergence and doesn't address the root cause. Model parallelism (option D) is more complex and doesn't necessarily reduce communication—it can actually increase it depending on the architecture.",
    difficulty: 'advanced',
    topic: 'Model Training Pipeline',
  },
  {
    id: 'mtp-mc-2',
    question:
      'Your training pipeline uses mixed precision training (FP16/FP32) to accelerate training on modern GPUs. However, you notice training becomes unstable with gradients occasionally becoming NaN. What is the most appropriate solution?',
    options: [
      'Disable mixed precision and train entirely in FP32',
      'Implement gradient clipping and loss scaling with dynamic adjustment',
      'Switch to BF16 (bfloat16) format instead of FP16',
      'Reduce the learning rate significantly to prevent overflow',
    ],
    correctAnswer: 1,
    explanation:
      "Gradient clipping prevents exploding gradients, while loss scaling (multiplying loss by a scale factor before backward pass, then unscaling gradients) prevents underflow in FP16 computations. Dynamic scaling automatically adjusts the scale factor based on gradient statistics. This is the standard approach for stable mixed precision training. Disabling mixed precision (option A) loses the speed benefits. BF16 (option C) can help but isn't always available and doesn't address gradient explosion. Simply reducing learning rate (option D) may slow convergence unnecessarily without addressing the numerical precision issues.",
    difficulty: 'advanced',
    topic: 'Model Training Pipeline',
  },
  {
    id: 'mtp-mc-3',
    question:
      "You're implementing a hyperparameter optimization pipeline that needs to evaluate 200 different configurations. The model takes 4 hours to train to convergence. Which strategy would most efficiently identify the best hyperparameters within a 48-hour window?",
    options: [
      'Use random search and train all 200 configurations in parallel on a large cluster',
      'Implement Bayesian optimization with early stopping based on validation performance',
      'Use grid search focusing on the 3 most important hyperparameters',
      'Train a smaller proxy model quickly to identify promising regions, then refine with full model',
    ],
    correctAnswer: 1,
    explanation:
      "Bayesian optimization intelligently samples the hyperparameter space based on previous results, requiring far fewer evaluations than random or grid search. Early stopping (e.g., with successive halving or ASHA) terminates unpromising configurations before full training, dramatically reducing compute time. This combination can evaluate configurations efficiently within the time budget. Option A would require 800 GPU-hours and isn't feasible in 48 hours without massive resources. Grid search (option C) is inefficient for high-dimensional spaces. The proxy model approach (option D) can work but may not transfer well and adds complexity.",
    difficulty: 'advanced',
    topic: 'Model Training Pipeline',
  },
  {
    id: 'mtp-mc-4',
    question:
      "Your training pipeline monitors GPU utilization and you notice it's consistently at 45-55% despite having enough memory. The model uses custom data preprocessing in PyTorch. What is the most likely bottleneck and solution?",
    options: [
      'GPU compute capacity is insufficient; upgrade to higher-end GPUs',
      'Data loading is CPU-bound; increase DataLoader workers and implement prefetching',
      'The model architecture is not optimized; switch to a more efficient architecture',
      'Memory bandwidth is limiting GPU performance; reduce batch size',
    ],
    correctAnswer: 1,
    explanation:
      "Low GPU utilization (45-55%) despite available memory indicates the GPU is waiting for data. This is a classic data loading bottleneck. Increasing DataLoader workers (num_workers) parallelizes data preprocessing on CPU, and prefetching (prefetch_factor) ensures batches are ready before the GPU needs them. PyTorch's DataLoader with pin_memory=True also helps. Option A is wrong because low utilization means the GPU isn't being fully used. Option C doesn't address the bottleneck cause. Option D would reduce utilization further and doesn't match the symptoms (memory isn't the issue).",
    difficulty: 'intermediate',
    topic: 'Model Training Pipeline',
  },
  {
    id: 'mtp-mc-5',
    question:
      "You're implementing gradient accumulation to simulate a larger batch size on memory-constrained GPUs. Your effective batch size is 128 (32 micro-batches × 4 accumulation steps). Training exhibits unstable loss curves compared to native batch size 128. What is the most likely issue?",
    options: [
      'Gradient accumulation is mathematically incorrect and cannot replicate large batch training',
      'Batch normalization statistics are computed per micro-batch instead of the full effective batch',
      'Learning rate needs to be scaled linearly with the effective batch size',
      'The optimizer state is not being properly synchronized across accumulation steps',
    ],
    correctAnswer: 1,
    explanation:
      "The most common issue with gradient accumulation is that batch normalization (BN) computes running statistics using only the micro-batch (32 samples) instead of the effective batch (128 samples). This creates a mismatch in BN behavior between accumulation and native large-batch training. Solutions include: switching to Group Normalization or Layer Normalization, synchronizing BN stats across accumulation steps, or adjusting BN momentum. Gradient accumulation is mathematically correct (option A is false). Learning rate scaling (option C) may help but isn't the primary issue here. Optimizer state (option D) is typically handled correctly by standard frameworks.",
    difficulty: 'advanced',
    topic: 'Model Training Pipeline',
  },
];
