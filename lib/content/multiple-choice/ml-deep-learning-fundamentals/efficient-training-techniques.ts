import { MultipleChoiceQuestion } from '../../../types';

export const efficientTrainingTechniquesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'efficient-training-mc1',
      question:
        'What is the primary benefit of using gradient accumulation during training?',
      options: [
        'It speeds up training by processing multiple batches in parallel',
        'It enables using a larger effective batch size without increasing GPU memory usage',
        'It improves model accuracy by smoothing gradient updates',
        'It automatically adjusts the learning rate during training',
      ],
      correctAnswer: 1,
      explanation:
        'Gradient accumulation allows you to simulate a larger batch size by accumulating gradients from multiple smaller batches before updating weights. This solves memory limitations - you get the benefits of a large batch size (more stable gradients) without needing to fit the entire large batch in GPU memory. It does not provide a speed benefit.',
    },
    {
      id: 'efficient-training-mc2',
      question:
        'In mixed precision training with PyTorch AMP, what is the purpose of loss scaling?',
      options: [
        'To normalize the loss across different batch sizes',
        'To prevent gradients from underflowing to zero in FP16 arithmetic',
        'To speed up the backward pass',
        'To improve model accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'Loss scaling multiplies the loss by a large factor before backward pass, preventing small gradients from underflowing to zero in FP16 arithmetic. The gradients are then scaled back down before updating the FP32 master weights. This allows FP16 training without losing gradient information, enabling 2x memory and speed benefits.',
    },
    {
      id: 'efficient-training-mc3',
      question: 'What is gradient checkpointing and when should you use it?',
      options: [
        'Saving model weights periodically during training',
        'Trading compute for memory by recomputing activations during backward pass',
        'Storing gradients to disk to save GPU memory',
        'Clipping gradients to prevent exploding gradients',
      ],
      correctAnswer: 1,
      explanation:
        'Gradient checkpointing trades compute for memory by not storing all intermediate activations. Instead, it recomputes them during the backward pass as needed. This reduces memory from O(n) to O(√n) for n layers, enabling much deeper models (~2-3x) at the cost of ~1.5x slower training. Use it when running out of memory but can afford extra compute time.',
    },
    {
      id: 'efficient-training-mc4',
      question:
        'Which PyTorch multi-GPU training approach is recommended for best performance?',
      options: [
        'nn.DataParallel',
        'nn.DistributedDataParallel (DDP)',
        'Manual model parallelism',
        'They all have similar performance',
      ],
      correctAnswer: 1,
      explanation:
        'DistributedDataParallel (DDP) is the recommended approach. It provides better performance than DataParallel through more efficient gradient communication, multi-process parallelism, and works across multiple machines. DataParallel is simpler but has limitations (single-process, Python GIL, less efficient). DDP is the industry standard for multi-GPU training.',
    },
    {
      id: 'efficient-training-mc5',
      question:
        'When training a model that is too large to fit on a single GPU, which approach should you use?',
      options: [
        'Data parallelism',
        'Model parallelism or pipeline parallelism',
        'Mixed precision training',
        'Gradient accumulation',
      ],
      correctAnswer: 1,
      explanation:
        "When the model itself doesn't fit on a single GPU, you need model parallelism (splitting model across GPUs) or pipeline parallelism (splitting model stages and pipelining micro-batches). Data parallelism won't help as it replicates the full model on each GPU. Mixed precision and gradient accumulation reduce memory but may not be sufficient for extremely large models.",
    },
  ];
