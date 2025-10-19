import { MultipleChoiceQuestion } from '../../../types';

export const regularizationTechniquesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'regularization-mc1',
      question:
        'What is the primary difference between L1 and L2 regularization in terms of their effect on model weights?',
      options: [
        'L1 penalizes large weights more heavily than L2',
        'L2 drives weights exactly to zero while L1 shrinks them proportionally',
        'L1 drives weights exactly to zero (sparse) while L2 shrinks them smoothly',
        'L2 is faster to compute than L1 but produces similar results',
      ],
      correctAnswer: 2,
      explanation:
        'L1 regularization adds λ*Σ|w| to the loss, with gradient λ*sign(w). This drives weights exactly to zero, creating sparse models. L2 regularization adds λ/2*Σ(w²) with gradient λ*w, shrinking weights proportionally but rarely to exactly zero. L1 is useful for feature selection, while L2 is more common for general regularization.',
    },
    {
      id: 'regularization-mc2',
      question:
        'During inference (test time), how should dropout be applied in a neural network that used dropout probability p=0.5 during training?',
      options: [
        'Apply dropout with the same p=0.5 to introduce randomness in predictions',
        'Apply dropout with a smaller probability like p=0.2',
        'Disable dropout entirely (p=0) and use all neurons',
        'Apply dropout but average results over multiple forward passes',
      ],
      correctAnswer: 2,
      explanation:
        'During inference, dropout should be completely disabled (p=0), using all neurons. Dropout is only for training regularization. At test time, we want deterministic predictions using the full network. With inverted dropout (modern approach), no scaling is needed at inference since we scaled during training by 1/(1-p).',
    },
    {
      id: 'regularization-mc3',
      question:
        'What is the main advantage of Batch Normalization beyond its regularization effect?',
      options: [
        'It reduces the number of parameters in the model',
        'It allows using higher learning rates and reduces sensitivity to initialization',
        'It eliminates the need for dropout',
        'It makes the model train faster by skipping certain computations',
      ],
      correctAnswer: 1,
      explanation:
        "Batch Normalization's main advantage is stabilizing training by reducing internal covariate shift. This allows using higher learning rates (faster convergence) and makes the network less sensitive to weight initialization. While BN does act as a regularizer, this is a secondary benefit. It doesn't reduce parameters or eliminate need for other regularization.",
    },
    {
      id: 'regularization-mc4',
      question:
        'When would Layer Normalization be preferred over Batch Normalization?',
      options: [
        'When training convolutional neural networks on image data',
        'When training with very large batch sizes (>256)',
        'When training RNNs/Transformers or using very small batch sizes',
        'Layer Normalization is always better and should replace Batch Normalization',
      ],
      correctAnswer: 2,
      explanation:
        'Layer Normalization is preferred for RNNs and Transformers because it normalizes across features (not batch), making each sample independent. This is crucial for variable-length sequences and small batch sizes. Batch Norm works better for CNNs with large batches. Layer Norm is used in all modern Transformer architectures (BERT, GPT, etc.).',
    },
    {
      id: 'regularization-mc5',
      question:
        'What is the purpose of the "patience" parameter in Early Stopping?',
      options: [
        'It determines how many epochs to train before evaluating validation loss',
        'It specifies the number of epochs to wait for improvement before stopping',
        'It controls the learning rate decay schedule',
        'It sets the minimum improvement required to continue training',
      ],
      correctAnswer: 1,
      explanation:
        'The patience parameter determines how many consecutive epochs without improvement to tolerate before stopping training. For example, patience=10 means "wait 10 epochs for validation loss to improve; if it doesn\'t, stop and restore the best weights." This prevents stopping too early due to temporary fluctuations while still catching true overfitting.',
    },
  ];
