/**
 * CNN Architectures Multiple Choice Questions
 */
export const cnnArchitecturesMultipleChoice = [
  {
    id: 'cnn-mc-1',
    question:
      'What was the key innovation in AlexNet (2012) that dramatically improved training speed compared to previous networks?',
    options: [
      'Using average pooling instead of max pooling',
      'Using ReLU activation functions instead of tanh/sigmoid',
      'Using smaller 3×3 filters throughout the network',
      'Using residual skip connections',
    ],
    correctAnswer: 1,
    explanation:
      "AlexNet\'s use of **ReLU (Rectified Linear Unit) activation functions** instead of tanh/sigmoid was revolutionary. ReLU trains approximately 6× faster than tanh because: (1) Gradient is either 0 or 1 (simple), (2) No expensive exponential calculations, (3) Helps mitigate vanishing gradients. Before AlexNet, most networks used tanh or sigmoid, which have gradients that approach zero for large inputs, slowing training dramatically. ReLU's simplicity (max(0, x)) and gradient properties made training deep networks on GPUs practical.",
  },
  {
    id: 'cnn-mc-2',
    question:
      "In ResNet\'s residual block, what does the skip/shortcut connection actually add to the output?",
    options: [
      'The gradient from the next layer',
      'A learned transformation of the input',
      'The original input (identity mapping)',
      'The average of all previous layer outputs',
    ],
    correctAnswer: 2,
    explanation:
      "The skip connection adds the **original input unchanged** (identity mapping): H(x) = F(x) + x. This is crucial because: (1) It allows gradients to flow directly backwards through the network without vanishing, (2) If the optimal function is close to identity, the network only needs to learn F(x) ≈ 0 which is easier than learning the full transformation, (3) It enables training of very deep networks (100+ layers). The '+x' term is not learned - it's a direct addition of the input to the transformed output. This simple addition has profound effects on trainability.",
  },
  {
    id: 'cnn-mc-3',
    question:
      'Why does VGG-16 use multiple stacked 3×3 convolutions instead of fewer larger filters (e.g., one 5×5 or 7×7)?',
    options: [
      '3×3 filters are always faster to compute',
      'Multiple 3×3 filters provide same receptive field with fewer parameters and more non-linearities',
      '3×3 is required for batch normalization to work properly',
      'Larger filters cause overfitting on ImageNet',
    ],
    correctAnswer: 1,
    explanation:
      'Stacking multiple 3×3 filters is more efficient than using larger filters: **Two 3×3 convs** (stacked) = 5×5 receptive field but with **18 parameters** (2×9) vs 25 for one 5×5 filter. **Three 3×3 convs** = 7×7 receptive field with 27 params vs 49 for one 7×7. Benefits: (1) ~30-40% fewer parameters, (2) **More non-linearities** (ReLU after each conv increases representation power), (3) Deeper network learns more complex features. This insight from VGG influenced nearly all subsequent architectures to use small filters.',
  },
  {
    id: 'cnn-mc-4',
    question:
      'What is the main purpose of 1×1 convolutions in the Inception module?',
    options: [
      'To increase the spatial dimensions of feature maps',
      'To reduce dimensionality (number of channels) before expensive operations',
      'To replace max pooling layers',
      'To enable residual connections',
    ],
    correctAnswer: 1,
    explanation:
      "1×1 convolutions are used for **dimensionality reduction** (reducing number of channels) before computationally expensive operations. Example: Instead of applying 5×5 conv on 256 channels directly (expensive), first use 1×1 conv to reduce 256→64 channels, then apply 5×5 conv, then expand back. This reduces parameters and FLOPs by ~4-10× while maintaining representational power. Also called 'bottleneck layers' or 'network-in-network', they were popularized by GoogLeNet and are now used in ResNet (bottleneck blocks), MobileNet, and many modern architectures.",
  },
  {
    id: 'cnn-mc-5',
    question:
      'You need to deploy a CNN on a mobile device with limited memory and computation. Which architecture design choice would you prioritize?',
    options: [
      'VGG-style: Deep network with many parameters for maximum accuracy',
      'AlexNet-style: Large FC layers with dropout for regularization',
      'EfficientNet-style: Compound scaling with depth-width-resolution balance and mobile optimizations',
      'Plain CNN: Simple architecture without skip connections or advanced techniques',
    ],
    correctAnswer: 2,
    explanation:
      'EfficientNet is specifically designed for **efficiency** - achieving high accuracy with minimal parameters and FLOPs. Its compound scaling ensures balanced growth that maximizes accuracy per FLOP. For mobile deployment: EfficientNet-B0 has only 5.3M parameters and achieves 77.3% ImageNet accuracy. Compare to: VGG-16 (138M params), AlexNet (60M params), or plain CNNs (lower accuracy). Modern variants like EfficientNet-Lite and MobileNetV3 further optimize for mobile/edge devices with techniques like depthwise-separable convolutions. The balanced scaling prevents over-parameterizing any single dimension (depth, width, resolution).',
  },
];
