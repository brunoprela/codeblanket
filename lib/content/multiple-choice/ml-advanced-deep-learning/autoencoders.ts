/**
 * Autoencoders Multiple Choice Questions
 */

export const autoencodersMultipleChoice = [
  {
    id: 'mc-1',
    question:
      'What is the primary purpose of the bottleneck (latent space) in an autoencoder?',
    options: [
      'To make the network train faster',
      'To force the network to learn a compressed representation that captures only the most important features',
      'To reduce memory usage during inference',
      'To prevent overfitting by having fewer neurons',
    ],
    correctAnswer: 1,
    explanation:
      "The bottleneck forces **information compression and feature learning**: **Without bottleneck**: Input (784-dim) → Hidden (784-dim) → Output (784-dim). Network can learn identity function (memorization). Just copy each input dimension to output, no feature learning! **With bottleneck**: Input (784-dim) → Latent (32-dim) → Output (784-dim). Information bottleneck! Can't copy all 784 dimensions through 32-dim space. Must learn WHICH features matter most. **Example** (face images): Can't store exact pixel values in 32 dimensions. Must learn high-level features: face shape, eyes, nose positions, lighting. Reconstruction: Decode from these compressed features. **Analogy**: Summarizing a book (must identify key points, can't copy verbatim). Training speed is actually SLOWER (harder optimization). Memory savings minimal (bottleneck is small part of network). Overfitting prevention is side benefit, not primary goal. The key insight: Compression forces abstraction and feature discovery.",
  },
  {
    id: 'mc-2',
    question:
      'In a denoising autoencoder, why do we train the network to reconstruct the clean image from a noisy input, rather than reconstructing the noisy image from itself?',
    options: [
      'Reconstructing noisy images is computationally more expensive',
      'To force the network to learn robust features and the underlying structure of the data, rather than just memorizing noise patterns',
      'Noisy images require larger networks',
      'This is just a convention with no real benefit',
    ],
    correctAnswer: 1,
    explanation:
      'Reconstructing CLEAN from NOISY forces learning **underlying structure**, not surface patterns: **Standard autoencoder**: Input: clean image x, Target: same clean image x. Problem: Can memorize specific pixel patterns, superficial features. **Denoising autoencoder**: Input: noisy image x̃ = x + noise, Target: clean image x (not x̃!). Network must: (1) Identify noise vs. signal, (2) Learn what "clean" looks like (statistical structure), (3) Reconstruct based on understanding, not memorization. **Example** (handwritten digit): Noisy "7" has random pixel noise. Reconstruction: Can\'t just copy noisy pixels (would copy noise too). Must recognize "this is a 7" → reconstruct prototypical "7". **Benefits**: Robust features (work despite variations), Prevents overfitting (can\'t memorize specific noise), Better generalization (learned structure, not specifics). **Analogy**: Learning to read messy handwriting forces understanding of letter shapes, not just copying. Cost is similar (same network), size unchanged. The training objective fundamentally changes what\'s learned from memorization to understanding.',
  },
  {
    id: 'mc-3',
    question:
      'What is the key difference between a basic autoencoder and a Variational Autoencoder (VAE)?',
    options: [
      'VAE uses convolutional layers while basic autoencoders do not',
      'VAE learns a probabilistic latent space with a smooth structure by encoding inputs as distributions rather than points',
      'VAE has more layers than basic autoencoders',
      'VAE uses a different activation function in the decoder',
    ],
    correctAnswer: 1,
    explanation:
      "VAE learns **structured probabilistic latent space** vs. autoencoder's **arbitrary point cloud**: **Basic Autoencoder**: Encoder: x → z (single point in latent space). Latent space: Scattered points, no structure. Problem: Can't generate new samples (no knowledge between training points). **Variational Autoencoder**: Encoder: x → (μ, σ²) distribution parameters. Sample: z ~ N(μ, σ²) (stochastic). Latent space: Smooth, continuous distribution. **Key differences**: (1) **Encoding**: Point vs. distribution, (2) **Loss**: MSE vs. MSE + KL divergence (forces latent to be standard normal), (3) **Sampling**: Deterministic vs. stochastic (reparameterization trick), (4) **Generation**: Can't generate vs. sample z ~ N(0,I) and decode. **Example**: Basic AE latent space: {z_cat₁, z_cat₂, ..., z_dog₁, z_dog₂}. Scattered, gaps between. Can't sample between points. VAE latent space: Smooth region for cats, smooth region for dogs. Can sample anywhere and decode to valid image. **Not about**: Architecture depth (both can use any), convolutions (both can), activations (similar). The FUNDAMENTAL difference is probabilistic modeling with structured latent space.",
  },
  {
    id: 'mc-4',
    question:
      'In VAE training, what is the purpose of the KL divergence term in the loss function?',
    options: [
      'To make training faster by simplifying gradients',
      'To regularize the latent space, forcing it to be close to a standard normal distribution N(0, I) so we can sample from it',
      'To increase reconstruction accuracy',
      'To prevent the decoder from overfitting',
    ],
    correctAnswer: 1,
    explanation:
      "KL divergence **regularizes latent space** to enable generation: **Loss function**: L = Reconstruction + β × KL(q(z|x) || N(0,I)). Reconstruction: How well we reconstruct input. KL divergence: How close encoded distribution is to standard normal. **Without KL term** (only reconstruction): Encoder learns: μ = [1000, 5000, -3000, ...] (arbitrary large values), σ ≈ 0 (no variance, deterministic). Latent space: Scattered points in arbitrary regions, huge gaps. Problem: Can't sample! If we sample z ~ N(0,I), it's nowhere near training points. **With KL term**: Forces: μ ≈ 0 (centered), σ ≈ 1 (reasonable variance). Latent space: Centered around origin, continuous coverage. Benefit: Can sample z ~ N(0,I) and get valid reconstructions! **Mathematical intuition**: KL(q || p) = -0.5 × Σ[1 + log σ² - μ² - σ²]. Penalizes: μ far from 0 (μ² term), σ far from 1 (σ² and log σ² terms). **Trade-off**: High β: Smooth latent space, blurry reconstructions. Low β: Sharp reconstructions, fragmented latent space. **Not about**: Training speed (adds computation), reconstruction quality (actually reduces it slightly), decoder overfitting (that's a separate concern). KL divergence is ESSENTIAL for VAE's generative capability.",
  },
  {
    id: 'mc-5',
    question: 'How can autoencoders be used for anomaly detection?',
    options: [
      'By using the latent space dimension as an anomaly score',
      'By measuring reconstruction error: normal data reconstructs well (low error), anomalies reconstruct poorly (high error)',
      'By counting the number of active neurons in the encoder',
      'Autoencoders cannot be used for anomaly detection',
    ],
    correctAnswer: 1,
    explanation:
      'Anomaly detection uses **reconstruction error** as anomaly score: **Training**: Train autoencoder on NORMAL data only. Network learns: What normal data looks like. How to compress and reconstruct normal patterns. **Testing**: Normal example: Encoder-decoder knows these patterns, reconstructs accurately. Reconstruction error: ||x - x̂||² ≈ 0.001 (small). Anomaly: Patterns outside training distribution, decoder struggles. Reconstruction error: ||x - x̂||² ≈ 0.05 (large). **Decision rule**: error > threshold → Anomaly, error ≤ threshold → Normal. **Example** (manufacturing defect detection): Training: 10K images of good products. Testing: Good product: Error = 0.002 (matches training), Defective product: Error = 0.08 (unusual patterns), Set threshold: 0.01 → Detect defect! **Why it works**: Autoencoder models normal data distribution. Anomalies are off-distribution → high reconstruction error. **Advantages over supervised methods**: (1) No anomaly labels needed (anomalies rare/unknown), (2) Learns normal patterns implicitly, (3) Detects novel anomalies (not just known types). **Not about**: Latent dimension (fixed, not score), neuron counting (not informative), and autoencoders are WIDELY used for anomaly detection. Reconstruction error is the key signal for unsupervised anomaly detection.',
  },
];
