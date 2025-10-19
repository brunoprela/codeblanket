/**
 * Generative Adversarial Networks (GANs) Multiple Choice Questions
 */

export const generativeAdversarialNetworksMultipleChoice = [
  {
    id: 'mc-1',
    question:
      'In GAN training, what is the relationship between the generator and discriminator?',
    options: [
      'They work cooperatively to minimize the same loss function',
      'They compete in an adversarial game: generator tries to fool discriminator, discriminator tries to detect fakes',
      'The generator trains first, then the discriminator uses the trained generator',
      'They are two versions of the same network with shared weights',
    ],
    correctAnswer: 1,
    explanation:
      "GANs use **adversarial training** - two networks compete in a minimax game: **Generator (G)**: Goal: Fool discriminator. Strategy: Generate fake data that looks real. Loss: Wants D(G(z)) → 1 (discriminator thinks fakes are real). **Discriminator (D)**: Goal: Distinguish real from fake. Strategy: Correctly classify real as 1, fake as 0. Loss: Wants D(real) → 1 and D(fake) → 0. **Minimax game**: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]. D tries to maximize V (correctly classify), G tries to minimize V (fool D). **Training dynamics**: Alternate updates: (1) Train D: Improve at detection, (2) Train G: Improve at fooling, (3) Repeat. Over time, both improve. **Equilibrium**: p_g = p_data (generator distribution matches real data), D(x) = 0.5 everywhere (can't tell real from fake). **Not cooperative**: They have opposing objectives! **Not sequential**: Train alternately, not one then the other. **Not shared weights**: Separate networks with different parameters. The adversarial competition drives both networks to improve, eventually producing realistic synthetic data.",
  },
  {
    id: 'mc-2',
    question: 'What is mode collapse in GAN training?',
    options: [
      'When the discriminator becomes too powerful and the generator stops learning',
      'When the generator produces only a limited variety of outputs, failing to capture the full diversity of the real data distribution',
      'When both generator and discriminator losses collapse to zero',
      'When the training process crashes due to numerical instability',
    ],
    correctAnswer: 1,
    explanation:
      "Mode collapse: Generator produces **limited diversity**, ignoring most of the data distribution: **Normal behavior**: Generator learns p_g ≈ p_data (full distribution). MNIST: Generates all digits 0-9 with varied styles. Face dataset: Generates diverse faces (ages, genders, expressions). **Mode collapse**: Generator learns to produce only subset of modes. Example: Only generates digit \"8\", never 0-7 or 9. Why? If G finds ONE sample that fools D, it exploits it. Loss goes down by always generating that sample! **Types**: (1) **Complete collapse**: Single output for all inputs, (2) **Partial collapse**: Limited variety (e.g., 3 types of faces), (3) **Mode hopping**: Cycles through different modes during training. **Causes**: (1) **Generator exploits D weakness**: D can't detect one fake → G produces only that, (2) **Lack of diversity incentive**: Loss doesn't penalize repeating outputs, (3) **Training instability**: Imbalanced G-D power dynamics. **Not about**: Discriminator being too strong (related but different issue), losses going to zero (actually opposite - losses reasonable), numerical crashes (that's exploding gradients). **Solutions**: Mini-batch discrimination, unrolled GANs, Wasserstein GAN with gradient penalty. Mode collapse is one of THE hardest problems in GAN training.",
  },
  {
    id: 'mc-3',
    question:
      'Why do we use the non-saturating loss -log D(G(z)) for training the generator instead of the original log(1-D(G(z)))?',
    options: [
      'It is computationally faster to compute',
      'It provides stronger gradients when D correctly identifies fakes (D(G(z)) near 0), helping G learn even when D is strong',
      'It is theoretically more correct according to the minimax formulation',
      'It requires less memory during training',
    ],
    correctAnswer: 1,
    explanation:
      "Non-saturating loss solves the **vanishing gradient problem** early in training: **Original loss**: L_G = E[log(1 - D(G(z)))]. Problem: When D is strong, D(G(z)) ≈ 0 (correctly identifies fakes). Then log(1 - 0) = log(1) = 0. Gradient: ∂L_G/∂G ≈ 0 (vanishing!). G receives NO learning signal when it needs it most (early training when generating bad fakes). **Non-saturating loss**: L_G = -E[log D(G(z))]. When D is strong: D(G(z)) ≈ 0, then -log(0) → +∞ (large loss!). Gradient: ∂L_G/∂G is LARGE. G receives strong signal to improve even when D is strong. **Gradient comparison**: Original: Gradient ∝ (1-D) × D × (1-D) = small when D is strong. Non-saturating: Gradient ∝ 1/D = large when D is strong. **Example** (early training): G generates obvious fakes, D outputs 0.01 (99% sure it's fake). Original loss gradient: ≈ 0.01 (tiny, slow learning). Non-saturating gradient: ≈ 100 (large, fast learning). **Trade-offs**: Non-saturating: Better gradients, faster initial learning. Original: Theoretically matches minimax (but impractical). Computation/memory nearly identical. Standard practice: Use non-saturating loss for G training!",
  },
  {
    id: 'mc-4',
    question:
      'What is the purpose of conditioning in a Conditional GAN (cGAN)?',
    options: [
      'To make training faster by providing additional information',
      'To control the output by providing extra input (e.g., class label) that specifies what type of data to generate',
      'To prevent mode collapse by forcing diversity',
      'To reduce the number of parameters in the generator',
    ],
    correctAnswer: 1,
    explanation:
      'Conditioning enables **controlled generation** by providing additional input: **Standard GAN**: Input: Random noise z. Output: Random sample from learned distribution. No control over what\'s generated! **Conditional GAN**: Input: Noise z + condition y (e.g., class label, text, image). Output: Sample matching condition y. Full control over output type! **Example** (MNIST digit generation): Standard GAN: Generate random digit (might be 0, 5, or 9 - unpredictable). Conditional GAN: G(z, y=7) → generates digit "7", G(z, y=3) → generates digit "3". **Architecture changes**: Generator: G(z, y) - concatenate noise and label embedding. Discriminator: D(x, y) - check if x is real AND matches y. Loss: Real label=1 only if image is real AND matches condition. **Applications**: (1) **Class-conditional image generation**: Specify object class, (2) **Text-to-image**: Condition on text description, (3) **Image-to-image translation**: Condition on source image (pix2pix), (4) **Attribute control**: Specify age, gender, emotion for face generation. **Not about**: Training speed (conditioning adds computation), preventing mode collapse (helps but not primary purpose), reducing parameters (adds embedding parameters). Conditioning transforms GANs from uncontrolled to controlled generation - fundamental capability upgrade!',
  },
  {
    id: 'mc-5',
    question:
      'What does a Fréchet Inception Distance (FID) score measure, and how should it be interpreted?',
    options: [
      'The accuracy of the discriminator; higher FID means better discriminator',
      'The distance between real and generated image distributions in feature space; lower FID means generated images are more similar to real images',
      'The diversity of generated images; higher FID means more diverse outputs',
      'The training stability; lower FID means more stable training',
    ],
    correctAnswer: 1,
    explanation:
      "FID measures **distribution similarity** in learned feature space - lower is better: **Computation**: (1) Extract features from real and fake images using Inception network (pre-trained classifier), (2) Model feature distributions: Real ~ N(μ_r, Σ_r), Fake ~ N(μ_g, Σ_g), (3) Calculate FID: distance between these Gaussian distributions, FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g)). **Interpretation**: FID = 0: Perfect match (generated = real distribution). FID = 10-20: Excellent quality. FID = 20-50: Good quality. FID > 100: Poor quality (visibly different from real). **What it captures**: (1) **Visual similarity**: Features encode image content, (2) **Diversity**: Covariance Σ captures variety, (3) **Realism**: Mean μ captures typical appearance. **Example scores**: Real MNIST vs. real MNIST: ≈ 0. Good GAN on MNIST: ≈ 5-10. Poor GAN on MNIST: ≈ 50-100. **Advantages**: (1) Sensitive to both quality AND diversity, (2) Correlates well with human judgment, (3) More robust than Inception Score. **Not about**: Discriminator accuracy (FID doesn't use discriminator!), diversity alone (also measures quality), training stability (offline metric after training). FID is the GOLD STANDARD for evaluating GAN-generated image quality.",
  },
];
