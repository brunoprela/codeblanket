/**
 * Quiz questions for Image Generation Fundamentals section
 */

export const imagegenerationfundamentalsQuiz = [
  {
    id: 'igcv-imgfund-q-1',
    question:
      'Explain how the diffusion process works in text-to-image generation. Why is this approach more stable than GANs for image generation?',
    hint: 'Think about the forward and reverse diffusion processes and training stability.',
    sampleAnswer:
      "Diffusion models work through two processes: **Forward diffusion** progressively adds Gaussian noise to an image until it becomes pure noise (used during training), and **Reverse diffusion** learns to remove noise step-by-step guided by text prompts (used for generation). This is more stable than GANs because: 1) **No adversarial training** - GANs require carefully balanced generator and discriminator which can be unstable, while diffusion models have a single objective. 2) **Gradual refinement** - Each denoising step makes small improvements rather than generating the entire image at once, allowing for more controlled generation. 3) **Better diversity** - GANs can suffer from mode collapse where they generate similar images, while diffusion naturally produces diverse outputs through stochastic sampling. 4) **Training stability** - The loss function is straightforward (predict noise added at each step) versus GAN's adversarial loss that can oscillate. This stability made diffusion models the foundation of modern text-to-image systems like Stable Diffusion and DALL-E.",
    keyPoints: [
      'Forward diffusion adds noise, reverse diffusion removes it',
      'No adversarial training needed unlike GANs',
      'Gradual refinement through multiple steps',
      'Better training stability and output diversity',
    ],
  },
  {
    id: 'igcv-imgfund-q-2',
    question:
      'Why do image generation models work in latent space rather than directly on pixel values? What are the computational and quality implications?',
    hint: 'Consider the size difference and what the VAE accomplishes.',
    sampleAnswer:
      "Working in latent space provides crucial advantages: **Computational efficiency** - A 512×512×3 RGB image has 786,432 values, but its latent representation might be only 64×64×4 (16,384 values), a ~48x reduction. This makes diffusion steps much faster since there's less data to process. **Memory benefits** - Smaller representations fit in GPU memory, allowing larger batch sizes and higher resolutions. **Semantic compression** - The VAE (Variational Autoencoder) learns to compress images into meaningful latent representations that capture essential visual features while discarding noise and irrelevant details. This makes the diffusion model's job easier since it works with semantic concepts rather than raw pixels. **Quality implications** - Despite compression, quality remains high because the VAE is trained to preserve perceptually important information. The decoder can reconstruct fine details from the compact latent code. However, some very fine details might be lost in compression. **Practical impact** - This approach enables consumer GPUs to run these models; working directly in pixel space would require expensive hardware. It\'s the key innovation that made Stable Diffusion \"stable\" and accessible.",
    keyPoints: [
      'Latent space is ~48x smaller than pixel space',
      'VAE provides semantic compression of images',
      'Enables faster generation and lower memory usage',
      'Quality remains high despite compression',
    ],
  },
  {
    id: 'igcv-imgfund-q-3',
    question:
      'You need to choose between Stable Diffusion 2.1, SDXL, and DALL-E 3 for a production application generating product images. What factors would guide your decision?',
    hint: 'Consider quality, speed, cost, control, and infrastructure requirements.',
    sampleAnswer:
      'Decision factors: **Quality requirements** - If highest quality is critical (marketing materials, client-facing), DALL-E 3 excels at prompt following and composition. SDXL is strong but slightly behind. SD 2.1 is adequate for internal use. **Volume and cost** - For high volume (>1000 images/day), SD models running locally have zero per-image cost after initial hardware investment. DALL-E 3 costs $0.04-$0.08 per image, which adds up quickly. For low volume (<100/day), DALL-E 3 might be cheaper than maintaining GPU infrastructure. **Control and customization** - SD models allow fine-tuning on your product style and complete parameter control. DALL-E 3 is API-only with limited customization. **Infrastructure** - SD 2.1 runs on basic GPUs (6GB VRAM), SDXL needs 10GB+, DALL-E 3 needs no infrastructure. **Speed** - SD 2.1 is fastest (~3s), SDXL slower (~10s), DALL-E 3 moderate (~8s) but with API latency. **Recommendation for product images**: Start with DALL-E 3 for prototyping and quality validation. If you reach 1000+ images/month or need specific product style, transition to SDXL with fine-tuning. Use SD 2.1 for internal previews/iterations.',
    keyPoints: [
      'DALL-E 3 best quality but costs per image',
      'SD models: no per-image cost but need infrastructure',
      'Consider volume, quality needs, and budget',
      'Hybrid approach: cloud for low volume, local for high volume',
    ],
  },
];
