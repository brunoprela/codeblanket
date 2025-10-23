/**
 * Quiz questions for Stable Diffusion section
 */

export const stablediffusionQuiz = [
  {
    id: 'igcv-sd-q-1',
    question:
      'Compare the practical differences between SD 2.1 and SDXL for a production application. When would the extra GPU requirements and slower speed of SDXL be justified?',
    hint: 'Consider quality, speed, hardware costs, and use case requirements.',
    sampleAnswer:
      'SD 2.1 vs SDXL production comparison: **Hardware requirements** - SD 2.1 needs 4-6GB VRAM (RTX 3060), SDXL needs 10-12GB (RTX 3080+). For cloud deployment, SDXL requires more expensive instances ($1.50/hr vs $0.75/hr). **Speed difference** - SD 2.1 generates in ~3-5 seconds, SDXL takes ~10-15 seconds at same step count. For high-volume applications (1000s/day), this compounds. **Quality gap** - SDXL produces noticeably better detail, composition, and photorealism. Text rendering is significantly improved. **Justification for SDXL**: 1) **Quality-critical applications** - Marketing materials, client deliverables, print materials where quality directly impacts business. 2) **Complex prompts** - SDXL handles intricate scene descriptions better. 3) **Text in images** - Signs, labels, typography. 4) **Larger outputs** - Native 1024×1024 vs 512×512 upscaled. **Stick with SD 2.1 for**: 1) **High volume** - Cost and speed matter more than quality. 2) **Iteration/prototyping** - Fast feedback loops crucial. 3) **Hardware constraints** - Limited GPU budget. 4) **Real-time applications** - Speed requirements. **Hybrid approach**: SD 2.1 for previews/iterations, SDXL for final generation. Or SD 2.1 for bulk, SDXL for featured/important images.',
    keyPoints: [
      'SDXL needs 2x GPU memory and 3x generation time',
      'SDXL provides significantly better quality and detail',
      'Justify SDXL for quality-critical, client-facing work',
      'Use SD 2.1 for high volume, iteration, speed-critical tasks',
    ],
  },
  {
    id: 'igcv-sd-q-2',
    question:
      'Explain how the choice of scheduler (Euler A, DPM++, DDIM) affects image generation. How would you determine the optimal scheduler for your use case?',
    hint: 'Consider speed, quality, determinism, and convergence patterns.',
    sampleAnswer:
      'Scheduler comparison and selection: **What schedulers do** - Control how noise is removed at each step during diffusion. Different algorithms trade off speed vs quality vs randomness. **Euler Ancestral (euler_a)**: Introduces stochasticity even with same seed (slightly different results each time). Fast convergence, good quality at low step counts (20-25). Popular for general use. Non-deterministic nature good for creative variation. **DPM++ 2M Karras**: Deterministic (same seed = same result), high quality, slower convergence needs 30-40 steps. Excellent for photorealism and final outputs. Smoother, more coherent images. **DDIM**: Original Stable Diffusion scheduler, very deterministic, stable but not fastest. Good baseline, predictable behavior. Needs 40-50 steps for best quality. **Selection methodology**: 1) **Test with your content** - Generate same prompt with each scheduler at various step counts. 2) **Measure**what matters - If speed critical, compare quality at 20 steps. If quality critical, compare at 50 steps. 3) **Consider determinism** - Need exact reproducibility? Avoid euler_a. 4) **Typical recommendations** - **Fast iteration**: euler_a at 20-25 steps. **High quality**: dpm++ at 35-40 steps. **Consistency**: ddim at 40 steps. **Production strategy**: euler_a for user-facing (variety desired), dpm++ for templates (consistency needed).',
    keyPoints: [
      'Schedulers control noise removal algorithm',
      'Euler A: fast, creative, non-deterministic',
      'DPM++: high quality, deterministic, slower',
      'Test with your content to find optimal choice',
    ],
  },
  {
    id: 'igcv-sd-q-3',
    question:
      'Design a memory optimization strategy for running Stable Diffusion on limited VRAM (6GB). What techniques would you combine to maximize image size and batch size?',
    hint: 'Consider attention slicing, VAE tiling, CPU offload, and resolution strategies.',
    sampleAnswer:
      'Comprehensive VRAM optimization for 6GB GPU: **Base optimizations (apply all)**: 1) **Attention slicing** - `pipe.enable_attention_slicing("max")` reduces memory for attention layers at slight speed cost. Saves ~1-2GB. 2) **xformers** - `pip install xformers` then `pipe.enable_xformers_memory_efficient_attention()`. Faster AND less memory. Critical optimization. 3) **float16** - Use `torch_dtype=torch.float16` instead of float32. Cuts model size in half (~4GB vs 8GB). **Advanced techniques**: 4) **VAE slicing** - `pipe.enable_vae_slicing()` for high-res images. Processes image in tiles during decode. 5) **CPU offload** - `pipe.enable_model_cpu_offload()` moves model parts to RAM when not in use. Significant VRAM savings but slower. 6) **Sequential CPU offload** - Even more aggressive, keeps only active layer on GPU. **Resolution strategies**: On 6GB with all optimizations, you can do: - 512×512: 4 images batch - 768×768: 1-2 images batch - 1024×1024: 1 image with CPU offload. **Alternative: Tiled generation** - Generate 512×512 tiles, stitch together for larger image. **Monitoring**: Track `torch.cuda.memory_allocated()` to find bottlenecks. **Production recommendation**: Enable all base optimizations always. Use CPU offload only for >768 resolution or if speed acceptable. Consider multiple 512×512 generations instead of one 1024×1024.',
    keyPoints: [
      'Attention slicing + xformers + float16 are essential',
      'VAE slicing for high resolution decode',
      'CPU offload trades speed for memory',
      '6GB can do 512×512 batch or 768×768 single with optimizations',
    ],
  },
];
