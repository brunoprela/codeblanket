import { MultipleChoiceQuestion } from '../../../types';

export const buildingMediaGenerationPlatformMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bmgp-mc-1',
      question:
        'For a media generation platform, how should job queues be prioritized?',
      options: [
        'All jobs treated equally',
        'Multi-tier: Fast queue (TTS, 30s), Standard (images, 2min), Heavy (video, 10min)',
        'Random selection',
        'Newest jobs first',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-tier queues optimize UX and resources: (1) Fast queue (TTS, simple images) - priority 1, max wait 30s, CPU workers, (2) Standard queue (complex images) - priority 2, max wait 2min, T4 GPUs, (3) Heavy queue (video) - priority 3, max wait 10min, A100 GPUs. This prevents slow video jobs from blocking fast text-to-speech, ensures predictable wait times per tier.',
    },
    {
      id: 'bcap-bmgp-mc-2',
      question:
        'What is the most cost-effective way to run GPU workers for image generation?',
      options: [
        'Always use on-demand instances',
        '80% spot instances (70% discount) + 20% on-demand for reliability',
        '100% spot instances',
        'Never use GPUs',
      ],
      correctAnswer: 1,
      explanation:
        '80/20 spot/on-demand balance: (1) Spot instances: 70% cheaper but can be terminated, (2) On-demand: Reliable but expensive. Mix ensures: cost savings (most work on spot) + reliability (on-demand handles interruptions). Auto-scaling: Scale spot first, on-demand when spot unavailable. Typical savings: $500/month vs $1,500 pure on-demand.',
    },
    {
      id: 'bcap-bmgp-mc-3',
      question:
        'How should you validate AI-generated content for NSFW before showing to users?',
      options: [
        'Trust all generated content',
        'Run through AWS Rekognition or CLIP-based NSFW detector during generation',
        'Only check manually after user reports',
        'Never show generated content',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-layer moderation: (1) Pre-generation: Filter prompts with moderation API, (2) Post-generation: Run NSFW detector (AWS Rekognition or open-source CLIP model) during generation, (3) Progressive release: Show blurred preview immediately, full-res after validation passes. Cost: ~$0.001-0.01 per image. Block if NSFW confidence >0.9, human review if 0.7-0.9, approve if <0.7.',
    },
    {
      id: 'bcap-bmgp-mc-4',
      question:
        'What is the best approach to improve video quality from Stable Video Diffusion?',
      options: [
        'Generate at highest resolution directly',
        'Multi-stage: Generate keyframes (Midjourney) → animate (SVD) → interpolate frames (FILM)',
        'Use only SVD without modifications',
        'Manually edit each frame',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-stage pipeline improves quality: (1) Generate high-quality keyframes with Midjourney (locks visual style), (2) Animate with SVD (image-to-video), (3) Frame interpolation with FILM/RIFE (8fps → 24fps for smooth motion), (4) Post-processing: Temporal smoothing (reduce flicker), color correction. This achieves better consistency and smoothness than direct SVD generation.',
    },
    {
      id: 'bcap-bmgp-mc-5',
      question:
        'How should long wait times (2-5 min) be handled in the user interface?',
      options: [
        'Show nothing until complete',
        'Queue position, time estimate, low-res preview, gallery of examples, option to close tab with email notification',
        'Only show a progress bar',
        'Force users to wait on page',
      ],
      correctAnswer: 1,
      explanation:
        'Reduce perceived wait time: (1) Show queue position ("You\'re 3rd in line, ~4 min"), (2) Time estimates from historical data, (3) Low-res preview in 5s (SDXL Turbo), full-res in background, (4) Entertainment: gallery of popular generations, AI facts, (5) Allow leaving: "Safe to close, we\'ll email you." (6) Monetization: "Skip queue for $1." This makes waiting feel productive and less frustrating.',
    },
  ];
