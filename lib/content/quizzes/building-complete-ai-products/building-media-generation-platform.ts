export const buildingMediaGenerationPlatformQuiz = [
  {
    id: 'bcap-bmgp-q-1',
    question:
      'Design a media generation platform supporting: text-to-image, image-to-video, and text-to-speech. Each has different requirements: images (10-30s, GPU), video (2-5min, heavy GPU), speech (2-10s, CPU). How do you architect: (1) Job queue prioritization, (2) GPU resource management, (3) Cost optimization, (4) User experience during long waits?',
    sampleAnswer:
      'Architecture: Multi-tier job queue system. Queues: (1) Fast queue (TTS, simple images) - priority 1, max wait 30s. (2) Standard queue (complex images) - priority 2, max wait 2min. (3) Heavy queue (video) - priority 3, max wait 10min. Workers: (1) CPU workers (5-10) for TTS (ElevenLabs API). (2) GPU workers - T4 (5) for images (Stable Diffusion), A100 (2) for video (Stable Video Diffusion). Resource management: (1) Dynamic allocation: Scale GPU workers based on queue depth (Kubernetes HPA). (2) Spot instances: Use spot for 70% cost savings, have 20% on-demand for reliability. (3) Batching: Process multiple images on same GPU (batch size 4-8). (4) Model caching: Keep models in GPU memory (avoid reload overhead). Cost optimization: (1) Route simple prompts to fast models (SDXL Lightning for images). (2) Offer "slow" tier: queue during off-peak hours (50% discount). (3) Cache: Store generated media, if similar prompt exists, return cached (hash prompt + params). UX: (1) Show position in queue (e.g., "15th in line, ~5min wait"). (2) Progress bar with time estimate. (3) Email notification when ready. (4) Allow payment to jump queue ($1 = front of line). (5) Generate low-res preview fast (SDXL Turbo), upgrade to high-res in background. Monitoring: Alert if: queue depth >100, GPU utilization <60% (waste), p95 wait time >10min.',
    keyPoints: [
      'Multi-tier queues: fast (TTS), standard (images), heavy (video)',
      'GPU resource management: dynamic scaling, spot instances, batching',
      'Cost optimization: model routing, off-peak discounts, caching',
      'UX: queue position, time estimates, notifications, priority upgrades',
      'Monitor: queue depth, GPU utilization, wait times',
    ],
  },
  {
    id: 'bcap-bmgp-q-2',
    question:
      "Your video generation platform uses Stable Video Diffusion (SVD). Users complain videos have flickering, inconsistent motion, and don't match prompts. How do you: (1) Improve video quality, (2) Add motion control, (3) Enable style consistency, (4) Validate output quality? Compare SVD, Runway, and Pika - when would you use each?",
    sampleAnswer:
      'Quality improvements: (1) Prompt engineering: Use detailed prompts with motion descriptors ("camera slowly pans right", "subject walks forward"). (2) Frame interpolation: Generate at lower fps (8fps), use FILM/RIFE to interpolate to 24fps (smoother motion). (3) Multi-stage: Generate keyframes first (Midjourney), animate with SVD (more control). (4) Post-processing: Temporal smoothing (reduce flicker), color correction. Motion control: (1) SVD: Use motion bucket parameter (low = subtle, high = dramatic). (2) ControlNet: Provide depth maps or pose sequences for motion guidance. (3) AnimateDiff: Fine-tune on specific motion styles. Style consistency: (1) Use same seed across generations. (2) LoRA fine-tuning on specific style. (3) Image-to-video: Start from consistent image (locks visual style). Quality validation: (1) Automated: Check for flicker (frame-to-frame diff), motion blur, artifacts. (2) ML classifier: Train model to detect "good" vs "bad" videos. (3) Sample validation: Human review 10% of outputs. Model comparison: SVD - open source, self-host, full control, but requires expertise. Runway Gen-2 - best quality ($0.50/sec), API easy, expensive. Pika - fast (3s max), good for iterations ($1/video). Use SVD for: High-volume, cost-sensitive, need control. Runway for: Premium quality, customer-facing. Pika for: Rapid prototyping, short clips. Hybrid: Pika for preview (fast, cheap), Runway for final (quality).',
    keyPoints: [
      'Improve quality: detailed prompts, frame interpolation, multi-stage generation',
      'Motion control: motion bucket params, ControlNet, depth maps',
      'Style consistency: same seed, LoRA fine-tuning, image-to-video',
      'Validation: automated (flicker detection), ML classifier, human sampling',
      'Model selection: SVD (control), Runway (quality), Pika (speed)',
    ],
  },
  {
    id: 'bcap-bmgp-q-3',
    question:
      "Design a moderation system for AI-generated media. Requirements: flag NSFW content, copyrighted characters, deepfakes, and policy violations before showing to users. How do you balance: (1) Speed (can't delay generation), (2) Accuracy (no false positives), (3) Cost? Include: pre-generation filtering, post-generation validation, and user reporting.",
    sampleAnswer:
      'Multi-layer moderation: (1) Pre-generation (prompt filtering). (2) Post-generation (content analysis). (3) User reporting + appeals. Pre-generation: (1) Prompt moderation: Run prompt through OpenAI moderation API (<100ms, $0.0001). Reject if flagged. (2) Keyword blacklist: Fast check (regex) for explicit terms, copyrighted characters ("Mickey Mouse", "Harry Potter"). (3) LLM classifier: "Is this prompt attempting to generate harmful content?" (Claude Haiku, <1s, $0.001). Block if confidence >0.9. Post-generation: (1) NSFW detection: Use AWS Rekognition or open-source CLIP model. Check for: nudity, violence, graphic content. (2) Copyright: CLIP-based similarity search against database of copyrighted characters/logos. If similarity >0.85, flag. (3) Deepfake: Detect if generation is mimicking specific person (face recognition). (4) Quality: Blurry/distorted images likely failed generation, auto-reject. Speed optimization: (1) Run fast checks (NSFW) during generation. (2) Queue expensive checks (copyright) for post-processing. (3) Progressive release: Show blurred preview immediately, full-res after validation. Accuracy: (1) Use multiple models (ensemble). (2) Confidence thresholds: >0.95 = auto-reject, 0.7-0.95 = human review, <0.7 = approve. (3) A/B test: Compare false positive rates across models. Cost: NSFW (free/cheap) + copyright ($0.01) + LLM ($0.001) = $0.011 per generation. User reporting: Flag button → human review within 24hr → if violation, remove + temp ban user.',
    keyPoints: [
      'Multi-layer: pre-generation (prompt filter), post-generation (content analysis), user reports',
      'Pre-filter fast: moderation API, keyword blacklist, LLM classifier',
      'Post-generation: NSFW, copyright similarity, deepfake detection',
      'Speed optimization: run fast checks during generation, queue expensive after',
      'Balance accuracy: confidence thresholds, ensemble models, human review for edge cases',
    ],
  },
];
