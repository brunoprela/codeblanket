export const lipSyncAvatarGenerationQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Compare the open-source approach (Wav2Lip, SadTalker) with commercial APIs (D-ID, HeyGen) for avatar generation. Design a system that uses both strategically based on use case, volume, and quality requirements. How would you handle failover and cost optimization?',
    sampleAnswer:
      'Open-source (Wav2Lip/SadTalker): Pros - no per-generation cost (just GPU), full control, privacy, unlimited generations. Cons - infrastructure management, GPU costs ($1-2/hour), quality can be inconsistent, requires ML expertise. Commercial APIs: Pros - consistent high quality, no infrastructure, latest improvements, easy integration. Cons - $0.20-1.00 per generation, usage limits, data sent to third party, less customization. Strategic use: Use open-source for: high volume (>1000/day makes economics favorable), privacy-sensitive content, custom avatar training, development/testing. Use commercial for: low volume startup phase (<100/day), highest quality needs, minimal technical team, unpredictable spikes. Hybrid architecture: 1) Route bulk/batch requests to self-hosted (overnight processing OK), 2) Use commercial APIs for real-time/priority requests, 3) Implement quality fallback (if open-source result scores <0.80, retry with commercial), 4) Load balance based on current GPU utilization. Cost optimization: ~$50/month baseline for GPU (covers ~1500 generations), commercial API costs $150-1000/month for same volume. Break-even at ~500 generations/month. Failover: monitor self-hosted success rate, if <90% route to API temporarily, alert engineering team, investigate issues, return to self-hosted when resolved. Implementation: unified interface abstracting both backends, feature flags for A/B testing, metrics tracking (quality score, cost, latency) per backend.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Design a video translation system that not only translates the audio but also adjusts lip movements to match the new language. What are the technical challenges and how would you ensure natural-looking results across different languages with varying phoneme counts?',
    sampleAnswer:
      'Video translation with lip sync pipeline: 1) Transcribe original audio (Whisper), 2) Translate text to target language (GPT-4), 3) Generate target TTS with voice cloning, 4) Detect face and track across frames, 5) Modify lip movements using lip sync model (Wav2Lip), 6) Composite back into video preserving quality. Technical challenges: Phoneme duration mismatch - Spanish typically 10-15% shorter than English, lip sync must handle timing differences. Solution: use prosody matching in TTS to adjust speed, time-stretch video slightly if needed (±10% imperceptible), smart keyframe selection for lip sync. Mouth shape variety across languages - some languages use mouth shapes rarely in others. Solution: train language-specific lip sync models or use multilingual training. Visual quality - direct lip replacement can look fake, especially HD video. Solution: blend lip region using Poisson blending, match color/lighting, preserve teeth, add subtle motion blur. Ensuring natural results: 1) Voice cloning from original speaker for consistency, 2) Preserve speaking style and emotion in translation, 3) Temporal smoothing of lip movements, 4) Quality checks (lip sync confidence score, visual artifacts detection), 5) A/B testing with native speakers. Implementation: process in 30-second chunks, parallel processing per chunk, progressive enhancement (quick draft, then refine), manual review for quality control. Cost: ~$0.50-2.00 per minute of video. Applications: content localization, accessibility, e-learning courses, marketing videos.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      'Explain how you would build a real-time avatar system for virtual meetings or customer service that generates lip-synced responses with minimal latency. What are the key technical constraints and how would you optimize the pipeline for real-time performance?',
    sampleAnswer:
      'Real-time avatar system requirements: end-to-end latency <500ms for natural conversation. Pipeline: 1) Voice input (streaming), 2) Speech-to-text (real-time Whisper), 3) LLM response generation (streaming), 4) TTS (streaming ElevenLabs turbo), 5) Lip sync (real-time model), 6) Render and stream video. Latency breakdown: STT 100-200ms, LLM first token 200-300ms, TTS first chunk 100-200ms, lip sync 50-100ms, render 50ms = total 500-850ms. Optimization strategies: 1) Anticipatory rendering - pre-render common avatar states (idle, thinking), 2) Overlap processing - start TTS while LLM still generating, begin lip sync on first audio chunk, 3) Lower quality modes - reduce avatar resolution/FPS under load, 4) Predictive buffering - maintain 200ms audio buffer for smooth playback, 5) GPU optimization - use TensorRT for lip sync model (2-3x faster), 6) Caching - cache frequent responses with pre-rendered avatars. Technical constraints: GPU required for real-time lip sync (CPU too slow), network bandwidth (stream video at 1-2 Mbps), audio sync critical (A/V drift noticeable at >100ms). Architecture: WebRTC for low-latency streaming, edge deployment to reduce network latency, dedicated GPU per 5-10 concurrent users, graceful degradation (static avatar if lip sync lags). Quality vs latency trade-off: at <300ms can use full quality, 300-500ms reduce avatar FPS to 15, >500ms switch to static avatar with audio only. Monitor: end-to-end latency p50/p95, lip sync quality score, user experience metrics (conversation naturalness).',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
