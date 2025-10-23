export const speechToTextWhisperQuiz = [
  {
    id: 'vag-q-1',
    question:
      'Design a production speech-to-text system using Whisper that handles multiple languages, speaker diarization, and generates accurate timestamps for subtitle creation. How would you optimize for both cost and latency while maintaining high accuracy?',
    sampleAnswer:
      'Production system architecture: 1) VAD preprocessing to skip silence (reduces processing time by 30-50%), 2) Language detection on first 30 seconds to avoid processing entire audio, 3) Dynamic model selection: tiny for real-time preview, base for most cases, large only for critical accuracy needs, 4) Speaker diarization using pyannote.audio run in parallel with transcription, 5) Word-level timestamps for precise subtitle sync. Optimization strategies: batch similar-language requests together, cache common phrases/segments, use faster-whisper (CTranslate2) for 2-4x speedup, process on CPU for very short audio (<30s) to avoid GPU queue wait. Cost vs latency trade-offs: tiny model (32x realtime, $0.01/min) for drafts, base model (16x realtime, $0.02/min) for production, large model (1x realtime, $0.10/min) only when accuracy is critical. Implement progressive enhancement: return fast transcription immediately, run higher quality model in background, notify user when improved version ready. Monitor accuracy with WER (Word Error Rate) benchmarks per language/domain.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-2',
    question:
      'Explain how you would build a meeting transcription service that not only transcribes speech but also identifies speakers, generates summaries, and extracts action items. What are the technical challenges with long-form audio (1+ hour meetings) and how would you address them?',
    sampleAnswer:
      'Complete meeting transcription pipeline: 1) Audio preprocessing (noise reduction, normalization), 2) VAD to identify speech segments, 3) Speaker diarization (pyannote.audio) to identify who spoke when, 4) Whisper transcription with timestamps, 5) Align transcription with speaker labels, 6) LLM post-processing (GPT-4) for summaries and action items. Technical challenges with long audio: 1) Memory constraints - Whisper processes entire audio at once, solution: split into 5-minute chunks with 1-second overlap, process in parallel, merge results. 2) Speaker drift - diarization accuracy decreases over time, solution: re-run diarization every 15 minutes with embedding consistency checks. 3) Context maintenance for summaries - LLM context limits, solution: use map-reduce approach (summarize chunks, then summarize summaries). 4) Cost control - 1-hour meeting costs $5-10 to process, solution: offer tiered service (basic auto-transcript vs full AI analysis). Implementation: async processing with progress updates, store intermediate results, enable resume on failure, provide real-time preliminary transcript with post-processing refinement. Quality metrics: speaker identification accuracy >90%, WER <5%, action item extraction F1 >0.85.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'vag-q-3',
    question:
      "Compare the trade-offs between using OpenAI's Whisper API versus self-hosting the open-source model. Under what circumstances would each approach be preferable, and how would you implement a hybrid system that leverages both?",
    sampleAnswer:
      'OpenAI Whisper API: Pros - no infrastructure, automatic updates, simple integration, 99.9% uptime. Cons - $0.006/minute (~$0.36/hour), 25MB file limit, internet dependency, no customization, data privacy concerns. Self-hosted: Pros - $0.50-2/hour GPU cost (cheaper at scale), unlimited file size, complete control, data stays private, can fine-tune. Cons - infrastructure management, GPU provisioning, model updates, scaling complexity. Break-even: ~100 hours/month (~3.3 hours/day). Choose API for: unpredictable workload, <100 hours/month, need latest model, limited ML expertise, startup phase. Choose self-hosted for: >100 hours/month, predictable load, privacy requirements, custom vocabulary needs, existing GPU infrastructure. Hybrid implementation: 1) Use API for spikes beyond self-hosted capacity, 2) Route short audio (<5 min) to API (queue avoidance), 3) Batch long audio to self-hosted, 4) Use API as failover when self-hosted down, 5) A/B test accuracy between versions. Monitor costs real-time, implement smart routing: if self-hosted queue >10 minutes, route to API; if daily API cost exceeds $X, queue for self-hosted. Track accuracy metrics per source to ensure quality parity.',
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
