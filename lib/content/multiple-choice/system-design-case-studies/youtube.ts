/**
 * Multiple choice questions for Design YouTube section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const youtubeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'YouTube encodes each uploaded video into 7 different bitrates (144p-4K). For a 10-minute video, how long does the encoding process take using parallel processing across 100 workers?',
    options: [
      '10 minutes (same as video length)',
      '20-30 minutes (2-3x video length with parallelization)',
      '2 hours (encoding is slow)',
      '5 minutes (faster than real-time)',
    ],
    correctAnswer: 1,
    explanation:
      'VIDEO ENCODING PROCESS: 10-min video split into 60 × 10-second chunks. Distribute 60 chunks across 100 workers (each worker gets < 1 chunk). Each worker encodes chunk to 7 bitrates in parallel (using multi-core FFmpeg). Encoding time: 2-3x real-time for high-quality settings. 10-second chunk → 20-30 seconds to encode all 7 versions. Since chunks processed in parallel: Total time ≈ 20-30 minutes (dominated by longest chunk, not sum of all chunks). For popular creators: Pre-allocated dedicated workers → faster processing (10-15 min). For small videos (<1 min): Processing in 2-3 minutes. Trade-off: Faster encoding (lower quality settings) vs better compression (slower encoding). YouTube prioritizes quality for permanent content.',
  },
  {
    id: 'mc2',
    question:
      'Why does YouTube use chunked (segmented) HLS streaming instead of serving the entire video file at once?',
    options: [
      'Chunking is required by law',
      'Enables adaptive bitrate switching (change quality mid-stream), efficient CDN caching, and seek/skip functionality',
      'Reduces storage costs',
      'Makes encoding faster',
    ],
    correctAnswer: 1,
    explanation:
      "CHUNKED STREAMING BENEFITS: (1) ADAPTIVE BITRATE: Each 10-second chunk can be different quality. User starts at 480p (chunk 1-3), switches to 1080p (chunk 4-6) when bandwidth improves. Impossible with single file. (2) CDN CACHING: Each chunk is separate file. Popular chunks (first 30 seconds) cached aggressively at edge. Unpopular parts (minute 45-50 of 1-hour video) served from origin. More efficient than caching entire 1-hour file. (3) SEEK/SKIP: User skips to minute 5 → Download only chunks from minute 5 onward. Don't waste bandwidth on unwatched beginning. (4) FAILURE RESILIENCE: If chunk download fails, retry only that chunk (not entire video). (5) LIVE STREAMING: Generate chunks in real-time as stream progresses. Viewers join mid-stream, download latest chunks. CHUNK SIZE: Too small (2 sec) → Many requests, overhead. Too large (60 sec) → Slow bitrate switching. Optimal: 6-10 seconds for VOD, 2-4 seconds for low-latency live.",
  },
  {
    id: 'mc3',
    question:
      "YouTube's Content ID scans 500 hours of uploaded video per minute for copyright matches. How is this computationally feasible?",
    options: [
      'Manual human review of all uploads',
      'Generate audio/video fingerprints, use approximate nearest neighbor search against 100M reference fingerprints',
      'Compare pixel-by-pixel against all copyrighted works',
      'Only check popular uploads',
    ],
    correctAnswer: 1,
    explanation:
      'CONTENT ID ARCHITECTURE: (1) FINGERPRINTING: Audio fingerprint: Generate spectrogram (frequency analysis), extract robust hash (resistant to compression, noise). Takes ~1 minute for 10-min video. Video fingerprint: Extract keyframes (1 per second), perceptual hash each frame. Takes ~30 seconds. (2) DATABASE: 100M+ reference fingerprints from copyright holders. Indexed in specialized ANN (Approximate Nearest Neighbor) data structure (e.g., LSH, HNSW). (3) MATCHING: Query: "Does this upload match any reference?" ANN search: O(log N) instead of O(N) linear scan. Finds top 100 most similar fingerprints in milliseconds. Threshold: If similarity > 95%, flag as match. (4) PARALLELIZATION: 500 hours/min = 30,000 minutes/min of content. Distribute across 10,000 workers → 3 videos/worker/min (manageable). (5) RESULT: Scan completes in 5-10 minutes (async, doesn\'t block upload). False positive rate: <1% (manual appeals process). This is why Content ID costs millions to build - specialized ML algorithms at scale.',
  },
  {
    id: 'mc4',
    question:
      'A user uploads a 1080p video. YouTube generates 7 bitrates including 144p. Why not just let the player downscale 1080p to 144p on the client side?',
    options: [
      'Client downscaling is illegal',
      'Serving 1080p to 144p viewers wastes 50x bandwidth; pre-encoded 144p is 50x smaller',
      'Client devices cannot downscale video',
      'Server downscaling is always better quality',
    ],
    correctAnswer: 1,
    explanation:
      "BANDWIDTH COMPARISON: 1080p bitrate: 5 Mbps. 144p bitrate: 100 Kbps (0.1 Mbps). Ratio: 50x difference. SCENARIO: Mobile user on 2G network (200 Kbps) in rural India. If served 1080p: Video buffers constantly (needs 5 Mbps, has 0.2 Mbps). Terrible UX. If served 144p: Video plays smoothly (needs 0.1 Mbps, has 0.2 Mbps). Acceptable UX. CLIENT DOWNSCALING PROBLEMS: (1) Wastes user's data plan (50x more download). (2) Wastes YouTube's bandwidth costs (50x more egress). (3) Video buffers (network can't handle 5 Mbps). (4) Device CPU/battery to downscale (not free). PRE-ENCODED BENEFITS: Server encodes once, serves millions of times. 144p file is 50x smaller (better compression than downscaling). Tailored to device capability. STORAGE COST: 7 versions = 7x storage, but storage is cheap ($0.023/GB/month) vs bandwidth ($0.02/GB transfer). Encode once, save bandwidth forever.",
  },
  {
    id: 'mc5',
    question:
      'YouTube recommendations optimize for "watch time" instead of "click-through rate". Why is this a better objective?',
    options: [
      'Watch time is easier to measure',
      'Optimizing for CTR leads to clickbait thumbnails with low satisfaction; watch time correlates with user satisfaction and engagement',
      'Watch time is required by law',
      'CTR optimization is technically impossible',
    ],
    correctAnswer: 1,
    explanation:
      'CTR OPTIMIZATION PROBLEMS: Train model to maximize P(click). Result: Recommend clickbait (sensational thumbnails, misleading titles). Users click, watch 10 seconds, leave disappointed (bounce). Metrics: High CTR (10%), low watch time (10 sec), low satisfaction. Long-term: Users lose trust, leave YouTube. WATCH TIME OPTIMIZATION: Train model to maximize expected_watch_time. Result: Recommend high-quality content users actually want. Users click, watch 8 minutes, satisfied. Metrics: Lower CTR (5%), high watch time (8 min), high satisfaction. Long-term: Users stay on YouTube longer, more engaged, more ad revenue. REAL-WORLD EXAMPLE: Video A: Clickbait thumbnail, 10% CTR, 30 sec avg watch = 3 sec expected value. Video B: Quality content, 5% CTR, 10 min avg watch = 30 sec expected value. Optimize for CTR → Show A. Optimize for watch time → Show B (10x better). YouTube switched from CTR to watch time in 2012 - immediately improved user retention and revenue. KEY INSIGHT: Optimize for long-term business objective (engagement), not short-term vanity metric (clicks).',
  },
];
