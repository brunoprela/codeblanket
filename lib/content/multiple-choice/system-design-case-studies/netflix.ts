/**
 * Multiple choice questions for Design Netflix section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const netflixMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Netflix encodes a 2-hour 4K video into 5 bitrates (4K, 1080p, 720p, 480p, 360p). Approximately how much storage is required for all versions combined?',
    options: [
      '10 GB (similar to original)',
      '50 GB (5 versions × 10 GB each)',
      '100 GB (higher quality requires more space)',
      '500 GB (massive overhead from encoding)',
    ],
    correctAnswer: 1,
    explanation:
      'Each bitrate requires different storage: 4K (25 Mbps) = 22.5 GB, 1080p (8 Mbps) = 7.2 GB, 720p (5 Mbps) = 4.5 GB, 480p (2.5 Mbps) = 2.25 GB, 360p (1 Mbps) = 0.9 GB. Total: ~37 GB, so approximately 50 GB with overhead and manifests. This 5x storage overhead is acceptable because: (1) Storage is cheap ($0.023/GB/month), (2) Encode once, serve millions of times (amortized), (3) Bandwidth savings from lower bitrates far exceed storage costs. This is the fundamental trade-off of ABR streaming.',
  },
  {
    id: 'mc2',
    question:
      'Why does Netflix use HLS (HTTP Live Streaming) instead of traditional RTMP streaming?',
    options: [
      'HLS is faster than RTMP',
      'HLS uses HTTP, leverages existing CDN infrastructure and passes through firewalls',
      'HLS has better video quality',
      'RTMP is newer and less tested',
    ],
    correctAnswer: 1,
    explanation:
      "HLS ADVANTAGES: (1) HTTP-based: Works over port 80/443 (no firewall issues), leverages HTTP caching at every layer (browser, proxy, CDN). (2) CDN-friendly: Chunks are static files, perfect for CDN caching. CloudFront/Akamai optimized for HTTP. (3) Adaptive: Built-in support for multiple bitrates via manifest. (4) Simple: No special streaming servers needed, just static file hosting. RTMP DISADVANTAGES: (1) Custom protocol: Requires special servers (Flash Media Server), doesn't use standard CDN infrastructure. (2) Port 1935: Often blocked by corporate firewalls. (3) Flash-dependent: Adobe deprecated Flash, RTMP usage declined. (4) No built-in ABR: Manual implementation required. RESULT: HLS won for on-demand streaming (Netflix, YouTube). RTMP still used for low-latency live streaming (Twitch ingestion), but even that migrating to WebRTC.",
  },
  {
    id: 'mc3',
    question:
      "Netflix\'s CDN cache hit rate is 95%. What percentage of video requests hit the origin (S3)?",
    options: [
      '5% (cache misses)',
      '95% (cache hits also hit origin)',
      '50% (half and half)',
      '0% (all cached)',
    ],
    correctAnswer: 0,
    explanation:
      '95% cache hit rate means 95% of requests served from CDN edge (no origin contact). 5% cache misses hit origin. IMPACT: Without CDN: 40M concurrent streams × 5 Mbps = 200 Tbps hits S3 directly (impossible - S3 regional limit ~100 Gbps). With 95% hit rate: Only 10 Tbps hits S3 (manageable with multiple regions). This 20x reduction is why CDN is mandatory. COST: 95% of bandwidth served at $0.01/GB (Open Connect), 5% at $0.09/GB (S3 egress). Average cost: $0.014/GB vs $0.09/GB without CDN (6.4x savings). KEY METRIC: Cache hit rate directly impacts origin load and cost. Netflix optimizes for >95% via intelligent pre-positioning.',
  },
  {
    id: 'mc4',
    question:
      'A user starts watching a video on their phone, then switches to their TV. How does Netflix handle "continue watching" across devices?',
    options: [
      'Video progress is not saved',
      'Progress is saved locally on each device separately',
      'Progress is saved to database in real-time as user watches, synced across devices',
      'User must manually mark progress',
    ],
    correctAnswer: 2,
    explanation:
      'IMPLEMENTATION: (1) Player sends heartbeat every 30 seconds with current position: POST /api/watch_progress {user_id, video_id, position: 1230 seconds}. (2) API updates database (Cassandra): INSERT INTO watch_history (user_id, video_id, position, updated_at) VALUES (...). Use UPSERT to overwrite previous position. (3) When user opens TV app: Query: SELECT position FROM watch_history WHERE user_id=X AND video_id=Y. Resume from saved position. (4) Database write batching: Queue writes in Redis, flush every 60 seconds (reduces DB load). TRADE-OFFS: Real-time sync (30-sec lag) vs database load. Some systems batch every 5 minutes (acceptable lag). Netflix chooses 30-sec for better UX. SCHEMA: Primary key (user_id, video_id) ensures single row per user per video (overwrite pattern). This is multi-device sync 101.',
  },
  {
    id: 'mc5',
    question:
      'Why does Netflix use H.265 (HEVC) encoding for 4K content but H.264 for most 1080p content?',
    options: [
      'H.264 has better quality than H.265',
      'H.265 is 50% smaller but requires more encoding time and newer devices; trade-off worthwhile only for 4K',
      'H.265 is free, H.264 requires licensing',
      'H.264 is newer than H.265',
    ],
    correctAnswer: 1,
    explanation:
      'H.265 (HEVC) vs H.264 (AVC): COMPRESSION: H.265 is 50% smaller at same quality (e.g., 4K @ 15 Mbps vs 25 Mbps). For 2-hour 4K video: 22.5 GB (H.264) vs 13.5 GB (H.265) = 9 GB savings. Multiplied by millions of streams = significant bandwidth savings. ENCODING COST: H.265 encoding is 2-5x slower (more complex algorithm). Requires more CPU/GPU, higher encoding cost. COMPATIBILITY: H.264: Supported by 99.9% of devices (iPhones, Androids, old TVs). H.265: Only newer devices (iPhone 7+, 2016+ TVs, recent GPUs). DECISION MATRIX: 1080p: Use H.264 (universal compatibility, bandwidth already manageable at 5-8 Mbps). 4K: Use H.265 (25 Mbps → 15 Mbps reduction critical, only new devices support 4K anyway). Netflix uses device detection: if client supports H.265, serve H.265 stream; else fallback to H.264. FUTURE: AV1 (royalty-free, 30% better than H.265) emerging, but encoding even slower.',
  },
];
