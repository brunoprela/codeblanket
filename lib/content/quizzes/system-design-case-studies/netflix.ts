/**
 * Quiz questions for Design Netflix section
 */

export const netflixQuiz = [
  {
    id: 'q1',
    question:
      "Explain adaptive bitrate streaming (ABR) in detail. How does the client player decide which bitrate to use, and why is this critical for Netflix's user experience?",
    sampleAnswer:
      'ADAPTIVE BITRATE STREAMING: Netflix encodes each video into multiple bitrates: 4K (25 Mbps), 1080p (8 Mbps), 720p (5 Mbps), 480p (2.5 Mbps), 360p (1 Mbps). Each version split into 10-second chunks. PLAYER ALGORITHM: (1) START LOW: Begin with 360p (fast start, 1 Mbps = 1.25 MB for 10sec chunk, downloads in <1 sec even on slow network). User sees video playing immediately (<2 sec start time). (2) MEASURE BANDWIDTH: While playing first chunk, player downloads second chunk and measures: Download time for 1.25 MB chunk = 0.5 sec → Bandwidth = 2.5 MB / 0.5s = 5 MB/s = 40 Mbps. (3) SWITCH UP: If bandwidth > 8 Mbps, switch to 720p for next chunk. If bandwidth > 16 Mbps, switch to 1080p. (4) ADAPTIVE: Continuously monitor. If bandwidth drops (network congestion), switch down to avoid buffering. (5) BUFFER: Maintain 30-60 seconds of buffered video. If buffer drains below 10 seconds, immediately downgrade bitrate. WHY CRITICAL? (1) USERS ON SLOW NETWORKS: Start immediately at low quality, upgrade as possible. Without ABR, 1080p would buffer for 30+ seconds (poor UX). (2) MOBILE DATA: Automatically use lower bitrates on cellular (saves data, reduces cost). (3) NETWORK VARIABILITY: Home WiFi can fluctuate 5-50 Mbps. ABR adapts seamlessly, no buffering. (4) REBUFFER RATE: Netflix target <0.5% (users rarely see loading spinner). ABR achieves this by dynamically adjusting. HLS MANIFEST: manifest.m3u8 file lists all bitrate options. Player downloads manifest, chooses bitrate, downloads chunks. KEY INSIGHT: Fixed bitrate = poor UX (buffering or wasted bandwidth). ABR = smooth experience for all network conditions.',
    keyPoints: [
      'Encode each video to 5+ bitrates (360p-4K)',
      'Start low (360p) for fast start (<2 sec)',
      'Measure bandwidth, switch up if available',
      'Switch down if bandwidth drops (avoid buffering)',
      'HLS manifest lists all options, player decides',
    ],
  },
  {
    id: 'q2',
    question:
      'Netflix serves 40 million concurrent streams at 5 Mbps average = 200 Tbps bandwidth. Explain why Netflix built Open Connect (custom CDN inside ISPs) instead of using only third-party CDNs like CloudFront.',
    sampleAnswer:
      "BANDWIDTH COST PROBLEM: 40M streams × 5 Mbps × 3 hours/day = 2.7 PB/day bandwidth. At typical CDN costs ($0.05/GB): 2.7 PB × $50/TB = $135K/day = $49M/year. Reality: Netflix streams 100+ PB/month → $5B/year in bandwidth costs (would make business unprofitable). THIRD-PARTY CDN LIMITATIONS: (1) Cost: $0.02-0.09/GB egress (expensive at petabyte scale). (2) Congestion: Traffic from CDN → ISP can be congested during peak hours (8-11 PM). (3) Peering agreements: CDN providers pay ISPs for bandwidth (cost passed to Netflix). OPEN CONNECT SOLUTION: Netflix places custom appliances (servers with 100-200 TB storage) INSIDE ISP networks (Comcast, Verizon, AT&T). BENEFITS: (1) COST: ISPs give Netflix free rack space (symbiotic: Netflix traffic stays local, doesn't congest ISP's upstream links). Cost reduced to $0.01/GB or less (power + hardware depreciation). (2) LATENCY: User → Netflix appliance inside same ISP = 5-20ms (vs 50-100ms from external CDN). (3) CAPACITY: No inter-ISP bandwidth limits. Comcast users stream from Comcast's Netflix appliances. (4) QUALITY: Reduced packet loss, jitter. Better video quality and lower rebuffer rate. DEPLOYMENT: Netflix pre-distributes popular content to Open Connect appliances every night (push CDN). Top 100 titles = 80% of traffic, fits in 10-20 TB. Long-tail content served from CloudFront (fallback). RESULT: 95% of traffic served from Open Connect, 5% from third-party CDNs. Bandwidth cost reduced 5-10x vs pure CDN approach. KEY INSIGHT: At Netflix scale (10% of global internet traffic), custom CDN inside ISPs is only economically viable solution. No third-party CDN can match the cost/performance.",
    keyPoints: [
      'Pure CDN cost: $5B/year (unsustainable)',
      'Open Connect: Appliances inside ISPs (Comcast, Verizon)',
      'ISPs provide free rack space (traffic stays local)',
      'Cost reduced to $0.01/GB (5-10x savings)',
      '95% traffic from Open Connect, 5% from CloudFront fallback',
    ],
  },
  {
    id: 'q3',
    question:
      'Design the video encoding pipeline for Netflix. A content provider uploads a 100 GB 4K raw video. Walk through the encoding process, parallelization strategy, and time to completion.',
    sampleAnswer:
      'VIDEO ENCODING PIPELINE: INPUT: 100 GB 4K raw video (H.264, MP4), 2 hours duration. TARGET: Generate HLS/DASH adaptive bitrate versions. PIPELINE STEPS: (1) UPLOAD: Content provider uploads to S3 bucket (dedicated upload accelerated endpoint). 100 GB at 1 Gbps = 800 seconds (~13 minutes). (2) TRIGGER ENCODING: S3 event triggers Lambda → Publishes job to SQS queue: {video_id, s3_uri, output_formats: [4K, 1080p, 720p, 480p, 360p]}. (3) SEGMENTATION: Encoding worker downloads video, splits into 10-second chunks: 2 hours = 7,200 seconds = 720 chunks. Each chunk: ~140 MB (100 GB / 720). (4) PARALLEL ENCODING: Distribute 720 chunks across 100 worker instances (AWS Elemental MediaConvert or EC2 spot instances). Each worker: Takes 1 chunk (140 MB), encodes to 5 bitrates (4K, 1080p, 720p, 480p, 360p), produces 5 output chunks (~3 minutes encoding time per chunk per bitrate with parallel FFmpeg). (5) OUTPUT: Each worker uploads encoded chunks to S3: s3://netflix-videos/video123/4K/chunk_001.ts, chunk_002.ts, ..., s3://netflix-videos/video123/1080p/chunk_001.ts, .... (6) MANIFEST GENERATION: After all chunks complete, generate HLS manifest (playlist.m3u8) listing all chunks and bitrates. (7) CDN DISTRIBUTION: Pre-push popular titles to Open Connect appliances. Update video catalog database (video_id, manifest_url, status: "ready"). TIME CALCULATION: 720 chunks ÷ 100 workers = 7.2 chunks/worker. Encoding time: 3 min/chunk × 7.2 chunks = 21.6 minutes per worker (with 5 bitrates in parallel on multi-core). Add upload time (5 min), total: ~30 minutes. COST: 100 spot instances × 0.5 hours × $0.05/hour = $2.50 per video. Amortized over millions of views = negligible. OPTIMIZATIONS: (1) Cache encoding profiles (H.264 settings) for consistency. (2) Use GPU instances for faster encoding (H.265). (3) Adaptive bitrate: Encode only bitrates needed (skip 4K for old TV shows). KEY INSIGHT: Parallelization transforms hours-long sequential encoding into 30-minute parallel job. Spot instances make it cost-effective.',
    keyPoints: [
      'Split 2-hour video into 720 × 10-second chunks',
      'Distribute across 100 workers (7-8 chunks each)',
      'Each worker encodes chunk to 5 bitrates (parallel)',
      'Total time: ~30 minutes (vs 10+ hours sequential)',
      'Generate HLS manifest, push to CDN',
    ],
  },
];
