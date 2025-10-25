/**
 * Quiz questions for YouTube Architecture section
 */

export const youtubearchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain YouTube\'s video upload and transcoding pipeline. How does it handle 500+ hours of video uploaded per minute?",
    sampleAnswer:
      'Upload pipeline: (1) User uploads video to nearest Edge server (regional upload nodes for faster transfer). (2) Video stored in temporary bucket, added to transcoding queue. (3) Transcoding cluster picks video, generates multiple formats (144p, 240p, 360p, 480p, 720p, 1080p, 4K) and codecs (H.264, VP9, AV1). (4) Store all versions in Google Cloud Storage/Colossus. (5) Update metadata database (video_id, formats available, duration, thumbnail URLs). (6) Publish to CDN. (7) Video live on YouTube. Scale: 500 hours/minute = 30,000 hours/hour. Distributed transcoding cluster with 100,000s of machines globally. Prioritization: Popular channels get faster transcoding, initially transcode only 360p/720p (on-demand for others), full transcode overnight. Result: Videos live in <10 minutes for most users.',
    keyPoints: [
      'Edge upload nodes for fast regional transfers',
      'Multiple formats (144p-4K) and codecs (H.264, VP9, AV1)',
      'Distributed transcoding: 100,000s of machines, queue-based processing',
      'Prioritization: Popular channels faster, incremental transcoding (360p/720p first)',
    ],
  },
  {
    id: 'q2',
    question:
      "How does YouTube\'s recommendation system work at scale? What techniques enable personalized video suggestions for 2 billion users?",
    sampleAnswer:
      'YouTube recommendation is two-stage: (1) Candidate generation: From millions of videos, select ~100 candidates. Use collaborative filtering (users who watched this also watched...), content-based filtering (similar topics/tags), trending videos. ML model trained on user history. (2) Ranking: Score and rank 100 candidates. ML model considers: click-through rate prediction, watch time prediction, user satisfaction signals. Features: user demographics, watch history, search history, video metadata, time of day. Model: Deep neural networks trained on petabytes of historical data. Infrastructure: TensorFlow for training, TPUs for inference. Offline: Train models daily on Google Dataflow. Online: Models deployed to serving infrastructure (Bigtable caching). Query: User visits homepage → fetch user features from Bigtable → run inference → return ranked videos. Latency: <100ms. Result: 70% of watch time from recommendations.',
    keyPoints: [
      'Two-stage: Candidate generation (millions→100) + Ranking (100→ordered list)',
      'ML models: collaborative filtering, content-based, deep neural networks',
      'Features: watch history, search, demographics, time of day, video metadata',
      'Infrastructure: TensorFlow + TPUs, Bigtable caching, <100ms latency',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe YouTube\'s approach to video delivery through CDNs. How does it achieve smooth playback for 1 billion hours watched daily?",
    sampleAnswer:
      "YouTube uses Google's global CDN (Cloud CDN + Edge nodes). Video delivery: (1) User requests video. (2) Player selects appropriate quality based on bandwidth (adaptive bitrate streaming). (3) Request routed to nearest Edge node via Anycast routing. (4) Edge checks cache. If hit (95%+ for popular videos), stream from cache. If miss, fetch from origin (Google Cloud Storage), cache at Edge, stream to user. (5) DASH (Dynamic Adaptive Streaming over HTTP) protocol: video split into 2-10 second chunks, each chunk available in multiple qualities. Player dynamically switches quality based on buffer level and bandwidth. Optimizations: (1) Predictive prefetching (preload next chunks). (2) Partial content caching (first 30 seconds always cached). (3) P2P distribution (ISP partnerships). (4) Different codecs for different devices (AV1 for modern browsers, H.264 for old devices). Result: <1% rebuffering globally.",
    keyPoints: [
      'Global CDN with 95%+ cache hit rate for popular videos',
      'Adaptive bitrate streaming (DASH): 2-10 second chunks, multiple qualities',
      'Dynamic quality switching based on bandwidth and buffer',
      'Optimizations: predictive prefetching, partial caching, P2P, codec selection',
    ],
  },
];
