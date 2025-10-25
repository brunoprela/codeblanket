/**
 * Multiple choice questions for YouTube Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const youtubearchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'How many formats and resolutions does YouTube typically generate for each uploaded video?',
    options: [
      'Only 2: SD and HD versions',
      '4 formats: 360p, 720p, 1080p, 4K',
      'Multiple formats across 144p to 4K and multiple codecs (H.264, VP9, AV1)',
      'Adaptive formats generated on-demand when requested',
    ],
    correctAnswer: 2,
    explanation:
      'YouTube generates multiple formats for each video: resolutions from 144p to 4K (144p, 240p, 360p, 480p, 720p, 1080p, 4K) and multiple codecs (H.264, VP9, AV1). This enables adaptive bitrate streaming where clients select the best quality based on bandwidth. Initially, only 360p and 720p are transcoded for quick availability, with full transcoding happening overnight for less popular channels.',
  },
  {
    id: 'mc2',
    question: "What is DASH in the context of YouTube\'s video delivery?",
    options: [
      'Dynamic Asynchronous Streaming Handler for multiplexing',
      'Data Aggregation Service for analytics',
      'Dynamic Adaptive Streaming over HTTP for quality switching',
      'Distributed Archival Storage Hierarchy',
    ],
    correctAnswer: 2,
    explanation:
      "DASH (Dynamic Adaptive Streaming over HTTP) is YouTube\'s protocol for video delivery. Videos are split into 2-10 second chunks, with each chunk available in multiple qualities. The player dynamically switches quality based on current buffer level and measured bandwidth. If bandwidth drops, the player switches to lower quality to avoid rebuffering. This ensures smooth playback across varying network conditions.",
  },
  {
    id: 'mc3',
    question:
      'What percentage of YouTube watch time comes from recommendations?',
    options: [
      'Approximately 30%',
      'Approximately 50%',
      'Approximately 70%',
      'Approximately 90%',
    ],
    correctAnswer: 2,
    explanation:
      "Approximately 70% of YouTube watch time comes from recommendations. YouTube\'s sophisticated two-stage recommendation system (candidate generation from millions of videos to ~100, then deep ranking to ordered list) is central to user engagement. The ML models consider watch history, search history, demographics, and video metadata, trained on petabytes of historical data with TensorFlow and TPUs.",
  },
  {
    id: 'mc4',
    question:
      'Which infrastructure does YouTube use for model training and inference at scale?',
    options: [
      'AWS SageMaker for training and Lambda for inference',
      'TensorFlow for training and TPUs for inference',
      'PyTorch distributed training with GPU clusters',
      'Spark MLlib with CPU-based inference',
    ],
    correctAnswer: 1,
    explanation:
      "YouTube uses TensorFlow for training recommendation models and TPUs (Tensor Processing Units) for inference. TPUs are Google\'s custom AI accelerators optimized for TensorFlow operations. Models are trained offline on Google Dataflow processing petabytes of data, then deployed to serving infrastructure with Bigtable caching user features. This enables sub-100ms recommendation queries for 2 billion users.",
  },
  {
    id: 'mc5',
    question:
      "What is YouTube\'s typical cache hit rate for popular videos in its CDN?",
    options: [
      'Approximately 60-70%',
      'Approximately 75-85%',
      'Approximately 95%+',
      'Approximately 99%+',
    ],
    correctAnswer: 2,
    explanation:
      "YouTube achieves 95%+ cache hit rate for popular videos in its global CDN (Google Cloud CDN + Edge nodes). When users request videos, they're routed to the nearest edge node via Anycast. Popular videos are cached at edges, avoiding origin fetches. Combined with predictive prefetching (preload next chunks) and partial content caching (first 30 seconds always cached), this enables smooth playback for 1 billion hours watched daily.",
  },
];
