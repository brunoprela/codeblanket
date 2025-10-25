/**
 * Multiple choice questions for Spotify Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const spotifyarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which three main approaches does Spotify use for music recommendations?',
    options: [
      'User surveys, listening history, and trending charts',
      'Collaborative filtering, NLP analysis, and audio analysis',
      'Demographic targeting, playlist curation, and social sharing',
      'Artist recommendations, genre matching, and popularity ranking',
    ],
    correctAnswer: 1,
    explanation:
      'Spotify uses three complementary approaches: (1) Collaborative filtering—users with similar listening history enjoy similar songs, using matrix factorization. (2) Natural Language Processing—analyzing blog posts, articles, descriptions to understand genre, mood, themes. (3) Audio analysis—extracting features from raw audio (tempo, key, loudness, danceability) using ML models. Combining these creates robust recommendations for Discover Weekly and other personalized playlists.',
  },
  {
    id: 'mc2',
    question: "What is Spotify's typical cache size limit for offline songs?",
    options: [
      '1 GB (approximately 200 songs)',
      '5 GB (configurable by user)',
      '10 GB (fixed limit)',
      'Unlimited (storage-dependent)',
    ],
    correctAnswer: 1,
    explanation:
      'Spotify allows users to cache up to 5 GB of recently played songs (configurable). This enables instant playback from cache and supports offline mode where users can download songs. The cache stores songs at the highest quality based on user settings. Combined with P2P sharing on local networks and predictive prefetching of playlist songs, this significantly reduces bandwidth usage.',
  },
  {
    id: 'mc3',
    question:
      'How does Spotify handle adaptive streaming based on network conditions?',
    options: [
      'Fixed quality based on subscription tier',
      'Multiple bitrates (96, 160, 320 kbps) selected based on network type',
      'Continuous quality adjustment every second',
      'User manually selects quality before playback',
    ],
    correctAnswer: 1,
    explanation:
      'Spotify encodes songs in multiple bitrates: 96 kbps (low quality), 160 kbps (normal), and 320 kbps (high quality). The client selects the appropriate bitrate based on network conditions—typically 160 kbps on cellular and 320 kbps on WiFi. Songs are split into 15-second chunks, and the client prefetches the next 30-60 seconds. This balances audio quality with bandwidth usage and ensures smooth playback.',
  },
  {
    id: 'mc4',
    question:
      'Which data processing technology does Spotify use for batch processing user data and generating weekly playlists?',
    options: [
      'Hadoop MapReduce',
      'Apache Spark',
      'AWS Lambda',
      'Google Dataflow',
    ],
    correctAnswer: 1,
    explanation:
      "Spotify uses Apache Spark for batch processing user data and generating weekly playlists like Discover Weekly. Spark jobs run on Sunday nights to process 200M+ users, analyzing listening history, applying collaborative filtering, and generating personalized recommendations. The results are stored in Cassandra for quick retrieval. Spark's in-memory processing and rich API make it ideal for large-scale ML and data processing workloads.",
  },
  {
    id: 'mc5',
    question:
      'What percentage of Discover Weekly songs are saved by users, indicating successful recommendations?',
    options: [
      'Approximately 10%',
      'Approximately 20%',
      'Approximately 40%',
      'Approximately 60%',
    ],
    correctAnswer: 2,
    explanation:
      "Approximately 40% of Discover Weekly songs are saved by users, indicating high-quality personalized recommendations. This metric demonstrates the effectiveness of Spotify's hybrid recommendation approach combining collaborative filtering, NLP, and audio analysis. The weekly batch generation process on Spark ensures every user receives a fresh, personalized playlist that balances familiarity (similar to liked songs) with discovery (new artists/genres).",
  },
];
