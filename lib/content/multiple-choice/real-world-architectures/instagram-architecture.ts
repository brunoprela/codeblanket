/**
 * Multiple choice questions for Instagram Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const instagramarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What database technology does Instagram primarily use for its social graph and feed data after migrating from PostgreSQL?',
    options: [
      'MySQL with custom sharding',
      "TAO (The Associations and Objects), Facebook's graph database",
      'MongoDB for document storage',
      'Redis with persistence enabled',
    ],
    correctAnswer: 1,
    explanation:
      "After Facebook's acquisition, Instagram migrated to TAO (The Associations and Objects), Facebook's distributed data store optimized for social graph queries. TAO provides a graph-oriented API with objects and associations while using MySQL and Memcached underneath. It offers 99%+ cache hit rates and is optimized for the read-heavy workload typical of social networks.",
  },
  {
    id: 'mc2',
    question:
      'Why does Instagram use fanout-on-read for celebrity accounts instead of fanout-on-write?',
    options: [
      'Celebrities post infrequently, making caching ineffective',
      'To avoid 100M+ writes per post when a celebrity has many followers',
      'Fanout-on-read provides better data consistency',
      'Celebrity content requires special moderation before fanout',
    ],
    correctAnswer: 1,
    explanation:
      'Instagram uses fanout-on-read for celebrities (typically >1M followers) because writing to millions or hundreds of millions of follower feeds would be infeasible—it would overload the system and take minutes. Instead, celebrity posts are fetched on-demand when followers request their feeds, merged with their pre-computed feed. This adds 20-50ms latency but avoids the massive write amplification.',
  },
  {
    id: 'mc3',
    question: 'Which storage solution does Instagram use for storing photos?',
    options: [
      'Custom distributed file system',
      'Amazon S3 with CloudFront CDN',
      'Cassandra with custom blob storage',
      'Ceph distributed storage',
    ],
    correctAnswer: 1,
    explanation:
      "Instagram stores photos in Amazon S3 (master and processed versions) and delivers them through CloudFront CDN with a 95%+ cache hit rate. Photos are processed into multiple sizes (thumbnail, medium, full) and compressed to reduce storage costs. The combination of S3's durability and CloudFront's global edge network ensures fast photo delivery worldwide.",
  },
  {
    id: 'mc4',
    question:
      'How does Instagram handle the generation of multiple photo sizes?',
    options: [
      'Photos are generated on-demand when first requested',
      'An async processing queue generates multiple sizes (thumbnail, medium, full) after upload',
      'Client devices upload pre-processed images in all required sizes',
      'Machine learning models automatically generate optimal sizes',
    ],
    correctAnswer: 1,
    explanation:
      "Instagram uses an async processing queue to generate multiple photo sizes after upload. When a user uploads a photo, it's stored in a temporary bucket and added to a processing queue. Workers then generate multiple sizes (236px thumbnail, 564px medium, full size), compress them (reducing size by 40-60%), and store all versions in S3. This async approach prevents upload delays while ensuring all sizes are available for different client needs.",
  },
  {
    id: 'mc5',
    question:
      'What optimization technique does Instagram use to improve perceived photo loading speed?',
    options: [
      'Lazy loading with dominant color placeholders and progressive JPEGs',
      'Pre-loading all photos on app startup',
      'Using WebP format exclusively for smaller file sizes',
      'Streaming photos in real-time as they are processed',
    ],
    correctAnswer: 0,
    explanation:
      'Instagram uses lazy loading with dominant color extraction/blurhash placeholders and progressive JPEGs. Progressive JPEGs load blurry first, then gradually sharpen, improving perceived speed. Dominant color placeholders show a solid color while the image loads. Combined with lazy loading (only loading images as users scroll), these techniques create a smooth, fast-feeling user experience even on slower connections.',
  },
];
