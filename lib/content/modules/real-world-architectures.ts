/**
 * Real-World System Architectures Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { netflixarchitectureSection } from '../sections/real-world-architectures/netflix-architecture';
import { instagramarchitectureSection } from '../sections/real-world-architectures/instagram-architecture';
import { uberarchitectureSection } from '../sections/real-world-architectures/uber-architecture';
import { twitterarchitectureSection } from '../sections/real-world-architectures/twitter-architecture';
import { youtubearchitectureSection } from '../sections/real-world-architectures/youtube-architecture';
import { dropboxarchitectureSection } from '../sections/real-world-architectures/dropbox-architecture';
import { spotifyarchitectureSection } from '../sections/real-world-architectures/spotify-architecture';
import { airbnbarchitectureSection } from '../sections/real-world-architectures/airbnb-architecture';
import { linkedinarchitectureSection } from '../sections/real-world-architectures/linkedin-architecture';
import { whatsapparchitectureSection } from '../sections/real-world-architectures/whatsapp-architecture';
import { pinterestarchitectureSection } from '../sections/real-world-architectures/pinterest-architecture';
import { slackarchitectureSection } from '../sections/real-world-architectures/slack-architecture';
import { zoomarchitectureSection } from '../sections/real-world-architectures/zoom-architecture';
import { doordasharchitectureSection } from '../sections/real-world-architectures/doordash-architecture';
import { stripearchitectureSection } from '../sections/real-world-architectures/stripe-architecture';

// Import quizzes
import { netflixarchitectureQuiz } from '../quizzes/real-world-architectures/netflix-architecture';
import { instagramarchitectureQuiz } from '../quizzes/real-world-architectures/instagram-architecture';
import { uberarchitectureQuiz } from '../quizzes/real-world-architectures/uber-architecture';
import { twitterarchitectureQuiz } from '../quizzes/real-world-architectures/twitter-architecture';
import { youtubearchitectureQuiz } from '../quizzes/real-world-architectures/youtube-architecture';
import { dropboxarchitectureQuiz } from '../quizzes/real-world-architectures/dropbox-architecture';
import { spotifyarchitectureQuiz } from '../quizzes/real-world-architectures/spotify-architecture';
import { airbnbarchitectureQuiz } from '../quizzes/real-world-architectures/airbnb-architecture';
import { linkedinarchitectureQuiz } from '../quizzes/real-world-architectures/linkedin-architecture';
import { whatsapparchitectureQuiz } from '../quizzes/real-world-architectures/whatsapp-architecture';
import { pinterestarchitectureQuiz } from '../quizzes/real-world-architectures/pinterest-architecture';
import { slackarchitectureQuiz } from '../quizzes/real-world-architectures/slack-architecture';
import { zoomarchitectureQuiz } from '../quizzes/real-world-architectures/zoom-architecture';
import { doordasharchitectureQuiz } from '../quizzes/real-world-architectures/doordash-architecture';
import { stripearchitectureQuiz } from '../quizzes/real-world-architectures/stripe-architecture';

// Import multiple choice
import { netflixarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/netflix-architecture';
import { instagramarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/instagram-architecture';
import { uberarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/uber-architecture';
import { twitterarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/twitter-architecture';
import { youtubearchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/youtube-architecture';
import { dropboxarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/dropbox-architecture';
import { spotifyarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/spotify-architecture';
import { airbnbarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/airbnb-architecture';
import { linkedinarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/linkedin-architecture';
import { whatsapparchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/whatsapp-architecture';
import { pinterestarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/pinterest-architecture';
import { slackarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/slack-architecture';
import { zoomarchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/zoom-architecture';
import { doordasharchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/doordash-architecture';
import { stripearchitectureMultipleChoice } from '../multiple-choice/real-world-architectures/stripe-architecture';

export const realWorldArchitecturesModule: Module = {
  id: 'real-world-architectures',
  title: 'Real-World System Architectures',
  description:
    'Deep dive into how major tech companies architect their systems at scale, including Netflix, Instagram, Uber, Twitter, YouTube, Dropbox, Spotify, Airbnb, LinkedIn, WhatsApp, Pinterest, Slack, Zoom, DoorDash, and Stripe',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🏛️',
  keyTakeaways: [
    'Netflix: 700+ microservices, custom tooling (Zuul, Eureka, Hystrix), chaos engineering, polyglot persistence',
    'Instagram: Hybrid fanout (write for regular, read for celebrities), TAO migration, aggressive caching',
    'Uber: H3 geospatial indexing, DISCO dispatch optimization, real-time location tracking, Schemaless MySQL',
    'Twitter: Snowflake IDs, hybrid timeline fanout, FlockDB for graph, Earlybird real-time search',
    'YouTube: Petabyte-scale storage (GCS), transcoding pipeline, ML recommendations, Content ID for copyright',
    'Dropbox: Block-level deduplication, Magic Pocket (custom storage), conflict resolution, AWS migration',
    'Spotify: Chunked HTTP streaming, ML recommendations (Discover Weekly), Cassandra for metadata, pre-fetching',
    'Airbnb: Elasticsearch search with ML ranking, distributed locking (double booking prevention), dynamic pricing',
    'LinkedIn: Espresso/Voldemort/Venice (custom stores), Kafka (LinkedIn-created), PYMK recommendations',
    'WhatsApp: Erlang (massive concurrency), stateless servers, Signal Protocol (E2EE), minimalist architecture',
    'Pinterest: CNN embeddings for visual search, Faiss for similarity, HBase for graph, Smart Feed ranking',
    'Slack: Vitess-sharded MySQL, WebSocket + Redis pub/sub, Elasticsearch search, real-time messaging',
    'Zoom: MMR (router) architecture, simulcast adaptive bitrate, UDP for low latency, global datacenters',
    'DoorDash: Batch dispatch optimization, H3 location tracking, ETA prediction ML, dynamic surge pricing',
    'Stripe: Idempotency keys, strong consistency (PostgreSQL), PCI compliance, multi-currency support',
  ],
  learningObjectives: [
    'Understand trade-offs in real-world architectures: consistency vs availability, latency vs throughput',
    'Learn custom infrastructure patterns: Netflix OSS, LinkedIn Kafka, Dropbox Magic Pocket, Twitter Snowflake',
    'Master feed generation strategies: Hybrid fanout, ML ranking, real-time updates, caching',
    'Apply geospatial algorithms: H3 indexing (Uber/DoorDash), geohashing, quadtrees',
    'Design real-time systems: WebSocket connections, location tracking, presence, live updates',
    'Implement search at scale: Elasticsearch, ML ranking, faceted search, autocomplete',
    'Build recommendation systems: Collaborative filtering, content-based, deep learning, A/B testing',
    'Handle payments and transactions: Idempotency, escrow, multi-currency, PCI compliance',
    'Scale databases: Sharding strategies, read replicas, caching layers, polyglot persistence',
    'Apply ML in production: Ranking models, ETA prediction, fraud detection, personalization',
    'Design for reliability: Circuit breakers, retry logic, chaos engineering, multi-region',
    'Optimize video/audio streaming: Adaptive bitrate, transcoding, CDN, protocols (DASH/HLS)',
    'Build three-sided marketplaces: Dispatch optimization, dynamic pricing, supply-demand balancing',
    'Ensure security at scale: E2EE, tokenization, verification, content moderation',
    'Learn from production incidents: Trade-offs made, lessons learned, evolution over time',
  ],
  sections: [
    {
      ...netflixarchitectureSection,
      quiz: netflixarchitectureQuiz,
      multipleChoice: netflixarchitectureMultipleChoice,
    },
    {
      ...instagramarchitectureSection,
      quiz: instagramarchitectureQuiz,
      multipleChoice: instagramarchitectureMultipleChoice,
    },
    {
      ...uberarchitectureSection,
      quiz: uberarchitectureQuiz,
      multipleChoice: uberarchitectureMultipleChoice,
    },
    {
      ...twitterarchitectureSection,
      quiz: twitterarchitectureQuiz,
      multipleChoice: twitterarchitectureMultipleChoice,
    },
    {
      ...youtubearchitectureSection,
      quiz: youtubearchitectureQuiz,
      multipleChoice: youtubearchitectureMultipleChoice,
    },
    {
      ...dropboxarchitectureSection,
      quiz: dropboxarchitectureQuiz,
      multipleChoice: dropboxarchitectureMultipleChoice,
    },
    {
      ...spotifyarchitectureSection,
      quiz: spotifyarchitectureQuiz,
      multipleChoice: spotifyarchitectureMultipleChoice,
    },
    {
      ...airbnbarchitectureSection,
      quiz: airbnbarchitectureQuiz,
      multipleChoice: airbnbarchitectureMultipleChoice,
    },
    {
      ...linkedinarchitectureSection,
      quiz: linkedinarchitectureQuiz,
      multipleChoice: linkedinarchitectureMultipleChoice,
    },
    {
      ...whatsapparchitectureSection,
      quiz: whatsapparchitectureQuiz,
      multipleChoice: whatsapparchitectureMultipleChoice,
    },
    {
      ...pinterestarchitectureSection,
      quiz: pinterestarchitectureQuiz,
      multipleChoice: pinterestarchitectureMultipleChoice,
    },
    {
      ...slackarchitectureSection,
      quiz: slackarchitectureQuiz,
      multipleChoice: slackarchitectureMultipleChoice,
    },
    {
      ...zoomarchitectureSection,
      quiz: zoomarchitectureQuiz,
      multipleChoice: zoomarchitectureMultipleChoice,
    },
    {
      ...doordasharchitectureSection,
      quiz: doordasharchitectureQuiz,
      multipleChoice: doordasharchitectureMultipleChoice,
    },
    {
      ...stripearchitectureSection,
      quiz: stripearchitectureQuiz,
      multipleChoice: stripearchitectureMultipleChoice,
    },
  ],
};
