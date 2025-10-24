/**
 * System Design Case Studies Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { tinyurlSection } from '../sections/system-design-case-studies/tinyurl';
import { pastebinSection } from '../sections/system-design-case-studies/pastebin';
import { twitterSection } from '../sections/system-design-case-studies/twitter';
import { instagramSection } from '../sections/system-design-case-studies/instagram';
import { facebookMessengerSection } from '../sections/system-design-case-studies/facebook-messenger';
import { netflixSection } from '../sections/system-design-case-studies/netflix';
import { youtubeSection } from '../sections/system-design-case-studies/youtube';
import { uberSection } from '../sections/system-design-case-studies/uber';
import { whatsappSection } from '../sections/system-design-case-studies/whatsapp';
import { dropboxSection } from '../sections/system-design-case-studies/dropbox';
import { yelpSection } from '../sections/system-design-case-studies/yelp';
import { ticketmasterSection } from '../sections/system-design-case-studies/ticketmaster';
import { webCrawlerSection } from '../sections/system-design-case-studies/web-crawler';
import { rateLimiterSection } from '../sections/system-design-case-studies/api-rate-limiter';
import { typeaheadSection } from '../sections/system-design-case-studies/typeahead-suggestion';
import { newsFeedSection } from '../sections/system-design-case-studies/news-feed';

// Import quizzes
import { tinyurlQuiz } from '../quizzes/system-design-case-studies/tinyurl';
import { pastebinQuiz } from '../quizzes/system-design-case-studies/pastebin';
import { twitterQuiz } from '../quizzes/system-design-case-studies/twitter';
import { instagramQuiz } from '../quizzes/system-design-case-studies/instagram';
import { facebookMessengerQuiz } from '../quizzes/system-design-case-studies/facebook-messenger';
import { netflixQuiz } from '../quizzes/system-design-case-studies/netflix';
import { youtubeQuiz } from '../quizzes/system-design-case-studies/youtube';
import { uberQuiz } from '../quizzes/system-design-case-studies/uber';
import { whatsappQuiz } from '../quizzes/system-design-case-studies/whatsapp';
import { dropboxQuiz } from '../quizzes/system-design-case-studies/dropbox';
import { yelpQuiz } from '../quizzes/system-design-case-studies/yelp';
import { ticketmasterQuiz } from '../quizzes/system-design-case-studies/ticketmaster';
import { webCrawlerQuiz } from '../quizzes/system-design-case-studies/web-crawler';
import { rateLimiterQuiz } from '../quizzes/system-design-case-studies/api-rate-limiter';
import { typeaheadQuiz } from '../quizzes/system-design-case-studies/typeahead-suggestion';
import { newsFeedQuiz } from '../quizzes/system-design-case-studies/news-feed';

// Import multiple choice
import { tinyurlMultipleChoice } from '../multiple-choice/system-design-case-studies/tinyurl';
import { pastebinMultipleChoice } from '../multiple-choice/system-design-case-studies/pastebin';
import { twitterMultipleChoice } from '../multiple-choice/system-design-case-studies/twitter';
import { instagramMultipleChoice } from '../multiple-choice/system-design-case-studies/instagram';
import { facebookMessengerMultipleChoice } from '../multiple-choice/system-design-case-studies/facebook-messenger';
import { netflixMultipleChoice } from '../multiple-choice/system-design-case-studies/netflix';
import { youtubeMultipleChoice } from '../multiple-choice/system-design-case-studies/youtube';
import { uberMultipleChoice } from '../multiple-choice/system-design-case-studies/uber';
import { whatsappMultipleChoice } from '../multiple-choice/system-design-case-studies/whatsapp';
import { dropboxMultipleChoice } from '../multiple-choice/system-design-case-studies/dropbox';
import { yelpMultipleChoice } from '../multiple-choice/system-design-case-studies/yelp';
import { ticketmasterMultipleChoice } from '../multiple-choice/system-design-case-studies/ticketmaster';
import { webCrawlerMultipleChoice } from '../multiple-choice/system-design-case-studies/web-crawler';
import { rateLimiterMultipleChoice } from '../multiple-choice/system-design-case-studies/api-rate-limiter';
import { typeaheadMultipleChoice } from '../multiple-choice/system-design-case-studies/typeahead-suggestion';
import { newsFeedMultipleChoice } from '../multiple-choice/system-design-case-studies/news-feed';

export const systemDesignCaseStudiesModule: Module = {
  id: 'system-design-case-studies',
  title: 'System Design Case Studies',
  description:
    'Design real-world systems from scratch using all learned concepts: URL shorteners, social networks, streaming platforms, messaging apps, and more',
  category: 'System Design',
  difficulty: 'Advanced',
  estimatedTime: '14-16 hours',
  prerequisites: [
    'system-design-fundamentals',
    'system-design-advanced-algorithms',
  ],
  icon: '📱',
  keyTakeaways: [
    'System design interviews test your ability to architect complete, scalable systems under realistic constraints',
    'Always start with requirements gathering and capacity estimation before jumping to architecture',
    'Feed generation: fanout-on-write for small followers, fanout-on-read for celebrities, hybrid for production',
    'URL shortening requires collision handling, base62 encoding, and careful database sharding',
    'Real-time systems (Uber, Messenger) need WebSockets, geospatial indexes, and efficient matching algorithms',
    'Video platforms (Netflix, YouTube) rely heavily on CDNs, adaptive bitrate streaming, and distributed encoding',
    'File storage systems (Dropbox) use block-level deduplication and conflict resolution strategies',
    'High-concurrency systems (Ticketmaster) require distributed locking and optimistic concurrency control',
    'Search systems (Yelp, Typeahead) leverage inverted indexes, tries, and geospatial data structures',
    'Rate limiters use token bucket or sliding window algorithms with Redis for distributed coordination',
    'Every design involves trade-offs: consistency vs availability, latency vs throughput, cost vs performance',
    'Production systems combine multiple patterns: caching, sharding, replication, load balancing, queues',
    'Scalability considerations: read vs write heavy, peak traffic patterns, data growth projections',
    'Real companies solve similar problems differently based on their specific constraints and scale',
  ],
  learningObjectives: [
    'Apply the systematic problem-solving framework to real-world system design scenarios',
    'Gather functional and non-functional requirements through strategic questioning',
    'Perform back-of-the-envelope calculations for storage, bandwidth, and QPS',
    'Design scalable architectures using core building blocks (load balancers, caches, databases)',
    'Choose appropriate database technologies based on access patterns and consistency requirements',
    'Implement efficient data partitioning and sharding strategies for massive scale',
    'Design real-time communication systems with WebSockets and message queues',
    'Architect video streaming platforms with CDN integration and adaptive bitrate',
    'Build geospatial systems with appropriate indexing for proximity queries',
    'Handle high-concurrency scenarios with distributed locking and optimistic concurrency',
    'Design resilient systems with fault tolerance, replication, and failover strategies',
    'Optimize for cost, performance, and operational complexity in production environments',
    'Articulate trade-offs clearly and justify architectural decisions during interviews',
    'Deep-dive into specific components when prompted by interviewers',
    'Recognize patterns across different system designs and adapt solutions appropriately',
  ],
  sections: [
    {
      ...tinyurlSection,
      quiz: tinyurlQuiz,
      multipleChoice: tinyurlMultipleChoice,
    },
    {
      ...pastebinSection,
      quiz: pastebinQuiz,
      multipleChoice: pastebinMultipleChoice,
    },
    {
      ...twitterSection,
      quiz: twitterQuiz,
      multipleChoice: twitterMultipleChoice,
    },
    {
      ...instagramSection,
      quiz: instagramQuiz,
      multipleChoice: instagramMultipleChoice,
    },
    {
      ...facebookMessengerSection,
      quiz: facebookMessengerQuiz,
      multipleChoice: facebookMessengerMultipleChoice,
    },
    {
      ...netflixSection,
      quiz: netflixQuiz,
      multipleChoice: netflixMultipleChoice,
    },
    {
      ...youtubeSection,
      quiz: youtubeQuiz,
      multipleChoice: youtubeMultipleChoice,
    },
    {
      ...uberSection,
      quiz: uberQuiz,
      multipleChoice: uberMultipleChoice,
    },
    {
      ...whatsappSection,
      quiz: whatsappQuiz,
      multipleChoice: whatsappMultipleChoice,
    },
    {
      ...dropboxSection,
      quiz: dropboxQuiz,
      multipleChoice: dropboxMultipleChoice,
    },
    {
      ...yelpSection,
      quiz: yelpQuiz,
      multipleChoice: yelpMultipleChoice,
    },
    {
      ...ticketmasterSection,
      quiz: ticketmasterQuiz,
      multipleChoice: ticketmasterMultipleChoice,
    },
    {
      ...webCrawlerSection,
      quiz: webCrawlerQuiz,
      multipleChoice: webCrawlerMultipleChoice,
    },
    {
      ...rateLimiterSection,
      quiz: rateLimiterQuiz,
      multipleChoice: rateLimiterMultipleChoice,
    },
    {
      ...typeaheadSection,
      quiz: typeaheadQuiz,
      multipleChoice: typeaheadMultipleChoice,
    },
    {
      ...newsFeedSection,
      quiz: newsFeedQuiz,
      multipleChoice: newsFeedMultipleChoice,
    },
  ],
};
