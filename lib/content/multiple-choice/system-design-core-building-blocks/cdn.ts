/**
 * Multiple choice questions for CDN (Content Delivery Network) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const cdnMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary benefit of using a CDN?',
    options: [
      'Reduced database load',
      'Reduced latency by serving content from geographically closer servers',
      'Increased security through encryption',
      'Automatic content compression',
    ],
    correctAnswer: 1,
    explanation:
      "CDN primary benefit: Reduced latency by caching content on edge servers worldwide. Users fetch from nearest edge (10-50ms) vs origin (200-300ms). Secondary benefits: Reduced origin bandwidth (80-90%), DDoS protection, high availability. Not primarily for database (that's caching), security (CDN can help but not primary), or compression (origin responsibility).",
  },
  {
    id: 'mc2',
    question:
      'Your CDN cache hit rate is 60% (target is 90%). What could be the problem?',
    options: [
      'TTL is too short (content expires too quickly)',
      'Origin server is too slow',
      'Users are geographically distributed',
      'CDN is overloaded',
    ],
    correctAnswer: 0,
    explanation:
      "Low cache hit rate (60%) typically caused by: TTL too short (content expires too quickly, edge refetches too often). Solution: Increase TTL for static content. Or: Frequent cache purges. Or: Many unique URLs (query strings not cached). Origin speed doesn't affect hit rate (affects cache miss latency). Geographic distribution is good (CDN handles it). CDN overload would cause slow responses, not low hit rate.",
  },
  {
    id: 'mc3',
    question: 'What is the recommended way to update cached content on a CDN?',
    options: [
      'Manually purge the cache every time you update content',
      'Set TTL to 1 second so content updates frequently',
      'Use versioned URLs (cache busting) like /logo.png?v=2',
      'Restart the CDN servers',
    ],
    correctAnswer: 2,
    explanation:
      'Best practice: Versioned URLs (/logo.png?v=2 or /logo-abc123.png). Benefits: Instant updates (no wait for purge), free (no purge costs), automatic (build tools), immutable (long TTL, high hit rate). Purging works but slow (1-5 min), costs money, manual effort. Short TTL defeats CDN purpose (low hit rate). Restarting CDN not possible (managed service).',
  },
  {
    id: 'mc4',
    question: 'When should you NOT use a CDN?',
    options: [
      'Your website has global users across multiple continents',
      'You serve large video files',
      'You have very low traffic (<100 visitors/day) and all users are in the same city as your server',
      'You want to reduce origin server bandwidth costs',
    ],
    correctAnswer: 2,
    explanation:
      "Don't use CDN when: Very low traffic (<100 visitors/day), all users near origin server (no latency benefit), CDN cost > benefit. Use CDN when: Global users (latency reduction), large files (bandwidth savings), high traffic (origin offload), DDoS protection needed. Option 2 is low traffic + same city = CDN overkill (adds latency for DNS lookups).",
  },
  {
    id: 'mc5',
    question:
      'What is the typical cache hit rate target for a well-configured CDN?',
    options: ['50-60%', '70-80%', '90-95%', '99-100%'],
    correctAnswer: 2,
    explanation:
      '90-95% cache hit rate is typical target. This means 90-95% of requests served from edge cache (fast), 5-10% from origin (cache misses). 99-100% unrealistic (uncacheable content, new content, low-traffic pages). 50-60% indicates problem (TTL too short, too many purges, poor cache config). Monitor hit rate and optimize if <85%.',
  },
];
