/**
 * Core Building Blocks Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { loadbalancingSection } from '../sections/system-design-core-building-blocks/load-balancing';
import { cachingSection } from '../sections/system-design-core-building-blocks/caching';
import { datapartitioningshardingSection } from '../sections/system-design-core-building-blocks/data-partitioning-sharding';
import { databasereplicationSection } from '../sections/system-design-core-building-blocks/database-replication';
import { messagequeuesSection } from '../sections/system-design-core-building-blocks/message-queues';
import { cdnSection } from '../sections/system-design-core-building-blocks/cdn';
import { apigatewaySection } from '../sections/system-design-core-building-blocks/api-gateway';
import { proxiesSection } from '../sections/system-design-core-building-blocks/proxies';

// Import quizzes
import { loadbalancingQuiz } from '../quizzes/system-design-core-building-blocks/load-balancing';
import { cachingQuiz } from '../quizzes/system-design-core-building-blocks/caching';
import { datapartitioningshardingQuiz } from '../quizzes/system-design-core-building-blocks/data-partitioning-sharding';
import { databasereplicationQuiz } from '../quizzes/system-design-core-building-blocks/database-replication';
import { messagequeuesQuiz } from '../quizzes/system-design-core-building-blocks/message-queues';
import { cdnQuiz } from '../quizzes/system-design-core-building-blocks/cdn';
import { apigatewayQuiz } from '../quizzes/system-design-core-building-blocks/api-gateway';
import { proxiesQuiz } from '../quizzes/system-design-core-building-blocks/proxies';

// Import multiple choice
import { loadbalancingMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/load-balancing';
import { cachingMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/caching';
import { datapartitioningshardingMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/data-partitioning-sharding';
import { databasereplicationMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/database-replication';
import { messagequeuesMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/message-queues';
import { cdnMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/cdn';
import { apigatewayMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/api-gateway';
import { proxiesMultipleChoice } from '../multiple-choice/system-design-core-building-blocks/proxies';

export const systemDesignCoreBuildingBlocksModule: Module = {
  id: 'system-design-core-building-blocks',
  title: 'Core Building Blocks',
  description:
    'Master the fundamental components that power all distributed systems including load balancing, caching, sharding, and message queues',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🏗️',
  keyTakeaways: [
    'Load balancers distribute traffic across multiple servers for availability and horizontal scaling',
    'Algorithm choice matters: Round Robin for simple/homogeneous servers, Least Connections for long-lived connections, Weighted for heterogeneous capacity',
    'Layer 4 (fast, protocol-agnostic) vs Layer 7 (intelligent HTTP routing, SSL termination) - choose based on needs',
    'Health checks are critical: detect and remove unhealthy servers automatically',
    'Avoid sticky sessions: design stateless servers with shared session storage (Redis)',
    'Caching dramatically reduces database load: 90% cache hit rate = 10× less database QPS',
    'Cache-aside most common pattern: check cache first, query DB on miss, populate cache',
    'LRU most common eviction policy: keeps hot data, evicts least recently used',
    'Cache invalidation is hard: use TTL + explicit invalidation for consistency',
    'Prevent cache stampede: jittered TTL or probabilistic early refresh',
    'Sharding = horizontal partitioning: splits data across machines by rows',
    'Partition key critical: choose high-cardinality, uniformly distributed key',
    'Consistent hashing: industry standard, minimal rebalancing when adding shards',
    'Avoid cross-shard joins: denormalize or colocate related data',
    'Replication = copying data to multiple databases for availability and read scaling',
    'Async replication: fast but eventual consistency (most common in practice)',
    'Sync replication: slow but strong consistency (use for critical data)',
    'Failover: automatic preferred, 30-90 second downtime typical',
    'Message queues decouple services, enable async processing, and absorb traffic spikes',
    'At-least-once delivery most common (idempotent consumers required)',
    'Dead Letter Queue handles failed messages after retries',
    'Queue for task distribution, Topic for event broadcasting',
    'CDN caches content on edge servers worldwide for low latency (10-50ms vs 200-300ms)',
    'Pull CDN most common (lazy loading, automatic), Push CDN for predictable traffic',
    'Versioned URLs best practice for cache invalidation (instant, free, automated)',
    '90-95% cache hit rate typical for well-configured CDN',
    'API Gateway = single entry point for all client requests (centralized auth, rate limiting)',
    'Use API Gateway + Load Balancers together (gateway for routing/auth, LB for distribution)',
    'Forward Proxy serves clients (hides client from server, content filtering)',
    'Reverse Proxy serves servers (hides server from client, load balancing, SSL termination)',
    'SSL termination at reverse proxy: 30% backend CPU reduction, centralized certificates',
  ],
  learningObjectives: [
    "Understand what load balancing is and why it's critical for distributed systems",
    'Master different load balancing algorithms and when to use each',
    'Learn the difference between Layer 4 and Layer 7 load balancing',
    'Implement health checks to detect and handle server failures',
    'Design stateless systems without relying on sticky sessions',
    'Master caching strategies to dramatically reduce database load',
    'Understand cache reading patterns and eviction policies',
    'Understand data partitioning (sharding) for horizontal scalability',
    'Master sharding strategies: hash-based, consistent hashing, range-based',
    'Choose effective partition keys and handle cross-shard operations',
    'Understand database replication for availability and read scalability',
    'Differentiate between synchronous and asynchronous replication',
    'Master failover processes and preventing split-brain scenarios',
    'Understand message queues for asynchronous service communication',
    'Differentiate between Queues (point-to-point) and Topics (pub/sub)',
    'Implement idempotent consumers and Dead Letter Queues',
    'Understand CDN architecture and how it reduces latency globally',
    'Master Pull vs Push CDN strategies and when to use each',
    'Implement cache invalidation using versioned URLs',
    'Understand API Gateway role in microservices architecture',
    'Master API Gateway responsibilities: routing, auth, rate limiting',
    'Learn when to use API Gateway vs Load Balancer vs both together',
    'Understand Forward Proxy vs Reverse Proxy differences',
    'Implement SSL termination at reverse proxy for performance',
    'Design corporate content filtering using forward proxy',
  ],
  sections: [
    {
      ...loadbalancingSection,
      quiz: loadbalancingQuiz,
      multipleChoice: loadbalancingMultipleChoice,
    },
    {
      ...cachingSection,
      quiz: cachingQuiz,
      multipleChoice: cachingMultipleChoice,
    },
    {
      ...datapartitioningshardingSection,
      quiz: datapartitioningshardingQuiz,
      multipleChoice: datapartitioningshardingMultipleChoice,
    },
    {
      ...databasereplicationSection,
      quiz: databasereplicationQuiz,
      multipleChoice: databasereplicationMultipleChoice,
    },
    {
      ...messagequeuesSection,
      quiz: messagequeuesQuiz,
      multipleChoice: messagequeuesMultipleChoice,
    },
    {
      ...cdnSection,
      quiz: cdnQuiz,
      multipleChoice: cdnMultipleChoice,
    },
    {
      ...apigatewaySection,
      quiz: apigatewayQuiz,
      multipleChoice: apigatewayMultipleChoice,
    },
    {
      ...proxiesSection,
      quiz: proxiesQuiz,
      multipleChoice: proxiesMultipleChoice,
    },
  ],
};
