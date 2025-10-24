/**
 * Multiple choice questions for Slack Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const slackarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What protocol does Slack use for real-time messaging between clients and servers?',
        options: [
            'Long polling with HTTP',
            'Server-Sent Events (SSE)',
            'WebSocket with 30-second heartbeats',
            'gRPC bidirectional streaming',
        ],
        correctAnswer: 2,
        explanation: 'Slack uses WebSocket connections for real-time messaging with 30-second heartbeats to keep connections alive. Clients connect to Gateway servers which maintain persistent connections. When a user sends a message, it goes through the Gateway to Kafka (partitioned by channel_id), then back to Gateway servers which propagate to all connected clients in that channel. This provides low-latency message delivery (sub-100ms) at scale.',
    },
    {
        id: 'mc2',
        question: 'How does Slack ensure message ordering within a channel?',
        options: [
            'Global timestamp synchronization across all servers',
            'Kafka partitioning by channel_id ensures FIFO ordering',
            'Application-level vector clocks',
            'Database sequence numbers with locks',
        ],
        correctAnswer: 1,
        explanation: 'Slack ensures message ordering by partitioning Kafka topics by channel_id. All messages for a channel go to the same Kafka partition, which guarantees FIFO (first-in-first-out) ordering. Each message receives a timestamp and sequence number. Kafka consumers process messages in order, write to MySQL, and propagate to connected clients. This distributed approach avoids centralized coordination while maintaining ordering guarantees.',
    },
    {
        id: 'mc3',
        question: 'What is Slack\'s database sharding strategy?',
        options: [
            'Shard by user_id for user-centric queries',
            'Shard by workspace_id (tenant-based sharding)',
            'Shard by channel_id for message distribution',
            'Shard by geographic region for latency',
        ],
        correctAnswer: 1,
        explanation: 'Slack shards by workspace_id (tenant-based sharding). Each workspace is an independent unit, and most queries are within a workspace (no cross-shard joins needed). Large workspaces (>10,000 users) get dedicated shards, while small workspaces share shards. Vitess manages the MySQL sharding topology, and a routing service maps workspace_id to shard. This enables Slack to scale to millions of workspaces across thousands of MySQL shards.',
    },
    {
        id: 'mc4',
        question: 'How quickly are messages indexed for search after being sent?',
        options: [
            'Instantly (real-time indexing)',
            'Within 1-2 seconds via Kafka streaming',
            'Within 5 minutes via batch jobs',
            'Hourly via scheduled jobs',
        ],
        correctAnswer: 1,
        explanation: 'Slack messages are indexed in Elasticsearch within 1-2 seconds via Kafka streaming. When a message is sent, it\'s stored in MySQL and published to Kafka. A search indexer service consumes from Kafka and indexes the message in Elasticsearch (one index per workspace). This near-real-time indexing enables users to search for just-sent messages almost immediately. Combined with BM25 ranking, recency boost, and permission filtering, this provides <200ms p95 search latency.',
    },
    {
        id: 'mc5',
        question: 'Which MySQL sharding framework does Slack use?',
        options: [
            'ProxySQL for query routing',
            'Vitess for shard management',
            'MySQL Router with InnoDB Cluster',
            'Custom sharding middleware',
        ],
        correctAnswer: 1,
        explanation: 'Slack uses Vitess, an open-source MySQL sharding framework originally developed by YouTube. Vitess manages shard topology, query routing, schema migrations, and rebalancing. It provides a single logical database interface while underneath managing thousands of MySQL shards. Vitess handles splitting shards as workspaces grow, automated failover, and connection pooling. This allows Slack to scale MySQL horizontally while maintaining a simple application API.',
    },
];

