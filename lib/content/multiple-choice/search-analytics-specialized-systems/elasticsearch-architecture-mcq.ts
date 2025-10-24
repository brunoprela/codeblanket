import { Quiz } from '@/lib/types';

const elasticsearchArchitectureMCQ: Quiz = {
  id: 'elasticsearch-architecture-mcq',
  title: 'Elasticsearch Architecture - Multiple Choice Questions',
  questions: [
    {
      id: 'es-arch-mcq-1',
      type: 'multiple-choice',
      question:
        'You have an Elasticsearch index configured with 5 primary shards and 2 replicas. Your cluster has 8 data nodes. How many total shards will be distributed across the cluster, and what is the minimum number of nodes required to maintain green cluster status?',
      options: [
        '5 total shards; minimum 3 nodes required',
        '15 total shards (5 primary + 10 replica); minimum 3 nodes required',
        '10 total shards (5 primary + 5 replica); minimum 2 nodes required',
        '15 total shards (5 primary + 10 replica); minimum 2 nodes required',
      ],
      correctAnswer: 1,
      explanation:
        "With 5 primary shards and 2 replicas, you have 5 primary + (5 × 2) = 15 total shards. Each primary shard has 2 complete copies (replicas). For green status, every primary must have all its replicas active, AND replicas cannot be on the same node as their primary. With 2 replicas per primary, you need at least 3 nodes: one node can hold primaries, and the other two nodes must hold the replicas separately. If you only had 2 nodes, you could fit the primaries and first set of replicas, but the second set of replicas couldn't be allocated without violating the same-node constraint, resulting in yellow status.",
    },
    {
      id: 'es-arch-mcq-2',
      type: 'multiple-choice',
      question:
        'You index a document with ID "user-12345" into an index with 10 primary shards using the default routing. Elasticsearch uses the formula: shard = hash(_id) % number_of_primary_shards. A user later searches for this document. Which statement is correct about how Elasticsearch processes this query?',
      options: [
        'The query is sent only to the shard calculated by hash("user-12345") % 10, minimizing network traffic',
        'The query is sent to all 10 primary shards, and results are merged by the coordinating node',
        'The query is sent to the master node, which looks up the document location in its index',
        'The query is sent to all primary and replica shards (20 total) to ensure consistency',
      ],
      correctAnswer: 1,
      explanation:
        'When you perform a search query in Elasticsearch (not a GET by ID), the query is broadcast to all primary shards (or their replicas) because Elasticsearch doesn\'t know which shard contains the matching documents without searching all of them. While the document "user-12345" resides on a specific shard determined by routing, a search query with a filter or match condition must check all shards. If you use GET /index/_doc/user-12345 (retrieve by ID), then Elasticsearch CAN route directly to the specific shard using the routing formula. But general search queries require checking all shards. The coordinating node sends the query to one copy of each shard (primary or replica), receives results, merges them, and returns the top N.',
    },
    {
      id: 'es-arch-mcq-3',
      type: 'multiple-choice',
      question:
        'Your Elasticsearch cluster has 3 master-eligible nodes and is experiencing network instability. You notice the cluster sometimes splits into two separate clusters, causing data inconsistency. What is the cause, and what is the correct solution?',
      options: [
        'This is split-brain caused by network partition. Set discovery.zen.minimum_master_nodes to 2 (quorum: N/2 + 1)',
        'This is caused by too many master nodes. Reduce to 1 master-eligible node to prevent conflicts',
        'This is normal behavior during network instability. No configuration change needed',
        'Set number_of_replicas to 0 to prevent conflicts during network partition',
      ],
      correctAnswer: 0,
      explanation:
        'This is the classic "split-brain" problem in distributed systems. With 3 master-eligible nodes, if they lose communication, they might form two separate clusters (e.g., nodes A+B vs node C), each electing its own master and accepting writes. This causes data divergence. The solution is to configure minimum_master_nodes = (N/2 + 1) = (3/2 + 1) = 2. This ensures a master election requires 2 out of 3 nodes to agree, preventing split-brain: in a partition, only the partition with 2+ nodes can elect a master, while the minority partition (1 node) cannot form a functional cluster and will reject operations. Note: Modern Elasticsearch (7.0+) handles this automatically with built-in quorum-based voting, but understanding the principle is crucial. Having only 1 master creates a single point of failure, and replicas don\'t affect this issue.',
    },
    {
      id: 'es-arch-mcq-4',
      type: 'multiple-choice',
      question:
        'You need to implement autocomplete functionality that suggests products as users type. You have a "product_name" field that should support both full-text search and autocomplete. Which mapping configuration would be most appropriate?',
      options: [
        'Map product_name as a single "text" field with standard analyzer',
        'Map product_name as a "keyword" field to preserve exact values for autocomplete',
        'Map product_name as "text" with standard analyzer, and add a "product_name.suggest" sub-field with "completion" type',
        'Use "text" field with edge n-grams analyzer for both search and autocomplete',
      ],
      correctAnswer: 2,
      explanation:
        'The best approach uses multi-fields with a specialized "completion" type for autocomplete. The mapping would look like: "product_name": { "type": "text", "fields": { "suggest": { "type": "completion" } } }. This configuration allows: (1) Full-text search on the main "product_name" field using standard analysis, and (2) Fast, optimized autocomplete using the "product_name.suggest" field with the completion suggester. The completion type uses a specialized in-memory data structure (FST - Finite State Transducer) optimized for prefix matching, providing sub-millisecond autocomplete performance. While edge n-grams can work for autocomplete, they increase index size significantly and aren\'t as fast as the purpose-built completion suggester. A keyword field doesn\'t support partial matching needed for autocomplete. A single text field doesn\'t provide optimized autocomplete performance.',
    },
    {
      id: 'es-arch-mcq-5',
      type: 'multiple-choice',
      question:
        'Your e-commerce application experiences a traffic spike, and you notice Elasticsearch query latency increases from 50ms to 500ms. Looking at cluster stats, you see that all 5 data nodes are at 85% CPU, but only 40% memory usage. You have 1 replica configured. What would be the most effective immediate solution to reduce query latency?',
      options: [
        'Increase the number of primary shards to distribute query load',
        'Add more data nodes and increase replicas to 2, distributing query load across more nodes',
        'Increase the memory allocation to each node to cache more data',
        'Reduce the refresh_interval to decrease CPU overhead from segment creation',
      ],
      correctAnswer: 1,
      explanation:
        "The bottleneck is CPU (85% usage), not memory (40% usage). With high CPU utilization across all nodes during query-heavy load, you need more compute capacity to handle queries. Adding more data nodes and increasing replicas achieves this: with 2 replicas instead of 1, you'll have 3 copies of each shard (primary + 2 replicas) that can all serve read queries. Adding nodes (e.g., from 5 to 8 nodes) distributes these shards across more machines, and Elasticsearch's load balancing automatically distributes queries across all shard copies. This effectively triples your query-serving capacity. Increasing primary shards requires reindexing (not immediate) and doesn't help if you're already distributing well. Adding memory won't help when memory isn't the bottleneck. Reducing refresh_interval would slightly help write overhead but would hurt search freshness and isn't addressing the core issue of insufficient query capacity.",
    },
  ],
};

export default elasticsearchArchitectureMCQ;
