import { MultipleChoiceQuestion } from '@/lib/types';

export const graphDatabasesMCQ: MultipleChoiceQuestion[] = [
  {
    id: 'graph-mcq-1',
    question:
      'You need to find "friends of friends" in a social network with 100M users. In PostgreSQL, this requires a self-join on the friendships table (20B rows). In Neo4j, how does query performance scale with the depth of traversal (1 hop vs 2 hops vs 3 hops)?',
    options: [
      'Performance degrades exponentially with depth (like SQL JOINs)',
      'Performance is proportional to the size of the subgraph traversed, NOT the total graph size',
      'Performance is constant regardless of depth or graph size',
      'Neo4j is always slower than PostgreSQL for relationship queries',
    ],
    correctAnswer: 1,
    explanation:
      'Neo4j performance is proportional to subgraph size, not total graph size. For a user with 200 friends: 1-hop query touches 200 nodes, 2-hop touches ~200×200=40,000 nodes, 3-hop touches ~8M nodes. But this is independent of whether you have 1M or 100M total users—you only traverse the local subgraph. In PostgreSQL, each self-join scans the entire friendships table (20B rows), and performance degrades exponentially: 1 join → 2 joins → 3 joins. Neo4j uses index-free adjacency: each node stores pointers to neighbors, so traversal is O(subgraph size). This is why graph databases can find "friends of friends" in 50ms while PostgreSQL times out. The key insight: graph query complexity depends on local graph structure (how many friends, not how many total users).',
  },
  {
    id: 'graph-mcq-2',
    question:
      'In Neo4j, you model users and products with a PURCHASED relationship. Should product category be a node property or a separate Category node with relationships?',
    options: [
      'Property: (Product {name: "Laptop", category: "Electronics"})',
      'Separate node: (Product)-[:IN_CATEGORY]->(Category {name: "Electronics"})',
      'Both approaches are equivalent in performance',
      'Neo4j does not support node properties',
    ],
    correctAnswer: 1,
    explanation:
      "Separate Category nodes enable powerful graph queries. With (Product)-[:IN_CATEGORY]->(Category), you can query \"find products in same category as products I bought\" with simple traversal: (me)-[:PURCHASED]->(p)-[:IN_CATEGORY]->(c)<-[:IN_CATEGORY]-(similar_products). You can also traverse category hierarchies: (Electronics)<-[:SUBCATEGORY_OF]-(Laptops). As properties, you'd need WHERE clauses and couldn't leverage graph algorithms (PageRank on categories, community detection). Rule of thumb: if you query relationships between values (products in same category), make it a node; if it's simple metadata never queried relationally (product SKU), make it a property. This is fundamental to graph modeling—relationships are first-class, so model them explicitly as edges, not as properties requiring WHERE comparisons.",
  },
  {
    id: 'graph-mcq-3',
    question:
      'Your fraud detection query finds accounts connected to a suspicious account within 3 hops using: MATCH path = (suspicious)-[*1..3]-(connected). This returns 100,000 paths. What is the most critical optimization?',
    options: [
      'Add LIMIT 100 to return only top results',
      'Specify relationship types: -[:USES_DEVICE|TRANSFERRED_TO*1..3]- to constrain traversal',
      'Use shortest path algorithm instead',
      'Increase Neo4j heap size to handle more results',
    ],
    correctAnswer: 1,
    explanation:
      "Constraining relationship types is critical because -[*1..3]- traverses ALL relationship types, creating a combinatorial explosion. If each account has 10 relationships of ANY type (purchases, logins, transfers, devices), 3 hops = 10^3 = 1,000 paths per account. By specifying -[:USES_DEVICE|TRANSFERRED_TO*1..3]-, you only traverse fraud-relevant relationships, reducing from 1,000 to ~10-20 paths. LIMIT (option A) reduces returned results but doesn't prevent the expensive traversal computation. Shortest path (option C) finds only one path, missing other connections. Heap size (option D) doesn't address the algorithmic problem. Unbounded relationship traversal is the most common Neo4j performance killer—always constrain by relationship type and depth. Real fraud queries often specify exact patterns: -[:USES_DEVICE]->()-[:USES_DEVICE]- (shared device).",
  },
  {
    id: 'graph-mcq-4',
    question:
      'You have a recommendation query that runs fast on 1M products but becomes slow on 10M products. The query finds co-purchased products. What is the likely cause?',
    options: [
      "Graph databases don't scale to 10M nodes",
      'Missing index on Product.id—Neo4j scans all products to find the target',
      'Query returns too many results without LIMIT',
      'Need to shard across multiple Neo4j instances',
    ],
    correctAnswer: 1,
    explanation:
      "Missing indexes cause full graph scans. MATCH (p:Product {id: $productId}) without an index on Product.id scans all 10M products (slow!). With index: CREATE INDEX product_id FOR (p:Product) ON (p.id), Neo4j does O(log n) lookup (milliseconds). The subsequent traversal (finding co-purchased products) is fast because it only touches the subgraph (users who bought this product). Graph databases absolutely scale to 10M+ nodes (option A is wrong)—Facebook\'s social graph has billions. LIMIT (option C) helps but doesn't fix the initial lookup. Sharding (option D) is complex and unnecessary if the real issue is missing indexes. This is the #1 Neo4j performance issue: developers forget that graph traversal is fast but initial node lookup needs indexes just like any database.",
  },
  {
    id: 'graph-mcq-5',
    question:
      'When would you choose PostgreSQL over Neo4j for a social network application?',
    options: [
      'When relationship queries (friends of friends) are the primary workload',
      'Never—graph databases are always better for social networks',
      'When the application primarily does CRUD operations (create user, update profile) with simple 1-hop queries (list my friends)',
      'When the social network has more than 1 million users',
    ],
    correctAnswer: 2,
    explanation:
      'PostgreSQL is better for simple CRUD with 1-hop queries. If your app mostly does "insert user", "update user profile", "select my friends" (single JOIN), PostgreSQL is simpler, more familiar, and performs adequately. Neo4j excels at complex traversals (2+ hops): friends of friends, shortest path, recommendation engines, fraud detection. But for basic operations, Neo4j adds complexity without benefit. Consider: Instagram-style app where you mostly view your own feed (1-hop: my followings\' posts) → PostgreSQL fine. LinkedIn-style where you explore 2nd/3rd connections, get recommendations → Neo4j shines. The key is complexity of relationship queries. Don\'t over-engineer with graph database when relationships are simple. Graph databases scale to billions of nodes (option D is wrong), and there ARE cases when PostgreSQL is better (option B is wrong). Use the right tool for the workload.',
  },
];
