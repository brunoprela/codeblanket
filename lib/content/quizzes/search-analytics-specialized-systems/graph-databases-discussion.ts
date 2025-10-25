import { QuizQuestion } from '@/lib/types';

export const graphDatabasesDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'graph-discussion-1',
    question:
      'You\'re building a social network with 100M users. Each user has avg 200 friends. You need to show "friend recommendations" (friends of friends who aren\'t already friends). In PostgreSQL, this requires complex recursive CTEs that time out. Design a Neo4j solution with query examples and discuss performance characteristics.',
    sampleAnswer: `**Neo4j Schema:**

\`\`\`cypher
CREATE (u:User {id: 123, name: "Alice"})
CREATE INDEX user_id FOR (u:User) ON (u.id)
CREATE (u1)-[:FRIENDS_WITH]->(u2)
\`\`\`

**Query: Friend Recommendations**
\`\`\`cypher
MATCH (me:User {id: $userId})-[:FRIENDS_WITH]->(friend)-[:FRIENDS_WITH]->(fof)
WHERE NOT (me)-[:FRIENDS_WITH]->(fof) 
  AND me <> fof
RETURN fof.id, fof.name, COUNT(DISTINCT friend) AS mutual_friends
ORDER BY mutual_friends DESC
LIMIT 10
\`\`\`

**Performance:**
- Neo4j: O(friends × friends) = O(200 × 200) = 40,000 traversals (< 50ms)
- PostgreSQL: Multiple self-joins on 100M × 200 = 20B edges (timeout!)

**Why Neo4j is faster:**
1. **Index-free adjacency**: Each node stores pointers to connected nodes
2. **Native traversal**: Follow pointers, no JOIN operations
3. **Local operation**: Only touches friends and friends-of-friends

**Scalability:**
- 100M users × 200 friends avg = 20B relationships
- Neo4j handles this with ~2TB storage
- Query time remains constant (only touches ~40k nodes)

**Result:** Friend recommendations in <100ms vs PostgreSQL timeout.`,
    keyPoints: [
      'Graph databases use index-free adjacency: nodes store pointers to neighbors',
      'Traversal is O(friends × friends) regardless of total graph size',
      'No JOINs required—follow pointers from node to node',
      'PostgreSQL requires self-joins on 20B edges (impossible at scale)',
      'Query complexity in graphs is proportional to subgraph size, not total graph size',
    ],
  },
  {
    id: 'graph-discussion-2',
    question:
      'Your fraud detection system needs to identify rings of suspicious accounts (accounts connected by shared devices, IP addresses, or transfers within 3 hops). This pattern matching is extremely difficult in SQL. Design a Neo4j solution using Cypher pattern matching.',
    sampleAnswer: `**Neo4j Schema:**

\`\`\`cypher
CREATE (a:Account {id: "A123", email: "user@example.com"})
CREATE (d:Device {fingerprint: "device123"})
CREATE (ip:IPAddress {address: "192.168.1.1"})
CREATE (a)-[:USES_DEVICE]->(d)
CREATE (a)-[:ACCESSED_FROM]->(ip)
CREATE (a1)-[:TRANSFERRED_TO {amount: 1000, date: "2024-01-15"}]->(a2)
\`\`\`

**Query: Find Fraud Ring (Connected Accounts)**
\`\`\`cypher
MATCH (suspicious:Account {flagged: true})
MATCH path = (suspicious)-[:USES_DEVICE|ACCESSED_FROM|TRANSFERRED_TO*1..3]-(connected:Account)
WHERE connected.flagged = false
RETURN DISTINCT connected, 
       length(path) AS hops,
       [rel IN relationships(path) | type(rel)] AS connection_types
\`\`\`

**Pattern: Shared Device Indicator**
\`\`\`cypher
MATCH (a1:Account)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(a2:Account)
WHERE a1 <> a2
RETURN a1.id, a2.id, d.fingerprint
\`\`\`

**Pattern: Circular Transfers (Money Laundering)**
\`\`\`cypher
MATCH path = (start:Account)-[:TRANSFERRED_TO*3..5]->(start)
WHERE ALL(rel IN relationships(path) WHERE rel.amount > 1000)
RETURN path
\`\`\`

**Why Graph Databases Excel:**
1. **Pattern matching**: Cypher naturally expresses "accounts connected via shared resources"
2. **Variable-depth traversal**: *1..3 finds paths of 1-3 hops
3. **Relationship types**: Multiple relationship types in single query
4. **Fast traversal**: Finds rings in seconds vs hours in SQL

**SQL Equivalent (Impossible at Scale):**
Would require recursive CTEs with UNION of multiple relationship tables,
traversing up to 3 levels—query would timeout on realistic data.`,
    keyPoints: [
      'Graph pattern matching naturally expresses "find connected nodes"',
      'Variable-depth traversal (*1..3) finds paths of different lengths',
      'Multiple relationship types in single query (USES_DEVICE|ACCESSED_FROM)',
      'SQL requires recursive CTEs with UNION across tables (complex, slow)',
      'Fraud rings often 2-4 hops deep—graph databases handle this efficiently',
    ],
  },
  {
    id: 'graph-discussion-3',
    question:
      'You\'re building a recommendation engine: "users who bought X also bought Y". With 10M users and 1M products, PostgreSQL queries with multiple JOINs time out. Design a collaborative filtering solution using Neo4j with specific queries for both item-based and user-based recommendations.',
    sampleAnswer: `**Neo4j Schema:**

\`\`\`cypher
CREATE (u:User {id: 123, name: "Alice"})
CREATE (p:Product {id: 456, name: "Laptop", category: "Electronics"})
CREATE (u)-[:PURCHASED {date: "2024-01-15", rating: 5}]->(p)
\`\`\`

**Approach 1: Item-Based (Users who bought X also bought Y)**

\`\`\`cypher
MATCH (target:Product {id: $productId})
MATCH (target)<-[:PURCHASED]-(u:User)-[:PURCHASED]->(recommendation:Product)
WHERE recommendation <> target
WITH recommendation, COUNT(DISTINCT u) AS frequency, 
     AVG(u.rating) AS avg_rating
RETURN recommendation.id, 
       recommendation.name,
       frequency,
       avg_rating
ORDER BY frequency DESC, avg_rating DESC
LIMIT 10
\`\`\`

**How it works:**
1. Find users who bought target product
2. Find other products those users bought
3. Count frequency and average rating
4. Return top 10

**Performance:** O(users_who_bought_X × their_purchases) = O(1000 × 50) = 50k traversals (~50ms)

**Approach 2: User-Based (Similar Users' Purchases)**

\`\`\`cypher
MATCH (me:User {id: $userId})-[:PURCHASED]->(p:Product)
WITH me, collect(p) AS my_products

MATCH (other:User)-[:PURCHASED]->(p:Product)
WHERE p IN my_products AND other <> me
WITH other, my_products, COUNT(DISTINCT p) AS overlap
WHERE overlap >= 3

MATCH (other)-[:PURCHASED]->(recommendation:Product)
WHERE NOT recommendation IN my_products
RETURN recommendation.name, 
       COUNT(*) AS frequency,
       AVG(overlap) AS similarity
ORDER BY frequency DESC
LIMIT 10
\`\`\`

**How it works:**
1. Find my purchased products
2. Find users who bought ≥3 of the same products (similar users)
3. Find products those similar users bought that I haven't
4. Return top 10

**Performance:** O(my_products × users_with_overlap × their_unique_products)

**Why PostgreSQL Fails:**

**SQL Equivalent:**
\`\`\`sql
SELECT p2.product_id, COUNT(*) AS frequency
FROM purchases p1
JOIN purchases p2 ON p1.user_id = p2.user_id
WHERE p1.product_id = 456  -- Target product
  AND p2.product_id != 456
GROUP BY p2.product_id
ORDER BY frequency DESC
LIMIT 10
\`\`\`

**Problem:** Self-join on 10M users × 50 purchases avg = 500M rows. 
Even with indexes, this creates a 500M × 500M intermediate result (timeout!).

**Neo4j Advantage:**
- Only traverses subgraph (users who bought product X)
- No massive intermediate results
- Constant performance regardless of total users/products

**Advanced: Hybrid Scoring**

\`\`\`cypher
MATCH (target:Product {id: $productId})<-[:PURCHASED]-(u)-[:PURCHASED]->(rec:Product)
WHERE rec <> target
WITH rec, COUNT(DISTINCT u) AS co_purchase_freq

MATCH (rec)<-[r:PURCHASED]-()
WITH rec, co_purchase_freq, 
     AVG(r.rating) AS avg_rating,
     COUNT(r) AS total_purchases

// Combine signals: co-purchases, ratings, popularity
RETURN rec.name,
       co_purchase_freq * 2 +      // Co-purchase weight
       avg_rating * 10 +            // Rating weight  
       LOG(total_purchases) * 5     // Popularity weight (logarithmic)
       AS score
ORDER BY score DESC
LIMIT 10
\`\`\`

**Result:** Personalized recommendations in <100ms vs PostgreSQL timeout or 10+ seconds.`,
    keyPoints: [
      'Item-based: Find products co-purchased with target product',
      'User-based: Find similar users (overlap in purchases), recommend their products',
      'Graph traversal is local (only touches relevant subgraph)',
      'PostgreSQL self-joins on 500M rows create massive intermediate results',
      'Neo4j performance independent of total users/products (only depends on subgraph size)',
    ],
  },
];
