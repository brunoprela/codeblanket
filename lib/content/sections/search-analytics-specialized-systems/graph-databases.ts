import { Section } from '@/lib/types';

const graphDatabasesSection: Section = {
  id: 'graph-databases',
  title: 'Graph Databases',
  content: `
# Graph Databases

## Introduction

**Graph databases** store data as nodes (entities) and edges (relationships), optimized for traversing connections between data. Unlike relational databases where relationships are modeled through foreign keys and require expensive JOIN operations, graph databases make relationships first-class citizens—they're as important as the data itself.

The world is fundamentally connected:
- **Social networks**: People know people, follow people, interact with people
- **Recommendation engines**: Users buy products, products relate to products
- **Knowledge graphs**: Entities have relationships, concepts connect to concepts
- **Fraud detection**: Accounts connect through devices, IPs, transfers
- **Network topology**: Routers connect to routers, dependencies between services

When your data is highly connected and you need to query relationships efficiently, graph databases excel where relational databases struggle.

## The Graph Data Model

### Nodes (Vertices)

**Nodes** represent entities with properties (key-value pairs).

\`\`\`
(:Person {name: "Alice", age: 30, city: "San Francisco"})
(:Company {name: "Acme Corp", founded: 2010})
(:Product {name: "Widget", price: 29.99})
\`\`\`

**Label**: Type of node (Person, Company, Product)
**Properties**: Attributes of the node

### Edges (Relationships)

**Edges** represent connections between nodes, with type and properties.

\`\`\`
(Alice)-[:WORKS_AT {since: 2020, role: "Engineer"}]->(Acme Corp)
(Alice)-[:FRIENDS_WITH {since: 2015}]->(Bob)
(Alice)-[:PURCHASED {date: "2024-01-15", quantity: 2}]->(Widget)
\`\`\`

**Relationship type**: WORKS_AT, FRIENDS_WITH, PURCHASED
**Direction**: From Alice to Acme Corp (directed edge)
**Properties**: Attributes of the relationship

### Property Graph Model

Both nodes AND edges can have arbitrary properties.

**Example social network**:
\`\`\`
(Alice:Person {name: "Alice", age: 30})
  -[:FRIENDS_WITH {since: 2015}]->
(Bob:Person {name: "Bob", age: 35})
  -[:FRIENDS_WITH {since: 2018}]->
(Charlie:Person {name: "Charlie", age: 32})
  -[:FRIENDS_WITH {since: 2015}]->
(Alice)
\`\`\`

**Query**: "Who are Alice's friends of friends?"

Graph database: Follow edges (constant time)
Relational database: Self-join friendships table (expensive)

## When to Use Graph Databases

### Perfect Use Cases

**1. Social Networks**
- Friends, followers, likes, comments, groups
- "Friends of friends" queries (network expansion)
- Community detection (who clusters together)
- Influence analysis (who has the most connections)

**2. Recommendation Engines**
- "Users who bought X also bought Y"
- "People similar to you liked Z"
- Collaborative filtering at scale

**3. Fraud Detection**
- Fraud rings (accounts connected through shared devices, IPs, bank accounts)
- Money laundering (circular transaction patterns)
- Identity verification (phone number used by 50 accounts = suspicious)

**4. Knowledge Graphs**
- Wikipedia: Articles connected by links
- Google Knowledge Graph: Entities and relationships
- Enterprise data: Products → Categories → Departments

**5. Network and IT Operations**
- Service dependencies ("What breaks if this service fails?")
- Root cause analysis (trace impact through dependency graph)
- Network topology (routers, switches, connections)

**6. Access Control**
- Users → Roles → Permissions
- Nested groups (user is in group A, group A is in group B with permission X)
- "What can this user access?" (traverse permission graph)

### When NOT to Use

**Simple 1:1 or 1:many relationships**: Relational DB is simpler
**No traversals**: If you never query relationships, graph DB is overkill
**Bulk analytics**: Data warehouses are better for aggregations across entire dataset
**Document storage**: If relationships are minimal, use document DB

**Rule of thumb**: If your queries frequently use 3+ JOINs in SQL, consider graph database.

## Neo4j

Neo4j is the most popular graph database, with Cypher as its query language.

### Creating Nodes

\`\`\`cypher
// Create person node
CREATE (alice:Person {name: "Alice", age: 30, city: "San Francisco"})

// Create multiple nodes
CREATE (bob:Person {name: "Bob", age: 35})
CREATE (charlie:Person {name: "Charlie", age: 32})
CREATE (acme:Company {name: "Acme Corp", founded: 2010})
\`\`\`

### Creating Relationships

\`\`\`cypher
// Find nodes and create relationship
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CREATE (a)-[:FRIENDS_WITH {since: 2015}]->(b)

// Create nodes and relationships in one statement
CREATE (alice:Person {name: "Alice"})-[:WORKS_AT {since: 2020}]->(acme:Company {name: "Acme Corp"})
\`\`\`

### Querying: Pattern Matching

**Cypher** uses ASCII-art patterns to match graph structures.

**Find Alice's friends**:
\`\`\`cypher
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN friend.name
\`\`\`

**Friends of friends (2-hop)**:
\`\`\`cypher
MATCH (me:Person {name: "Alice"})-[:FRIENDS_WITH]->()-[:FRIENDS_WITH]->(fof)
WHERE NOT (me)-[:FRIENDS_WITH]->(fof) AND me <> fof
RETURN fof.name
\`\`\`

**Shortest path**:
\`\`\`cypher
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(target:Person {name: "Zoe"})
)
RETURN path, length(path)
\`\`\`

### Indexes

\`\`\`cypher
// Create index for fast lookups
CREATE INDEX person_name FOR (p:Person) ON (p.name)
CREATE INDEX company_name FOR (c:Company) ON (c.name)

// Composite index
CREATE INDEX person_location FOR (p:Person) ON (p.city, p.age)
\`\`\`

**Critical**: Without indexes, \`MATCH (p:Person {name: "Alice"})\` scans ALL Person nodes.

## Use Case: Social Network

### Schema Design

\`\`\`cypher
// Nodes
(:Person {user_id, name, email, age, city, joined_date})
(:Post {post_id, content, timestamp, likes_count})
(:Comment {comment_id, content, timestamp})

// Relationships
(Person)-[:FRIENDS_WITH]->(Person)
(Person)-[:FOLLOWS]->(Person)
(Person)-[:POSTED]->(Post)
(Person)-[:LIKES]->(Post)
(Person)-[:COMMENTED_ON]->(Post)
(Comment)-[:REPLY_TO]->(Comment)
\`\`\`

### Query: Friend Recommendations

"Recommend people Alice might know (friends of friends)"

\`\`\`cypher
MATCH (me:Person {user_id: $userId})-[:FRIENDS_WITH]->(friend)-[:FRIENDS_WITH]->(recommendation)
WHERE NOT (me)-[:FRIENDS_WITH]->(recommendation) 
  AND me <> recommendation
WITH recommendation, COUNT(DISTINCT friend) AS mutual_friends
RETURN recommendation.user_id, 
       recommendation.name,
       mutual_friends
ORDER BY mutual_friends DESC
LIMIT 10
\`\`\`

**Explanation**:
1. Find my friends (1-hop)
2. Find their friends (2-hop)
3. Exclude people I already know
4. Count how many mutual friends we have
5. Sort by mutual friends (higher = better match)

**Performance**:
- Graph DB: O(friends × friends) = O(200 × 200) = 40,000 node visits (~50ms)
- Relational DB: Self-join on 20 billion friendships (timeout!)

### Query: Activity Feed

"Show posts from people I follow"

\`\`\`cypher
MATCH (me:Person {user_id: $userId})-[:FOLLOWS]->(person)-[:POSTED]->(post)
RETURN person.name, 
       post.content, 
       post.timestamp,
       post.likes_count
ORDER BY post.timestamp DESC
LIMIT 20
\`\`\`

**Relational equivalent** (3 tables, 2 JOINs):
\`\`\`sql
SELECT u.name, p.content, p.timestamp, p.likes_count
FROM users u
JOIN follows f ON u.user_id = f.followed_id
JOIN posts p ON f.followed_id = p.user_id
WHERE f.follower_id = ?
ORDER BY p.timestamp DESC
LIMIT 20
\`\`\`

Graph database eliminates JOINs—just follow edges.

### Query: Common Interests

"Find friends who like the same posts as me"

\`\`\`cypher
MATCH (me:Person {user_id: $userId})-[:LIKES]->(post)<-[:LIKES]-(friend)
WHERE (me)-[:FRIENDS_WITH]-(friend)
WITH friend, COUNT(DISTINCT post) AS common_likes
WHERE common_likes > 5
RETURN friend.name, common_likes
ORDER BY common_likes DESC
\`\`\`

## Use Case: Recommendation Engine

### Collaborative Filtering

**Problem**: Amazon has 100M users, 10M products. Recommend products to users based on purchase history.

**Relational approach** (slow):
\`\`\`sql
-- "Users who bought X also bought Y"
SELECT p2.product_id, COUNT(*) as frequency
FROM purchases p1
JOIN purchases p2 ON p1.user_id = p2.user_id
WHERE p1.product_id = ?
  AND p2.product_id != ?
GROUP BY p2.product_id
ORDER BY frequency DESC
LIMIT 10
\`\`\`

**Problem**: Self-join on purchases table (100M users × 50 purchases avg = 5 billion rows). Even with indexes, this is slow.

### Graph Approach

**Schema**:
\`\`\`cypher
(:User {user_id, name})
(:Product {product_id, name, category, price})
(:Category {name})

(User)-[:PURCHASED {date, quantity, rating}]->(Product)
(Product)-[:IN_CATEGORY]->(Category)
\`\`\`

**Query: Item-based recommendations**

"Users who bought product X also bought..."

\`\`\`cypher
MATCH (target:Product {product_id: $productId})<-[:PURCHASED]-(u:User)-[:PURCHASED]->(rec:Product)
WHERE rec <> target
WITH rec, COUNT(DISTINCT u) AS frequency, AVG(u.rating) AS avg_rating
RETURN rec.product_id, 
       rec.name,
       frequency,
       avg_rating
ORDER BY frequency DESC, avg_rating DESC
LIMIT 10
\`\`\`

**Performance**:
- Only traverses subgraph: users who bought product X (maybe 10,000 users)
- Then their purchases (10,000 × 50 = 500,000 purchases)
- Total: ~500K node visits in 50ms

Relational database: Must join 5 billion rows (slow!)

**Query: User-based recommendations**

"Users similar to me bought..."

\`\`\`cypher
// Step 1: Find my purchased products
MATCH (me:User {user_id: $userId})-[:PURCHASED]->(product:Product)
WITH me, COLLECT(product) AS my_products

// Step 2: Find users who bought similar products (overlap >= 3)
MATCH (other:User)-[:PURCHASED]->(product:Product)
WHERE product IN my_products AND other <> me
WITH other, my_products, COUNT(DISTINCT product) AS overlap
WHERE overlap >= 3

// Step 3: Find products those similar users bought that I haven't
MATCH (other)-[:PURCHASED]->(recommendation:Product)
WHERE NOT recommendation IN my_products
WITH recommendation, COUNT(DISTINCT other) AS frequency, AVG(overlap) AS similarity
RETURN recommendation.name, frequency, similarity
ORDER BY frequency DESC, similarity DESC
LIMIT 10
\`\`\`

**Advanced: Category-aware recommendations**

\`\`\`cypher
MATCH (target:Product {product_id: $productId})-[:IN_CATEGORY]->(cat:Category)
MATCH (target)<-[:PURCHASED]-(u:User)-[:PURCHASED]->(rec:Product)-[:IN_CATEGORY]->(cat)
WHERE rec <> target
WITH rec, COUNT(DISTINCT u) AS frequency
RETURN rec.product_id, rec.name, frequency
ORDER BY frequency DESC
LIMIT 10
\`\`\`

This recommends products in the **same category** as the target product.

## Use Case: Fraud Detection

### Finding Fraud Rings

**Problem**: Fraudsters create multiple accounts, but they share devices, IP addresses, phone numbers, or transfer money between accounts.

**Schema**:
\`\`\`cypher
(:Account {account_id, email, created_date, flagged})
(:Device {device_id, fingerprint})
(:IPAddress {address, country})
(:Phone {number})
(:BankAccount {account_number, routing_number})

(Account)-[:USES_DEVICE]->(Device)
(Account)-[:ACCESSED_FROM]->(IPAddress)
(Account)-[:PHONE_NUMBER]->(Phone)
(Account)-[:LINKED_BANK]->(BankAccount)
(Account)-[:TRANSFERRED_TO {amount, date}]->(Account)
\`\`\`

### Query: Connected Accounts

"Find accounts connected to this suspicious account (within 3 hops)"

\`\`\`cypher
MATCH (suspicious:Account {account_id: $accountId, flagged: true})
MATCH path = (suspicious)-[:USES_DEVICE|ACCESSED_FROM|TRANSFERRED_TO|PHONE_NUMBER|LINKED_BANK*1..3]-(connected:Account)
WHERE connected.flagged = false
RETURN DISTINCT connected.account_id, 
       connected.email,
       length(path) AS hops,
       [rel IN relationships(path) | type(rel)] AS connection_types
\`\`\`

**Explanation**:
- Start with known fraudulent account
- Traverse up to 3 hops through ANY relationship type
- Find connected accounts that aren't yet flagged
- Return connection path details

**Result**: Discover fraud ring where accounts share devices/IPs/phones.

### Pattern: Shared Device

"Find accounts using the same device"

\`\`\`cypher
MATCH (a1:Account)-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(a2:Account)
WHERE a1 <> a2
RETURN a1.account_id, a2.account_id, d.fingerprint, COUNT(*) AS shared_devices
\`\`\`

**Red flag**: 20 different accounts using the same device fingerprint.

### Pattern: Circular Transfers (Money Laundering)

"Find circular money transfers (A → B → C → A)"

\`\`\`cypher
MATCH path = (start:Account)-[:TRANSFERRED_TO*3..5]->(start)
WHERE ALL(rel IN relationships(path) WHERE rel.amount > 1000)
  AND length(path) >= 3
RETURN path, [rel IN relationships(path) | rel.amount] AS amounts
\`\`\`

**Explanation**:
- \`*3..5\`: Variable-length path (3 to 5 hops)
- \`->(start)\`: Ends at the starting account (circular)
- \`ALL(rel WHERE rel.amount > 1000)\`: All transfers > $1,000

**Result**: Detect money laundering patterns.

### Pattern: Velocity Check

"Account opened 20 accounts in 1 hour from same IP"

\`\`\`cypher
MATCH (ip:IPAddress)<-[:ACCESSED_FROM]-(account:Account)
WHERE account.created_date > datetime() - duration('PT1H')
WITH ip, COUNT(DISTINCT account) AS accounts_created
WHERE accounts_created > 20
RETURN ip.address, accounts_created
\`\`\`

## Relational vs Graph: Comparison

### Friends of Friends Example

**Relational (SQL)**:

**Schema**:
\`\`\`sql
CREATE TABLE persons (id, name, age);
CREATE TABLE friendships (person1_id, person2_id, since);
\`\`\`

**Query**:
\`\`\`sql
SELECT p3.name
FROM persons p1
JOIN friendships f1 ON p1.id = f1.person1_id
JOIN persons p2 ON f1.person2_id = p2.id
JOIN friendships f2 ON p2.id = f2.person1_id
JOIN persons p3 ON f2.person2_id = p3.id
WHERE p1.name = 'Alice'
  AND p3.id NOT IN (
    SELECT person2_id FROM friendships WHERE person1_id = p1.id
  )
  AND p3.id <> p1.id;
\`\`\`

**Problems**:
- 4 table JOINs for 2-hop query
- 6 JOINs for 3-hop query (N hops = 2N JOINs)
- Subquery to filter existing friends (expensive)
- Performance degrades exponentially with depth
- Complex SQL difficult to read/maintain

**Execution plan**:
1. Scan persons (find Alice): 100M rows
2. JOIN friendships: 20B rows
3. JOIN persons again: Intermediate result × 100M
4. JOIN friendships again: Huge intermediate result
5. JOIN persons third time: Result explosion

### Graph (Cypher)

**Query**:
\`\`\`cypher
MATCH (me:Person {name: "Alice"})-[:FRIENDS_WITH]->()-[:FRIENDS_WITH]->(fof)
WHERE NOT (me)-[:FRIENDS_WITH]->(fof) AND me <> fof
RETURN fof.name
\`\`\`

**Advantages**:
- Single query, intuitive syntax
- No JOINs (follow pointers)
- Performance consistent regardless of graph size
- **Query complexity proportional to subgraph, not entire graph**

**Execution**:
1. Index lookup: Find Alice (O(log n))
2. Follow FRIENDS_WITH edges: 200 friends (O(1) per edge)
3. For each friend, follow their FRIENDS_WITH edges: 200 × 200 = 40,000 (O(1) per edge)
4. Filter: Already friends? (O(1) per check)
5. **Total: ~40,000 node visits in 50ms**

### Key Difference: Index-Free Adjacency

**Relational database**:
- Relationships stored as foreign keys
- JOIN requires index lookup for EVERY row
- Intermediate result sets materialize in memory

**Graph database**:
- Each node stores pointers to adjacent nodes
- Traversal is O(1)—follow pointer, no index lookup
- No intermediate results (stream processing)

**Analogy**:
- Relational: Phone book (lookup by name every time)
- Graph: Personal contacts (direct connections)

## Graph Algorithms

Neo4j Graph Data Science library provides graph algorithms.

### PageRank (Importance/Influence)

"Who are the most influential people in the network?"

\`\`\`cypher
CALL gds.pageRank.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10
\`\`\`

**Use case**: Find thought leaders, influential accounts, hub nodes.

### Community Detection (Louvain)

"Find clusters/communities in the graph"

\`\`\`cypher
CALL gds.louvain.stream('myGraph')
YIELD nodeId, communityId
WITH communityId, COLLECT(gds.util.asNode(nodeId).name) AS members
WHERE SIZE(members) > 10
RETURN communityId, members, SIZE(members) AS size
ORDER BY size DESC
\`\`\`

**Use case**: Detect fraud rings, find friend groups, segment users.

### Shortest Path

"What's the shortest path between Alice and Zoe?"

\`\`\`cypher
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FRIENDS_WITH*]-(zoe:Person {name: "Zoe"})
)
RETURN path, length(path) AS hops
\`\`\`

**Use case**: Degrees of separation, network analysis.

### Centrality Measures

**Betweenness centrality**: "Who acts as a bridge between communities?"

\`\`\`cypher
CALL gds.betweenness.stream('myGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name, score
ORDER BY score DESC
LIMIT 10
\`\`\`

**Use case**: Find key connectors, potential single points of failure in networks.

## Amazon Neptune

Amazon Neptune is AWS's managed graph database supporting two models:

**1. Property Graphs (Gremlin/Apache TinkerPop)**
**2. RDF Graphs (SPARQL)**

### Gremlin Example

\`\`\`groovy
// Add vertices
g.addV('Person').property('name', 'Alice').property('age', 30)
g.addV('Person').property('name', 'Bob').property('age', 35)

// Add edge
g.V().has('Person', 'name', 'Alice')
  .addE('FRIENDS_WITH')
  .to(g.V().has('Person', 'name', 'Bob'))
  .property('since', 2015)

// Query: Friends of friends
g.V().has('Person', 'name', 'Alice')
  .out('FRIENDS_WITH')
  .out('FRIENDS_WITH')
  .dedup()
  .values('name')
\`\`\`

**Gremlin** is more imperative (step-by-step traversal) vs Cypher (declarative pattern matching).

## Performance Considerations

### 1. Indexing

**Critical**: Index properties used in MATCH clauses.

\`\`\`cypher
// Without index: Scans ALL Person nodes
MATCH (p:Person {name: "Alice"})

// With index: O(log n) lookup
CREATE INDEX person_name FOR (p:Person) ON (p.name)
\`\`\`

### 2. Avoid Cartesian Products

**Bad**:
\`\`\`cypher
MATCH (p1:Person), (p2:Person)
WHERE p1.city = p2.city
RETURN p1, p2
\`\`\`

**Problem**: 100M persons × 100M persons = 10 quadrillion comparisons!

**Good**:
\`\`\`cypher
MATCH (p1:Person)-[:LIVES_IN]->(city:City)<-[:LIVES_IN]-(p2:Person)
RETURN p1, p2
\`\`\`

Follow edges, don't compare all pairs.

### 3. Limit Traversal Depth

**Dangerous**:
\`\`\`cypher
MATCH path = (a:Person)-[*]-(b:Person)  // Unbounded!
WHERE a.name = 'Alice'
RETURN path
\`\`\`

**Problem**: Could traverse entire graph (millions of paths).

**Safe**:
\`\`\`cypher
MATCH path = (a:Person)-[*1..3]-(b:Person)  // Max 3 hops
WHERE a.name = 'Alice'
RETURN path
LIMIT 100
\`\`\`

### 4. Use Relationship Types

**Slow**:
\`\`\`cypher
MATCH (a:Account)-[*1..3]-(b:Account)  // Traverses ALL relationship types
\`\`\`

**Fast**:
\`\`\`cypher
MATCH (a:Account)-[:USES_DEVICE|TRANSFERRED_TO*1..3]-(b:Account)  // Only relevant types
\`\`\`

### 5. Query Planning

Use \`PROFILE\` to see execution plan:

\`\`\`cypher
PROFILE
MATCH (p:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN friend.name
\`\`\`

Shows:
- Index usage
- Number of rows scanned
- Database hits

## Scaling Graph Databases

### Challenges

**Graph queries often need the entire graph**:
- "Find shortest path from A to B" might span multiple shards
- Fraud ring detection needs to traverse connected accounts

**Sharding is complex**:
- Can't easily partition connected data
- Cross-shard queries require network hops

### Solutions

**1. Read Replicas**:
- Read-heavy workloads (common in graphs)
- Scale reads horizontally

**2. Causal Clustering (Neo4j Enterprise)**:
- Core servers (write consensus)
- Read replicas (eventually consistent reads)

**3. Specialized sharding**:
- Partition by geography (users in US on shard 1, Europe on shard 2)
- Some queries stay local, some cross shards

**Reality**: Vertical scaling (bigger single node) often more practical than horizontal.

## Best Practices

### 1. Model Relationships Explicitly

**Bad**: Store relationship as node property
\`\`\`cypher
(:Person {name: "Alice", friends: ["Bob", "Charlie"]})  // Array of names
\`\`\`

**Good**: Create relationship edges
\`\`\`cypher
(alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(bob:Person {name: "Bob"})
\`\`\`

### 2. Use Specific Relationship Types

**Bad**: Generic relationship
\`\`\`cypher
(alice)-[:RELATED_TO {type: "friend"}]->(bob)
\`\`\`

**Good**: Specific type
\`\`\`cypher
(alice)-[:FRIENDS_WITH]->(bob)
\`\`\`

Enables efficient traversal: \`-[:FRIENDS_WITH]->\` vs \`-[r:RELATED_TO WHERE r.type='friend']->\`

### 3. Denormalize When Appropriate

**Trade-off**: Duplicate data vs query performance

**Example**: Store user's friend count on Person node
\`\`\`cypher
(:Person {name: "Alice", friend_count: 342})
\`\`\`

Instead of counting every query:
\`\`\`cypher
MATCH (alice:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)
RETURN COUNT(friend)  // Slow for users with 10k friends
\`\`\`

### 4. Use Parameters

**Bad** (query plan not cached):
\`\`\`cypher
MATCH (p:Person {name: "Alice"}) RETURN p
MATCH (p:Person {name: "Bob"}) RETURN p  // Different query plan!
\`\`\`

**Good** (query plan reused):
\`\`\`cypher
MATCH (p:Person {name: $name}) RETURN p  // Same plan, different parameter
\`\`\`

### 5. Monitor Query Performance

Watch for:
- Cartesian products (queries slow down over time)
- Missing indexes (node scans)
- Unbounded traversals (\`[*]\` without limit)

## Limitations

### 1. Not for Bulk Analytics

**Bad fit**: "Calculate average age of all users"

Graph DBs optimize for traversals, not table scans. Use data warehouse for analytics.

### 2. Complex Sharding

Horizontal scaling is harder than traditional databases.

### 3. Learning Curve

New query language (Cypher, Gremlin), new data modeling mindset.

### 4. Overkill for Simple Relationships

If your relationships are simple (1:1, 1:many with single JOIN), relational DB is simpler.

## Trade-Offs

### Graph DB vs Relational

| Aspect | Graph DB | Relational |
|--------|----------|------------|
| **Relationship queries** | Fast (follow pointers) | Slow (JOINs) |
| **Query complexity** | Proportional to subgraph | Proportional to entire dataset |
| **Data model** | Flexible (schema-optional) | Rigid (schema-required) |
| **ACID** | Yes (Neo4j) | Yes |
| **Horizontal scaling** | Complex | Straightforward (shard by key) |
| **Learning curve** | Moderate (Cypher) | Low (SQL standard) |
| **Best for** | Connected data, traversals | Tabular data, aggregations |

### When to Choose Graph

- Relationships are first-class (as important as data)
- Need to traverse 2+ hops frequently
- Variable-depth queries ("find all connected nodes")
- Pattern matching (fraud rings, recommendations)
- SQL queries have 3+ JOINs consistently

### When to Stick with Relational

- Simple 1:1 or 1:many relationships
- Primarily CRUD operations (create, read, update, delete)
- No deep traversals
- Team expertise in SQL
- Mature relational ecosystem needed

## Summary

Graph databases excel at:

**Relationship traversal**: Friends of friends, shortest path
**Pattern matching**: Fraud rings, recommendation patterns
**Variable-depth queries**: "Find all connected nodes within 5 hops"
**Performance**: Query time proportional to subgraph, not entire graph

**Key concepts**:
- **Nodes**: Entities with properties
- **Edges**: Relationships with type and properties
- **Index-free adjacency**: Follow pointers, no index lookups
- **Cypher**: Declarative pattern matching language

**Popular choices**:
- **Neo4j**: Most mature, Cypher, graph algorithms
- **Amazon Neptune**: Managed, supports Gremlin and SPARQL
- **ArangoDB**: Multi-model (graph + document + key-value)
- **JanusGraph**: Distributed, built on Cassandra/HBase

**Use cases**:
- Social networks (friends, followers, interactions)
- Recommendations (collaborative filtering)
- Fraud detection (connected accounts, circular transfers)
- Knowledge graphs (entities and relationships)
- Network analysis (dependencies, topology)

Use graph databases when relationships are as important as the data itself, and relational JOINs become a performance bottleneck.
`,
  mcqQuizId: 'graph-databases-mcq',
  discussionQuizId: 'graph-databases-discussion',
};

export default graphDatabasesSection;
