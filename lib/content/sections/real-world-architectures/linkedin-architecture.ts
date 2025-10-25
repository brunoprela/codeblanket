/**
 * LinkedIn Architecture Section
 */

export const linkedinarchitectureSection = {
  id: 'linkedin-architecture',
  title: 'LinkedIn Architecture',
  content: `LinkedIn is the world's largest professional networking platform with over 900 million members across 200+ countries. LinkedIn\'s architecture powers complex social graph operations, feed generation, job matching, and one of the largest data infrastructures in the world. This section explores the technical systems behind LinkedIn, including many innovations that have become industry standards.

## Overview

LinkedIn's scale and challenges:
- **900+ million members** worldwide
- **58+ million companies** on platform
- **15+ million job postings** at any time
- **9 billion content impressions** per week
- **100+ petabytes** of data
- **Billions of API calls** per day

### Key Architectural Challenges

1. **Social graph**: Store and query 900M members with complex relationships
2. **Feed generation**: Personalized feeds for each member (billions of feed views daily)
3. **Search**: Members, jobs, companies, content across massive index
4. **Recommendations**: People You May Know, job recommendations, content recommendations
5. **Messaging**: Real-time messaging for 900M members
6. **Data infrastructure**: Process and analyze petabytes of data

---

## Evolution of LinkedIn's Architecture

### Phase 1: Monolithic Application (2003-2010)

Early LinkedIn was built as a Java monolith called **Leo**.

\`\`\`
Browser → Load Balancer → Leo (Java monolith)
                                ↓
                         Oracle Database
\`\`\`

**Challenges at Scale**:
- Single database couldn't handle load
- Tight coupling made changes risky
- Slow deployment cycle (deploy entire app)
- Database read replicas maxed out

---

### Phase 2: Service-Oriented Architecture (2010-2015)

LinkedIn decomposed Leo into services.

**Key Services**:
- **Member Service**: Member profiles, connections
- **Communication Service**: Messaging, InMail
- **Jobs Service**: Job postings, applications
- **Feed Service**: Activity feed generation
- **Search Service**: Member and job search

**Communication**:
- REST APIs between services
- Later: RPC framework called **Distributed Service Framework** (DSF)

**Data Stores**:
- **Voldemort**: Distributed key-value store (LinkedIn-built, inspired by Amazon Dynamo)
- **Espresso**: Distributed document store (LinkedIn-built)
- **MySQL**: For some legacy data
- **Oracle**: Phased out over time

---

### Phase 3: Data-Driven Platform (2015-present)

LinkedIn evolved into a data-driven platform.

**Innovations**:
- **Apache Kafka**: Event streaming platform (created at LinkedIn, open-sourced)
- **Samza**: Stream processing framework (LinkedIn-built)
- **Venice**: Derived data platform
- **Pinot**: Real-time analytics database
- **Rest.li**: REST framework for services

---

## Core Components

### 1. Social Graph (Economic Graph)

LinkedIn models the **Economic Graph**: members, companies, jobs, skills, schools.

**Graph Structure**:

\`\`\`
Members:
- 900M+ members
- Connections: A is connected to B (bidirectional)
- Followers: A follows B (unidirectional, for influencers)

Companies:
- 58M+ companies
- Employees: Members work at companies
- Followers: Members follow companies

Jobs:
- 15M+ active job postings
- Applications: Members apply to jobs
- Companies post jobs

Skills:
- 35K+ standardized skills
- Members have skills
- Jobs require skills
- Skills endorsed by connections

Schools:
- Universities, colleges
- Members attended schools
- Alumni connections
\`\`\`

**Data Model**:

LinkedIn uses **graph database** concepts but implements on top of custom stores.

**Espresso** (Distributed Document Store):

\`\`\`json
Member Document:
{
  "member_id": "12345",
  "first_name": "John",
  "last_name": "Doe",
  "headline": "Software Engineer at Google",
  "location": "San Francisco, CA",
  "industry": "Technology",
  "connections": ["67890", "54321", ...],  // Array of member IDs
  "connection_count": 500,
  "follower_count": 1200,
  "skills": ["Python", "Machine Learning", "Distributed Systems"],
  "experience": [
    {
      "company": "Google",
      "title": "Software Engineer",
      "start_date": "2020-01",
      "end_date": null
    },
    ...
  ],
  "education": [...]
}
\`\`\`

**Connection Storage**:

Connections stored in **Voldemort** (key-value store) for fast lookups.

\`\`\`
Key: member:{id}:connections
Value: [connection1_id, connection2_id, ..., connection500_id]

Query: "Get all connections for member X" → O(1) lookup, return list
\`\`\`

**Graph Queries**:

**1st-Degree Connections** (Direct connections):
\`\`\`
GET member:{id}:connections
\`\`\`

**2nd-Degree Connections** (Friends of friends):
\`\`\`
For each 1st-degree connection:
  GET member:{connection_id}:connections
Merge and deduplicate
Filter out 1st-degree and self
\`\`\`

**Challenge**: Member with 500 connections, each having 500 connections = 250K 2nd-degree connections!

**Solution: Pre-compute and sample**:
- Batch job computes 2nd-degree connections nightly
- Store top 1000 (sorted by mutual connections)
- Good enough for "People You May Know"

---

### 2. Feed Generation

LinkedIn feed shows posts, articles, job changes, connections, ads.

**Feed Requirements**:
- **Personalized**: Different content for each member
- **Real-time**: New posts appear quickly
- **Relevance**: High-quality, engaging content
- **Scale**: Billions of feed views per week

**Architecture**:

**Feed Mixer** (Hybrid Fanout):

**Fanout-on-Write** (for regular members):
\`\`\`
When member posts:
1. Fetch connections (followers)
2. For each connection, write post to their feed
3. Store in Venice (derived data platform)
\`\`\`

**Fanout-on-Read** (for influencers):
\`\`\`
When influencer posts:
1. Don't fanout (too many followers)
2. Store post in influencer's outbox

When member requests feed:
1. Fetch pre-computed feed (fanout-on-write posts)
2. Check if member follows any influencers
3. Fetch recent posts from influencers' outboxes
4. Merge all posts
5. Rank using ML model
\`\`\`

**Feed Storage** (Venice):

**Venice** is LinkedIn\'s derived data platform:
- Stores pre-computed data (feeds, recommendations)
- Built on top of Kafka (change data capture)
- Read-optimized (SSD storage, aggressive caching)

\`\`\`
Key: feed:{member_id}
Value: [post_id_1, post_id_2, ..., post_id_100]

Query: GET feed:{member_id}
       → Returns list of post IDs
       → Fetch post details (batch query to Espresso)
       → Return to client
\`\`\`

**Feed Ranking**:

LinkedIn uses ML to rank feed posts.

**Goal**: Show content member will engage with (like, comment, share)

**Features**:

**Post Features**:
- Content type (text, image, video, article)
- Length, hashtags, mentions
- Author (connection, influencer, company)
- Age (recency)

**Member Features**:
- Engagement history (liked similar posts)
- Interests (extracted from profile, activity)
- Activity level (active vs passive user)

**Context Features**:
- Time of day, day of week
- Device (mobile, desktop)
- Previous session activity

**Model**:
- **Deep Neural Network** (TensorFlow)
- Trained on billions of (member, post, engagement) tuples
- Predict: P(engagement | member, post)
- Rank posts by predicted engagement score

**Training Pipeline**:
\`\`\`
1. Kafka streams all feed impressions and engagements
2. Samza processes stream, joins impressions with engagements
3. Features extracted and stored in Hadoop (HDFS)
4. Spark trains neural network on historical data (last 30 days)
5. Model exported and deployed to scoring service
6. Feed service calls scoring service for inference
7. Retrain model daily with new data
\`\`\`

**Serving**:
- Scoring service (gRPC)
- Inference: <50ms for batch of 100 posts
- Cache scores in Redis (TTL: 5 minutes)

---

### 3. People You May Know (PYMK)

PYMK is LinkedIn's recommendation system for suggesting connections.

**Signals**:

**1. Mutual Connections** (Strongest signal):
- A and B have many mutual connections → Likely know each other
- Query: 2nd-degree connections sorted by mutual connection count

**2. Shared Attributes**:
- **Same company**: Worked at same company (current or past)
- **Same school**: Attended same university
- **Same location**: Live in same city
- **Same industry**: Work in same industry
- **Same skills**: Have overlapping skills

**3. Profile Views**:
- A viewed B's profile → B is relevant to A
- Asymmetric signal (A might be job hunting, researching B)

**4. Similar Profiles**:
- Content-based similarity (cosine similarity of skill vectors)
- Members with similar profiles often connect

**ML Ranking**:

Train model to predict connection probability.

\`\`\`python
Features:
- mutual_connection_count
- same_company (boolean)
- same_school (boolean)
- same_location (boolean)
- skill_overlap_count
- profile_view_count (A viewed B)
- network_distance (2nd-degree, 3rd-degree)
- member_activity (active vs inactive)

Label: Did member send connection request? (1/0)

Model: Gradient Boosted Trees (XGBoost)

Predict: P(connection | member A, candidate B)
\`\`\`

**Candidate Generation**:

\`\`\`
For each member:
1. Get 2nd-degree connections (top 1000 by mutual connections)
2. Get coworkers (same company)
3. Get alumni (same school)
4. Get location matches (same city)
5. Deduplicate (union of all candidates)
6. Filter out: Already connected, invitations pending, declined invitations
7. Rank by ML model (top 10-20 candidates)
8. Store in Venice (PYMK feed)
9. Refresh daily
\`\`\`

**Serving**:
\`\`\`
GET /pymk/{member_id}
  → Query Venice: pymk:{member_id}
  → Returns list of candidate member IDs + scores
  → Fetch candidate profiles (batch query to Espresso)
  → Return to client
\`\`\`

---

### 4. Job Recommendations

LinkedIn matches members to relevant jobs.

**Two-Sided Marketplace**:
- **Job seekers**: Members looking for jobs
- **Recruiters**: Companies looking for candidates

**Matching Algorithms**:

**Member → Jobs** (Job Recommendations):

**Candidate Generation**:
\`\`\`
For member:
1. Extract profile features:
   - Current title: "Software Engineer"
   - Skills: ["Python", "Machine Learning"]
   - Location: "San Francisco, CA"
   - Experience level: "Mid-level" (3-7 years)
   
2. Query job index (Elasticsearch):
   - Title matches: "Software Engineer", "ML Engineer"
   - Required skills: Python OR Machine Learning
   - Location: Within 25 miles of SF
   - Experience: Mid-level
   
3. Return top 1000 candidate jobs
\`\`\`

**Ranking**:
\`\`\`
Features:
- Title similarity (member title vs job title)
- Skill match (overlap between member skills and job requirements)
- Location distance
- Company match (member's company preferences)
- Salary range (if member specified)
- Job age (recency)
- Application rate (popularity of job)

Model: Neural network (TensorFlow)
Predict: P(apply | member, job)
Rank jobs by predicted application probability
\`\`\`

**Jobs → Members** (Recruiter Search):

Recruiters search for candidates using filters and keywords.

**Search Index** (Elasticsearch):
\`\`\`json
Member Document:
{
  "member_id": "12345",
  "name": "John Doe",
  "current_title": "Software Engineer",
  "current_company": "Google",
  "location": "San Francisco, CA",
  "skills": ["Python", "Machine Learning", "Distributed Systems"],
  "years_of_experience": 5,
  "education": [...],
  "open_to_work": true  // Signal: actively looking
}
\`\`\`

**Recruiter Query**:
\`\`\`
Search: "Software Engineer with Python experience in Bay Area"
Filters:
- Location: Bay Area
- Skills: Python (required)
- Experience: 3-10 years

Elasticsearch query:
- Full-text search on title, skills
- Geo filter (location)
- Range filter (experience)

Ranking:
- Relevance score (keyword match)
- Open to work (boost if true)
- Profile completeness
- Activity (active members ranked higher)
\`\`\`

---

### 5. Search

LinkedIn search covers members, jobs, companies, posts, groups.

**Search Infrastructure**: **Galene** (LinkedIn\'s custom search engine)

**Galene Architecture**:
- Built on Apache Lucene (like Elasticsearch)
- Distributed across 100s of nodes
- Sharded by entity type (members, jobs, companies)

**Indexing**:

**Member Index**:
\`\`\`json
{
  "member_id": "12345",
  "name": "John Doe",
  "headline": "Software Engineer at Google",
  "current_company": "Google",
  "current_title": "Software Engineer",
  "location": "San Francisco, CA",
  "industry": "Technology",
  "skills": ["Python", "Machine Learning", "Distributed Systems"],
  "experience": [...],  // Full text indexed
  "education": [...],   // Full text indexed
  "connection_count": 500,
  "follower_count": 1200
}
\`\`\`

**Search Ranking**:

\`\`\`
Factors:
1. Relevance: Keyword match (title, headline, skills)
2. Social graph: Connections (1st-degree, 2nd-degree)
3. Popularity: Profile views, connection count
4. Activity: Recent activity (posts, comments)
5. Completeness: Profile completeness score
6. Personalization: Match user's network, interests

Model: Learning-to-rank (LambdaMART)
Trained on search clicks and connection requests
\`\`\`

**Typeahead / Autocomplete**:

As user types, suggest completions.

\`\`\`
User types: "John D"

Autocomplete service:
1. Query prefix index (trie data structure)
2. Return top 10 matches:
   - "John Doe" (1st-degree connection) → Boosted
   - "John Davis" (2nd-degree, high profile views)
   - "John Duncan" (same company)
3. Display with avatars and headlines
\`\`\`

**Faceted Search**:

Filters to narrow results:
- Connections (1st, 2nd, 3rd+)
- Location (cities, regions, countries)
- Company (current, past)
- Industry
- School

Implemented as Elasticsearch aggregations.

---

### 6. Messaging (LinkedIn InMail and Chat)

LinkedIn messaging includes:
- **InMail**: Messages to people outside your network (paid feature)
- **Chat**: Real-time messaging with connections

**Architecture**:

**Akka** (Actor Model Framework):
- Each conversation = Akka actor
- Actors handle millions of concurrent connections
- Fault-tolerant (supervisor strategy)
- Distributed across cluster

**Message Flow**:

\`\`\`
1. Member A sends message to Member B
2. Client → API Gateway → Messaging Service
3. Messaging Service looks up conversation actor
   - If actor exists: Forward message to actor
   - If not exists: Create actor for conversation
4. Actor stores message in Espresso (persistence)
5. Actor forwards message to Member B's connection
   - If Member B online: Push via WebSocket
   - If Member B offline: Store as unread
6. Send push notification (mobile) and email (if enabled)
\`\`\`

**Real-Time Delivery**:

\`\`\`
Member online:
- WebSocket connection to messaging server
- Server pushes messages instantly
- Latency: <100ms

Member offline:
- Message stored as unread
- Push notification sent (APNs, FCM)
- Email notification (optional)
- When member comes online, fetch unread messages
\`\`\`

**Message Storage** (Espresso):

\`\`\`json
Conversation Document:
{
  "conversation_id": "abc123",
  "participants": ["member_id_1", "member_id_2"],
  "messages": [
    {
      "message_id": "msg_001",
      "sender_id": "member_id_1",
      "text": "Hi, I saw your profile...",
      "timestamp": 1698158400,
      "read": false
    },
    ...
  ],
  "last_message_at": 1698158400
}
\`\`\`

**Sharding**:
- Conversations sharded by conversation_id
- Hash (conversation_id) → Shard
- All messages for conversation on same shard (data locality)

**Scalability**:
- Akka actors scale horizontally
- Add more servers → Distribute actors
- Akka cluster handles actor placement and migration

---

### 7. Data Infrastructure (Kafka, Hadoop, Spark)

LinkedIn pioneered several data technologies.

### Apache Kafka

**LinkedIn created Kafka** (open-sourced 2011):
- Event streaming platform
- Publish-subscribe messaging
- Horizontal scalability
- Distributed, partitioned, replicated

**Use Cases at LinkedIn**:
- **Activity streams**: Member actions (page views, clicks, posts)
- **Metrics**: Application metrics, logs
- **Change data capture (CDC)**: Database changes streamed to Kafka
- **Real-time analytics**: Stream processing with Samza

**Scale at LinkedIn**:
- **7 trillion messages** per day
- **4.5 petabytes** per day
- **100+ Kafka clusters**
- **4000+ brokers**

**Example: Feed Generation**:
\`\`\`
Member posts → Kafka topic: "member_posts"
              ↓
Feed Service consumes event
              ↓
Fanout post to connections' feeds (Venice)
\`\`\`

---

### Hadoop and Spark

**Hadoop** (HDFS + MapReduce):
- Store petabytes of data
- Batch processing (nightly jobs)

**Spark**:
- Replace MapReduce (10-100x faster)
- In-memory processing
- ML model training

**Use Cases**:
- **Data warehouse**: Store all LinkedIn data (profiles, activity, messages)
- **ML training**: Train models on historical data
- **Analytics**: Business intelligence, reporting
- **Recommendations**: Pre-compute PYMK, job recommendations

---

### Venice (Derived Data Platform)

**Venice** stores pre-computed, read-optimized data.

**Architecture**:
\`\`\`
Kafka (change stream) → Venice Writer → Venice Storage (SSD)
                                             ↓
                                      Venice Router
                                             ↓
                                      Application (read)
\`\`\`

**How it works**:
1. Application writes data to primary store (Espresso, MySQL)
2. Change captured in Kafka (CDC)
3. Venice Writer consumes Kafka, transforms data
4. Writes to Venice Storage (SSD, optimized for reads)
5. Application reads from Venice (low latency, high throughput)

**Use Cases**:
- **Feeds**: Pre-computed member feeds
- **Recommendations**: PYMK, job recommendations
- **Feature stores**: ML features for online inference

**Benefits**:
- **Read performance**: SSD storage, aggressive caching
- **Write isolation**: Writes don't impact reads (separate paths)
- **Derivation**: Transform data for specific access patterns

---

## Technology Stack

### Backend

- **Java**: Primary backend language (services, data processing)
- **Scala**: Data processing (Spark jobs), some services
- **Python**: Data science, ML model development
- **Node.js**: Some frontend services

### Data Storage

- **Espresso**: Primary data store (distributed document database, LinkedIn-built)
- **Voldemort**: Key-value store (LinkedIn-built, eventually consistent)
- **Venice**: Derived data platform (read-optimized, LinkedIn-built)
- **MySQL**: Legacy relational data
- **Hadoop (HDFS)**: Data lake (petabytes of historical data)
- **Pinot**: Real-time analytics database (LinkedIn-built, open-sourced)

### Data Processing

- **Kafka**: Event streaming (LinkedIn-created, 7 trillion messages/day)
- **Samza**: Stream processing framework (LinkedIn-built)
- **Spark**: Batch processing, ML training
- **Airflow**: Workflow orchestration

### Search

- **Galene**: Search engine (LinkedIn-built, Lucene-based)

### Machine Learning

- **TensorFlow**: Deep learning models (feed ranking, recommendations)
- **XGBoost**: Gradient boosted trees (PYMK, job matching)
- **Scikit-learn**: Classical ML

### Infrastructure

- **Own datacenters**: LinkedIn runs own datacenters (not public cloud)
- **Some Azure**: After Microsoft acquisition (2016), some integration with Azure
- **Kubernetes**: Container orchestration (migrating to)
- **Terraform**: Infrastructure as code

---

## Key Lessons from LinkedIn Architecture

### 1. Custom Data Stores for Custom Needs

LinkedIn built Espresso, Voldemort, Venice because off-the-shelf solutions didn't meet requirements. Investments paid off at scale.

### 2. Kafka Revolutionized Data Infrastructure

Kafka (LinkedIn-created) became industry standard for event streaming. Enabled real-time data pipelines at LinkedIn and beyond.

### 3. Economic Graph Powers Insights

Modeling economy (members, companies, jobs, skills) enables powerful recommendations, analytics, and insights.

### 4. Pre-Computation (Venice) Improves Read Performance

Deriving and storing data optimized for reads (Venice) drastically improves latency and throughput for read-heavy workloads.

### 5. ML Everywhere

Feed ranking, PYMK, job recommendations, search ranking, spam detection all powered by ML. Continuous experimentation and improvement.

---

## Interview Tips

**Q: How does LinkedIn generate personalized feeds?**

A: Use hybrid fanout with ML ranking. Fanout-on-write for regular members: when member posts, write to followers' feeds (stored in Venice, read-optimized). Fanout-on-read for influencers (>10K followers): fetch posts on-demand from influencer's outbox, merge with pre-computed feed. Rank all posts using deep neural network: predict P(engagement | member, post) based on post features (content type, author, age), member features (engagement history, interests), context (time of day, device). Train model on billions of impressions and engagements streamed via Kafka. Serve predictions via scoring microservice (<50ms for 100 posts). Cache ranked feed in Redis (TTL: 5 minutes). Real-time updates: new posts invalidate cache, trigger feed regeneration.

**Q: How does LinkedIn implement "People You May Know"?**

A: Multi-stage approach. Candidate generation: (1) 2nd-degree connections (friends of friends), pre-computed nightly via Spark, top 1000 by mutual connections. (2) Same company (current/past coworkers). (3) Same school (alumni). (4) Same location, industry, skills. (5) Profile views (asymmetric signal). Deduplicate and filter (already connected, pending invitations). Ranking: Train XGBoost model on historical connection requests. Features: mutual_connection_count, same_company, same_school, skill_overlap, network_distance, member_activity. Predict P(connection | member, candidate). Rank candidates by score. Store top 10-20 in Venice (user_id → [recommended_member_ids]). Refresh daily. Serve via API, fetch candidate profiles from Espresso in batch. Diversity: Mix different signal types (work, school, location).

**Q: How does LinkedIn\'s search system work?**

A: Use Galene (LinkedIn's search engine, Lucene-based). Index member profiles: name, headline, title, company, skills, experience, education. Shard by entity type and geography. Search query: parse keywords, apply filters (location, industry, connections). Elasticsearch query with full-text search on indexed fields. Ranking: Learning-to-rank model (LambdaMART) trained on search clicks and connection requests. Features: keyword relevance, social graph distance (1st/2nd-degree), popularity (profile views, connections), activity (recent posts), completeness (profile score), personalization (match user's network). Return top 20 results. Typeahead: Prefix matching on names using trie, prioritize 1st-degree connections. Faceted search: Aggregations for filters (location, company, industry). Cache popular queries in Redis.

---

## Summary

LinkedIn\'s architecture demonstrates building a professional networking platform at massive scale:

**Key Takeaways**:

1. **Custom data stores**: Espresso (document), Voldemort (key-value), Venice (derived data) built for LinkedIn's needs
2. **Apache Kafka**: LinkedIn-created event streaming platform, 7 trillion messages/day, powers real-time data pipelines
3. **Economic graph**: Models global economy (members, companies, jobs, skills) for powerful insights
4. **Hybrid fanout**: Fanout-on-write for regular members, fanout-on-read for influencers, ML ranking
5. **People You May Know**: 2nd-degree connections + shared attributes + ML ranking
6. **Job recommendations**: Two-sided marketplace, match members to jobs and jobs to members
7. **Galene search**: Custom search engine, learning-to-rank, social graph integration
8. **Akka messaging**: Actor model for millions of concurrent connections, real-time delivery

LinkedIn's success comes from custom-built infrastructure (Espresso, Kafka, Venice, Galene), graph-based recommendations, and ML-powered personalization at every layer.
`,
};
