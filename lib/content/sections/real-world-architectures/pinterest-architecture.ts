/**
 * Pinterest Architecture Section
 */

export const pinterestarchitectureSection = {
  id: 'pinterest-architecture',
  title: 'Pinterest Architecture',
  content: `Pinterest is a visual discovery engine where users find and save ideas through "Pins" organized on boards. With 450+ million monthly active users saving 240+ billion Pins, Pinterest's architecture must handle massive image storage, visual search, personalized recommendations, and graph operations at scale. This section explores the technical systems that power Pinterest.

## Overview

Pinterest's scale and challenges:
- **450+ million monthly active users**
- **240 billion+ Pins** saved
- **5+ billion boards** created
- **2+ billion searches** per month
- **600+ million visual searches** per month
- **Petabytes of image data** stored

### Key Architectural Challenges

1. **Image storage and delivery**: Store and serve billions of images globally with low latency
2. **Visual search**: Find visually similar images using computer vision and ML
3. **Recommendations**: Personalize home feed, related Pins, and search results
4. **Graph operations**: Process complex user-Pin-board relationships at scale
5. **Real-time personalization**: Update interests and recommendations based on user activity
6. **Infrastructure scalability**: Handle traffic spikes (holidays, events)

---

## Evolution of Pinterest's Architecture

### Phase 1: Monolithic Application (2010-2012)

Early Pinterest was built as a Django (Python) monolith.

\`\`\`
Browser → Django App → MySQL + Memcached → S3 (images)
\`\`\`

**Simple Stack**:
- Django for web framework
- MySQL for data (users, Pins, boards)
- Memcached for caching
- S3 for image storage
- EC2 instances

**Scaling Challenges** (2011-2012):
- MySQL read replicas maxed out
- Database sharding needed
- Image processing bottlenecks
- Monolith deployment slow and risky

---

### Phase 2: Sharded MySQL + Service Extraction (2012-2016)

Pinterest sharded MySQL and extracted key services.

**Database Sharding**:
- Shard by user_id (all user's data on same shard)
- 100s of MySQL shards
- Avoids cross-shard queries for most operations

**Service Extraction**:
- **Pin Service**: Store and serve Pins
- **Board Service**: Manage boards
- **User Service**: User profiles, follows
- **Feed Service**: Home feed generation
- **Search Service**: Elasticsearch-based search

**Why Shard by user_id?**
- Most queries are user-specific ("Get user X's Pins", "Get user X's boards")
- Keeps related data together (data locality)
- Simplifies application logic (no distributed joins)

---

### Phase 3: HBase + Microservices (2016-present)

Pinterest migrated to HBase for hot path and expanded microservices.

**HBase Adoption**:
- Replace MySQL for high-throughput use cases
- Store user feeds, Pin saves, graph edges
- Horizontal scalability (add nodes for capacity)
- Better performance for sequential scans

**Microservices**:
- 100s of microservices
- Independent scaling and deployment
- Polyglot (Python, Java, Go, Elixir)

---

## Core Components

### 1. Image Storage and Delivery Pipeline

Pinterest's core asset is images. Handling billions of images requires sophisticated pipeline.

**Image Upload Flow**:

\`\`\`
1. User uploads image or provides URL:
   - Direct upload: Client → Pinterest API
   - URL scraping: Pinterest → External website → Download image

2. Image Validation:
   - File type (JPEG, PNG, GIF, WebP)
   - Size limits (max 32 MB)
   - Malware scan
   - Content moderation (detect inappropriate content via ML)

3. Image Processing:
   - Generate multiple sizes:
     * Thumbnail: 236x (for grid)
     * Medium: 564x (for modal)
     * Full size: Original (max 2048x)
   - Optimize compression (reduce file size by 40-60%)
   - Extract dominant colors (for UI theming)
   - Generate blurhash (placeholder while loading)

4. Storage:
   - Upload to S3 (multiple sizes)
   - Organized: s3://pinterest-images/{pin_id}/original.jpg
   - s3://pinterest-images/{pin_id}/236x.jpg
   - s3://pinterest-images/{pin_id}/564x.jpg

5. CDN Distribution:
   - Images served via CloudFront (AWS CDN)
   - Edge locations worldwide (100+ PoPs)
   - Cache hit rate: 95%+
   - Lazy loading on client (progressive image loading)

6. Metadata Storage:
   - Store Pin metadata in HBase/MySQL:
     * pin_id, user_id, board_id
     * title, description, link
     * image_urls (different sizes)
     * creation_timestamp
     * engagement_stats (saves, clicks)
\`\`\`

**Image Processing Pipeline**:

Pinterest uses **Apache Kafka** and distributed workers:

\`\`\`
Image Upload → Kafka Topic: image_processing_queue
                    ↓
             Workers (EC2 instances)
                    ↓
          Process image (resize, optimize)
                    ↓
          Upload to S3
                    ↓
          Update metadata (HBase/MySQL)
                    ↓
          Publish event: image_processed
\`\`\`

**Challenges**:

**1. Scale**: 100,000+ image uploads per minute during peak

**Solution**:
- Horizontal scaling (100s of processing workers)
- Auto-scaling based on Kafka queue depth
- Prioritize: High-value users, viral Pins processed first

**2. Storage Cost**: Petabytes of images = millions in S3 costs

**Solution**:
- Compression (reduce file size)
- Tiered storage: Hot (S3 Standard) vs Cold (S3 Glacier for old, rarely accessed images)
- Deduplication: Hash images, store once if duplicate (save 10-15% storage)

**3. Latency**: Images must load <500ms globally

**Solution**:
- CDN with 100+ edge locations
- Pre-fetch next images (predict user scrolling)
- Progressive image loading (show low-res first, then high-res)

---

### 2. Pinterest Graph Database

Pinterest's data is inherently graph-based: Users follow boards, save Pins, create boards.

**Graph Structure**:

\`\`\`
Nodes:
- Users (450M+)
- Pins (240B+)
- Boards (5B+)
- Topics/Interests (1M+ topics)

Edges:
- User → Follows → User, Board, Topic
- User → Creates → Board, Pin
- User → Saves → Pin to Board
- User → Likes/Clicks → Pin
- Pin → Belongs_to → Board
- Pin → Has_topic → Topic
\`\`\`

**Storage: HBase**

Pinterest uses HBase for storing graph edges (high write throughput, sequential scans).

**Data Model** (HBase):

\`\`\`
Table: user_saves
Row Key: user_id
Column Family: saves
Column Qualifiers: pin_id_1, pin_id_2, ..., pin_id_10000
Values: timestamp of save

Query: "Get all Pins saved by user X"
→ Single row scan, return all columns
→ O(1) for row lookup, O(n) for scanning columns (fast because columns in same row)

Table: pin_boards
Row Key: pin_id
Column Family: boards
Column Qualifiers: board_id_1, board_id_2, ...
Values: count (how many times saved to this board)

Query: "Which boards contain Pin Y?"
→ Single row scan

Table: board_pins
Row Key: board_id
Column Family: pins
Column Qualifiers: pin_id_1, pin_id_2, ...
Values: save_timestamp

Query: "Get all Pins in board Z"
→ Single row scan, sorted by timestamp
\`\`\`

**Why HBase?**

**Advantages**:
- High write throughput (millions of saves per minute)
- Horizontal scalability (add RegionServers for capacity)
- Sequential scans are fast (row-based storage)
- No schema changes needed (add columns dynamically)

**Disadvantages**:
- Complex operations (joins) require application logic
- No transactions across rows
- Operational complexity

**Alternative Considered**: MySQL

- Advantages: ACID transactions, joins, familiar
- Disadvantages: Hard to scale writes, sharding complex for graphs

**Pinterest's Approach**: Hybrid
- **HBase**: Hot path (saves, follows, feeds) - high throughput
- **MySQL**: Cold path (analytics, reporting, user profiles) - complex queries

---

### 3. Home Feed (Smart Feed)

Pinterest's Smart Feed shows personalized Pins to users.

**Feed Goals**:
- **Relevance**: Show Pins user will engage with (save, click)
- **Freshness**: Mix new and old content
- **Diversity**: Multiple interests, sources, formats
- **Discovery**: Introduce new topics user might like

**Feed Generation Architecture**:

**Two-Stage Approach**:

**Stage 1: Candidate Generation**

Generate 1000-5000 candidate Pins:

\`\`\`python
def generate_candidates(user_id):
    candidates = []
    
    # 1. Pins from followed boards (30% weight)
    followed_boards = get_followed_boards(user_id)
    for board in followed_boards:
        recent_pins = get_recent_pins(board, limit=50)
        candidates.extend(recent_pins)
    
    # 2. Pins similar to past saves (40% weight)
    saved_pins = get_recent_saves(user_id, limit=100)
    for pin in saved_pins:
        similar_pins = find_similar_pins(pin, limit=20)
        candidates.extend(similar_pins)
    
    # 3. Trending Pins in user's interests (20% weight)
    interests = get_user_interests(user_id)
    for interest in interests:
        trending_pins = get_trending_pins(interest, limit=50)
        candidates.extend(trending_pins)
    
    # 4. Explore (10% weight) - new topics
    explore_pins = get_random_high_quality_pins(limit=100)
    candidates.extend(explore_pins)
    
    # Deduplicate
    candidates = deduplicate(candidates)
    
    return candidates[:5000]  # Top 5000 candidates
\`\`\`

**Stage 2: Ranking**

Rank candidates using ML model:

\`\`\`
For each (user, pin) pair:
1. Extract features:
   - User features: Interests, past saves, engagement patterns
   - Pin features: Image quality, description, topic, age, virality
   - Context features: Time of day, device, session history

2. ML model predicts engagement probability:
   - P(save | user, pin)
   - P(click | user, pin)
   - P(close-up | user, pin) - zooming in to see details

3. Combine probabilities:
   score = 0.5 * P(save) + 0.3 * P(click) + 0.2 * P(close-up)

4. Sort candidates by score (descending)

5. Apply diversity:
   - Don't show 10 similar Pins in a row
   - Mix topics, sources, formats

6. Return top 100 Pins
\`\`\`

**ML Ranking Model**:

**Model**: Gradient Boosted Decision Trees (XGBoost) + Deep Neural Network (TensorFlow)

**Features** (1000+ features):

**User Features**:
- User interests (extracted from saves, clicks, searches)
- Engagement patterns (time of day, device, session length)
- Demographics (age, gender, location)
- Account age, activity level

**Pin Features**:
- Image quality score (resolution, composition, lighting)
- Description (NLP sentiment, keyword extraction)
- Topic classification (Home Decor, Fashion, Food, etc.)
- Pin age (recency)
- Virality (save rate, click-through rate)
- Creator quality (popular Pinner vs new user)

**User-Pin Features**:
- Topic match (user interested in Fashion, Pin is Fashion)
- Past interactions (user saved similar Pins)
- Social proof (friends saved this Pin)

**Context Features**:
- Time of day (morning, evening)
- Day of week (weekend behavior different)
- Device (mobile, desktop, tablet)
- Session position (1st Pin in session vs 50th)

**Training**:

\`\`\`
Data: Billions of (user, pin, engagement) tuples
- Positive examples: User saved/clicked Pin
- Negative examples: User saw Pin but didn't engage

Model: XGBoost (tree-based) or DNN (neural network)

Loss: Binary cross-entropy (classification)

Training pipeline:
1. Kafka streams user events (impressions, saves, clicks)
2. Spark jobs process events, extract features
3. Store in HDFS (data lake)
4. Train model on historical data (last 30 days)
5. Export model (PMML or SavedModel format)
6. Deploy to scoring service (gRPC)
7. Feed service calls scoring service for inference
8. Retrain model daily with new data
\`\`\`

**Inference**:

\`\`\`
Scoring Service (deployed on 100s of instances):
- Loads model into memory
- Receives batch requests: (user_id, [pin_id_1, ..., pin_id_1000])
- Extracts features (cached in Redis)
- Runs model inference (GPU-accelerated if DNN)
- Returns scores: [0.85, 0.62, 0.91, ...]
- Latency: <50ms for batch of 1000 Pins
\`\`\`

---

### 4. Visual Search (Pinterest Lens)

Pinterest Lens allows users to search using images: Take photo → Find similar Pins.

**Visual Search Challenges**:
- Find visually similar images from billions of Pins
- Sub-second latency
- Handle variations (angle, lighting, occlusion, cropping)

**Technology: Convolutional Neural Networks (CNNs)**

**Architecture**:

**1. Image Embedding**:

Use CNN to convert images to vectors (embeddings):

\`\`\`
Input: Image (RGB, 224x224 pixels)
       ↓
CNN Model (ResNet50 or EfficientNet)
       ↓
Output: Embedding vector (256 dimensions)

Embeddings capture visual features:
- Low-level: Edges, colors, textures
- High-level: Objects, patterns, styles

Similar images → Similar embeddings (cosine similarity)
\`\`\`

**Training**:

\`\`\`
Dataset: Billions of Pinterest images
Labels: User engagement (Pins saved together likely similar)

Training approach: Triplet Loss
- Anchor: Image A
- Positive: Visually similar image A+ (same product, different angle)
- Negative: Different image A-

Loss: ||emb(A) - emb(A+)||² - ||emb(A) - emb(A-)||² + margin

Goal: Make similar images closer in embedding space
\`\`\`

**2. Indexing**:

Store all Pin embeddings in vector index:

\`\`\`
For each Pin:
- Generate embedding (256-d vector)
- Store in vector index: pin_id → embedding

Index type: Faiss (Facebook AI Similarity Search)
- Supports billion-scale search
- GPU-accelerated
- Approximate Nearest Neighbor (ANN) search
\`\`\`

**Faiss Index**:

\`\`\`
Index: IndexIVFPQ (Inverted File with Product Quantization)
- Divide space into clusters (e.g., 100K clusters)
- Each embedding assigned to nearest cluster
- Within cluster, use product quantization (compress 256-d to 64 bytes)

Query:
1. Find k nearest clusters (e.g., 100 clusters)
2. Search within these clusters (ANN)
3. Return top 1000 nearest neighbors
4. Latency: <100ms for billion-scale index
\`\`\`

**3. Search Flow**:

\`\`\`
User takes photo (or crops Pin image):
1. Client sends image to backend
2. Resize to 224x224
3. CNN generates embedding (GPU inference, <50ms)
4. Query Faiss index for K nearest neighbors (K=1000)
5. Retrieve Pin metadata for top 1000 results (batch query to HBase)
6. Rank by relevance (ML model considers user interests)
7. Return top 100 results
\`\`\`

**4. Multi-Object Detection**:

User can select specific object within image:

\`\`\`
Image with multiple objects (lamp, table, chair):
1. Object detection model (YOLO or Mask R-CNN)
   - Detects bounding boxes: [lamp, table, chair]
2. User taps on lamp
3. Crop image to lamp bounding box
4. Generate embedding for cropped image
5. Search for similar lamps
\`\`\`

**Visual Search Variations**:

**Shop the Look**:
- Detect fashion items in image (dress, shoes, bag)
- Find shoppable products for each item
- Link to merchant websites

**Try On**:
- Virtual try-on using AR (augmented reality)
- Overlay product on user's image

**Similar Products**:
- Search for products similar to one in image
- E-commerce integration

---

### 5. Search

Pinterest search allows users to find Pins by keywords.

**Search Index**: Elasticsearch

**Index Structure**:

\`\`\`json
{
  "pin_id": "12345",
  "title": "DIY Home Office Desk",
  "description": "Build your own modern desk with reclaimed wood...",
  "board_id": "67890",
  "board_name": "Home Office Ideas",
  "user_id": "11111",
  "creator_name": "John Doe",
  "topics": ["home-decor", "diy", "furniture", "office"],
  "image_url": "https://...",
  "image_dominant_colors": ["#FFFFFF", "#8B4513"],
  "save_count": 5420,
  "click_count": 12350,
  "created_at": "2024-01-15T10:30:00Z",
  "engagement_score": 0.85,
  "is_verified": true  // verified creator
}
\`\`\`

**Search Query Flow**:

\`\`\`
User searches: "home office desk DIY"

1. Query parsing:
   - Tokenize: ["home", "office", "desk", "diy"]
   - Expand: Add synonyms (desk → table, workspace)
   - Detect intent: User wants DIY tutorials, not products

2. Elasticsearch query:
   {
     "query": {
       "multi_match": {
         "query": "home office desk DIY",
         "fields": ["title^3", "description^2", "topics^2", "board_name"]
         // ^3 means 3x weight for title matches
       }
     },
     "filter": {
       "range": {
         "engagement_score": {"gte": 0.5}  // Filter low-quality
       }
     },
     "size": 1000
   }

3. Candidate retrieval: Top 1000 Pins matching query

4. Personalized ranking:
   - ML model scores each Pin for this user
   - Features: Query relevance, user interests, Pin quality
   - Re-rank candidates

5. Diversity:
   - Mix: Tutorials, product Pins, inspiration
   - Avoid: 10 identical Pins in a row

6. Return top 50 results (pagination)
\`\`\`

**Search Ranking Model**:

\`\`\`
Features:
- Query-Pin relevance (BM25 score from Elasticsearch)
- User-Pin relevance (user interested in topic?)
- Pin quality (engagement rate, image quality)
- Pin freshness (recent Pins boosted)
- Diversity (topic, creator, format)

Model: LambdaMART (Learning to Rank)
Training: Billions of (query, pin, click) tuples
Objective: Maximize clicks, saves on search results
\`\`\`

**Autocomplete**:

As user types, suggest query completions:

\`\`\`
User types: "home off"

Autocomplete service:
1. Prefix matching in trie data structure
2. Return top 10 suggestions:
   - "home office" (most popular)
   - "home office ideas"
   - "home office decor"
   - "home office desk"
3. Personalized: Boost suggestions matching user's interests
4. Trending: Boost queries spiking in popularity
\`\`\`

---

### 6. Recommendations

Pinterest recommendations power multiple surfaces:

**1. Related Pins** (Below Pin modal):
- Show Pins similar to currently viewed Pin
- Collaborative filtering: Users who saved X also saved Y
- Content-based: Pins with similar embeddings (visual similarity)

**2. Board Recommendations**:
- Suggest boards user might want to follow
- Based on: User's interests, followed boards, similar users

**3. Shopping Recommendations**:
- Personalized product recommendations
- Based on: Past saves, searches, browsing history
- E-commerce integration (Shopify, BigCommerce)

**Recommendation Pipeline**:

\`\`\`
Offline (Batch):
1. Spark jobs compute recommendations nightly
2. Collaborative filtering (matrix factorization)
   - User-Pin matrix (sparse, billions of entries)
   - Factorize into user embeddings × Pin embeddings
   - Predict: user_embedding · pin_embedding = score
3. Store top 1000 recommendations per user in HBase
   - Key: user_id → Value: [pin_id_1, pin_id_2, ..., pin_id_1000]

Online (Real-time):
1. User requests recommendations
2. Fetch pre-computed candidates from HBase
3. Re-rank with real-time model (considers recent activity)
4. Return top 50
\`\`\`

---

### 7. Spam and Content Moderation

Pinterest must moderate billions of Pins for spam, policy violations, inappropriate content.

**Challenges**:
- 100,000+ Pins uploaded per minute
- Impossible to manually review all content
- Need automated systems

**ML-Based Moderation**:

**1. Image Classification**:

\`\`\`
CNN models classify images:
- Safe for work vs NSFW (nudity, violence)
- Spam (low-quality, misleading)
- Policy violations (hate symbols, dangerous content)

Models trained on millions of labeled examples
Precision: 95%+ (few false positives)
Recall: 90%+ (catch most violations)
\`\`\`

**2. Text Classification**:

\`\`\`
NLP models classify Pin descriptions:
- Spam keywords
- Prohibited content (weapons, drugs)
- Clickbait

Models: BERT-based transformers
\`\`\`

**3. Human Review**:

\`\`\`
Automated system flags suspicious Pins
Human moderators review flagged content
Feedback loop improves ML models
\`\`\`

**Actions**:
- Remove: Delete Pin immediately
- Demote: Lower in search/feed ranking
- Warn: Notify creator, allow edit
- No action: False positive, approved

---

## Technology Stack

### Backend

- **Python**: Primary language (Django originally, now microservices)
- **Java**: High-performance services (search, recommendations)
- **Go**: Real-time services (notifications, WebSocket)
- **Elixir**: Some services (concurrency, fault tolerance)

### Data Storage

- **HBase**: Graph data, user feeds, high-throughput writes
- **MySQL**: User profiles, boards, sharded by user_id
- **Redis**: Caching, rate limiting, session storage
- **Memcached**: Additional caching layer
- **S3**: Image storage (petabytes)
- **Elasticsearch**: Search index (Pins, users, boards)

### Data Processing

- **Apache Kafka**: Event streaming (saves, clicks, searches)
- **Apache Spark**: Batch processing (recommendations, analytics)
- **Apache Flink**: Stream processing (real-time metrics)
- **Airflow**: Workflow orchestration (data pipelines)

### Machine Learning

- **TensorFlow**: Deep learning (image embeddings, ranking)
- **PyTorch**: Research, experimentation
- **XGBoost**: Gradient boosted trees (ranking, classification)
- **Faiss**: Vector similarity search (visual search)

### Infrastructure

- **AWS**: Primary cloud provider (EC2, S3, RDS, etc.)
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code
- **Spinnaker**: Deployment platform

### Monitoring

- **Datadog**: Metrics, logs, APM
- **PagerDuty**: On-call, alerting
- **Sentry**: Error tracking

---

## Key Lessons from Pinterest Architecture

### 1. Visual Search Requires Specialized Infrastructure

CNNs for embeddings + Faiss for billion-scale vector search enable sub-second visual search. Off-the-shelf solutions inadequate.

### 2. Graph Data Benefits from Column-Oriented Storage

HBase's column-family model perfect for graph edges (user saves, follows). Single-row scans fast for "get all saves for user."

### 3. Two-Stage Ranking Scales

Candidate generation (retrieve 1000s) + ML ranking (re-rank) balances recall and precision. Can't run expensive ML model on billions of Pins.

### 4. Hybrid Storage Strategy

HBase for hot path (high throughput), MySQL for cold path (complex queries). No single database perfect for all use cases.

### 5. Image Processing Pipeline Critical

Multiple image sizes, compression, CDN distribution ensure fast loading. Pinterest is image-first platform - images must load instantly.

---

## Interview Tips

**Q: How would you implement Pinterest's visual search?**

A: Use CNNs to generate image embeddings. Train ResNet50 or EfficientNet on billions of Pinterest images using triplet loss: make similar images (saved together) closer in embedding space. Convert each Pin image to 256-dimensional vector. Store embeddings in Faiss (Facebook AI Similarity Search) - billion-scale vector index with GPU acceleration. User uploads query image: (1) Generate embedding (CNN inference, <50ms). (2) Query Faiss for K-nearest neighbors (K=1000, <100ms). (3) Retrieve Pin metadata (batch query HBase). (4) Rank by relevance (ML model considers user interests). (5) Return top 100. Handle multi-object: Use object detection (YOLO/Mask R-CNN) to identify objects, let user select, generate embedding for cropped region.

**Q: How does Pinterest generate personalized home feeds?**

A: Two-stage approach. Stage 1 - Candidate generation: Generate 1000-5000 candidates from multiple sources: (1) Pins from followed boards (30%). (2) Pins similar to past saves (collaborative filtering, 40%). (3) Trending Pins in user's interests (20%). (4) Explore - new topics (10%). Deduplicate. Stage 2 - Ranking: ML model (XGBoost or DNN) predicts engagement probability P(save | user, pin). Features: User (interests, demographics), Pin (quality, virality, topic), Context (time, device). Combine probabilities: score = 0.5*P(save) + 0.3*P(click) + 0.2*P(close-up). Sort by score. Apply diversity (mix topics, sources). Return top 100. Pre-compute candidates offline (Spark batch jobs), real-time ranking online (scoring service <50ms latency).

**Q: Why does Pinterest use HBase for graph data?**

A: HBase's column-family model suits graph edges. Store user saves: Row key = user_id, Columns = pin_id_1, pin_id_2, ..., pin_id_10000. Query "get all saves for user X" = single row scan (fast, data locality). High write throughput: Millions of saves per minute, HBase handles easily. Horizontal scalability: Add RegionServers for capacity. Sequential scans fast (row-oriented storage). Alternative MySQL requires: (user_id, pin_id, timestamp) table, indexed by user_id, but millions of small rows vs HBase's single row with many columns. HBase wins for high-throughput, sequential-scan workloads. Trade-off: Complex queries (joins, aggregations) harder in HBase, use MySQL for analytics.

---

## Summary

Pinterest's architecture demonstrates building a visual discovery platform at massive scale:

**Key Takeaways**:

1. **Image pipeline**: S3 storage, multiple sizes, CDN (CloudFront), progressive loading, compression
2. **Visual search**: CNN embeddings (ResNet50), Faiss for billion-scale vector search, object detection
3. **Graph storage**: HBase for user saves/follows (column-family model, high throughput)
4. **Personalized feed**: Two-stage (candidate generation + ML ranking), offline batch + online real-time
5. **Search**: Elasticsearch with LambdaMART ranking, autocomplete, personalization
6. **Recommendations**: Collaborative filtering (matrix factorization), content-based (embeddings)
7. **ML infrastructure**: TensorFlow/PyTorch for training, XGBoost for production, Kafka+Spark pipeline
8. **Content moderation**: CNN classifiers for images, NLP for text, human review for edge cases

Pinterest's success from visual-first design, ML-powered recommendations, and specialized infrastructure (Faiss for visual search, HBase for graphs).
`,
};
