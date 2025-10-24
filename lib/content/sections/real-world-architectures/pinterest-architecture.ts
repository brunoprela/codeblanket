/**
 * Pinterest Architecture Section
 */

export const pinterestarchitectureSection = {
    id: 'pinterest-architecture',
    title: 'Pinterest Architecture',
    content: `Pinterest is a visual discovery platform where users find and save ideas through "Pins" organized on boards. With 450+ million monthly active users and billions of Pins, Pinterest's architecture must handle massive image storage, recommendations, and visual search at scale.

## Overview

**Scale**: 450M+ users, 240 billion Pins, 5 billion boards, 2+ billion searches/month

**Key Challenges**: Image storage and delivery, visual search, recommendations, graph processing, real-time personalization

## Core Components

### 1. Pin Storage and Image Pipeline

**Image Storage**:
- S3 for raw images
- Multiple sizes generated (thumbnail, medium, full)
- CDN (CloudFront) for delivery
- Cache hit rate: 95%+

**Upload Flow**:
\`\`\`
1. User uploads image or provides URL
2. Download image, validate
3. Generate thumbnails (5 sizes)
4. Upload to S3
5. Store metadata in HBase/MySQL
6. Index in Elasticsearch for search
7. Generate image embeddings for visual search (CNN)
\`\`\`

---

### 2. Graph Database (Pinterest Graph)

Pinterest's data is inherently graph-based.

**Graph Structure**:
\`\`\`
Users → Follow → Users, Boards
Users → Create → Boards, Pins
Users → Save (Repin) → Pins to Boards
Users → Like, Comment → Pins
\`\`\`

**Storage**:
- **MySQL**: Sharded for user data, boards, pins
- **HBase**: For graph edges (user follows, saves)
- **Redis**: Caching hot data

**Data Model** (HBase):
\`\`\`
Table: user_saves
Row Key: user_id
Column Family: saves
Columns: pin_id_1, pin_id_2, ..., pin_id_1000
Values: timestamp

Query: "Get all pins saved by user X"
→ Single row scan (fast)
\`\`\`

---

### 3. Home Feed (Smart Feed)

Pinterest's Smart Feed shows personalized Pins.

**Architecture**:
- **Candidate Generation**: Fetch Pins from followed boards, related Pins
- **Ranking**: ML model predicts engagement (save, click, close-up)
- **Diversity**: Mix interests, formats (image, video), sources

**ML Ranking Model**:
\`\`\`
Features:
- Pin features: Image quality, description, category, age
- User features: Interests, past saves, search history
- Context: Time of day, device, session

Model: Deep neural network (TensorFlow)
Objective: Predict P(engagement | user, pin)
Training: Billions of impressions and engagements
\`\`\`

---

### 4. Visual Search (Lens)

Pinterest Lens allows visual search: Take photo → Find similar Pins.

**Technology**:
- **Convolutional Neural Networks (CNNs)**: Extract image features
- **Embeddings**: 256-dimensional vectors representing images
- **Approximate Nearest Neighbor**: Find similar embeddings quickly

**Pipeline**:
\`\`\`
1. User takes photo or crops Pin image
2. CNN extracts embedding (256-d vector)
3. Search index (Faiss) for nearest neighbors
4. Return top 100 similar Pins
5. Rank by relevance, quality
\`\`\`

**Faiss** (Facebook AI Similarity Search):
- Billion-scale vector search
- GPU-accelerated
- Sub-100ms latency

---

### 5. Recommendations

**Related Pins**:
- Collaborative filtering: Users who saved X also saved Y
- Content-based: Pins with similar embeddings
- Graph-based: Pins on same board as X

**Personalized Recommendations**:
- Interest graph: User's interests extracted from saves, searches
- ML model predicts relevance
- Explore vs exploit (show new interests)

---

### 6. Real-Time Analytics

**Pinterest uses Apache Kafka and Apache Storm/Flink**:
- Stream user events (saves, clicks, searches)
- Real-time metrics (trending Pins, popular searches)
- Fraud detection (spam Pins, bots)

---

## Technology Stack

**Backend**: Python (main), Java (some services), Go (performance-critical)
**Data**: MySQL (sharded), HBase, Redis, Memcached, S3
**Search**: Elasticsearch (text), Faiss (visual)
**ML**: TensorFlow, PyTorch
**Data Processing**: Kafka, Spark, Flink
**Infrastructure**: AWS, Kubernetes

---

## Key Lessons

1. **Visual search** requires CNN embeddings + vector search (Faiss)
2. **Graph data** stored in HBase for fast edge traversals
3. **Image pipeline** optimized: Multiple sizes, CDN, aggressive caching
4. **ML recommendations** drive engagement (Smart Feed, Related Pins)

---

## Interview Tips

**Q: How would you implement Pinterest's visual search?**

A: Use CNNs to extract image embeddings. Train CNN (ResNet50 or similar) on Pinterest's dataset to generate 256-dimensional vectors. For each Pin, store embedding in vector index (Faiss). When user uploads query image: (1) Extract embedding using same CNN. (2) Search Faiss index for K nearest neighbors (cosine similarity). (3) Return top 100 similar Pins. (4) Rank by quality score, user relevance. Use GPU for CNN inference (<50ms). Faiss supports billion-scale search with sub-100ms latency. Pre-compute embeddings for all Pins (batch job nightly). Update index incrementally for new Pins.

---

## Summary

Pinterest's architecture handles visual discovery at scale: image storage and delivery via S3/CDN, graph database (HBase) for connections, visual search with CNNs and Faiss, ML-powered recommendations for Smart Feed. Success comes from optimized image pipeline, vector search for visual similarity, and personalized ranking models.
`,
};

