/**
 * Design YouTube Section
 */

export const youtubeSection = {
    id: 'youtube',
    title: 'Design YouTube',
    content: `YouTube is the world's largest video sharing platform with 2+ billion users, 500 hours of video uploaded every minute, and billions of views per day. Unlike Netflix (curated content), YouTube handles user-generated content with massive upload volume, real-time processing, and complex recommendation algorithms.

## Problem Statement

Design YouTube with:
- **Video upload**: Users upload videos (up to 12 hours, 256 GB)
- **Video processing**: Transcode, compress, generate thumbnails
- **Video streaming**: Adaptive bitrate streaming globally via CDN
- **Search & discovery**: Find videos by title, tags, description
- **Recommendations**: Personalized homepage and suggestions
- **Comments & engagement**: Likes, comments, subscriptions
- **Live streaming**: Real-time video broadcasts
- **Analytics**: View counts, watch time, revenue tracking

**Scale**: 2B users, 500M hours uploaded/day, 1B hours watched/day

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Upload Video**: Users upload videos (mp4, mov, avi, etc.)
2. **Process Video**: Transcode to multiple formats and bitrates
3. **Watch Video**: Stream with adaptive bitrate
4. **Search**: Find videos by keywords, tags, channels
5. **Recommendations**: Personalized video suggestions
6. **Engage**: Like, comment, subscribe, share
7. **Analytics**: Track views, watch time, revenue
8. **Live Stream**: Broadcast live video to viewers
9. **Monetization**: Ads, super chat, memberships

### Non-Functional Requirements

1. **High Availability**: 99.99% uptime
2. **Low Latency**: < 2sec video start time, < 1sec for 240p
3. **Scalable**: Handle 500M hours uploaded/day, 1B hours watched/day
4. **Consistency**: Eventual consistency acceptable for view counts, likes
5. **Global**: Low latency worldwide (CDN in 100+ countries)
6. **Cost-Effective**: Optimize encoding and bandwidth costs
7. **Copyright Protection**: Content ID for copyright detection

---

## Step 2: Capacity Estimation

### Upload Traffic

**Videos uploaded**:
- 500 hours of video uploaded per minute
- 500 × 60 = 30,000 hours/hour = 720,000 hours/day
- Average video length: 10 minutes
- 720K hours × 6 videos/hour = 4.3 million videos/day
- Peak: ~6 million videos/day

**Upload bandwidth**:
- Average raw video: 500 MB per minute (1080p @ 50 Mbps)
- 500 hours/min × 60 min × 500 MB = 15,000 GB/min = 250 GB/sec = 2 Tbps upload bandwidth

### Watch Traffic

**Views**:
- 1 billion hours watched per day
- Average watch duration: 40 minutes (users skip through videos)
- 1B hours × 60/40 = 1.5B video views/day = ~17,000 views/sec
- Peak: ~40,000 views/sec

**Download bandwidth**:
- Average bitrate: 2.5 Mbps (mix of 144p mobile to 4K desktop)
- 1B hours/day × 2.5 Mbps = 2.5 PB/day
- Peak: 40K concurrent viewers × 2.5 Mbps = 100 Gbps × 10K edges = 1 Pbps (petabit per second!)

### Storage

**Video storage**:
- 4.3M videos/day × 10 min avg × 500 MB/min = 21.5 PB/day raw uploads
- After encoding (5 versions): 21.5 PB × 5 = 107 PB/day
- With compression and deduplication: ~50 PB/day
- 1 year: 50 PB × 365 = 18 exabytes (EB)

**Metadata storage**:
- 4.3M videos/day × 5 KB metadata = 21.5 GB/day
- 1 year: 7.8 TB metadata

---

## Step 3: System API Design

### Upload Video

\`\`\`
POST /api/v1/videos/upload
Content-Type: multipart/form-data

Request:
- video: <binary file>
- title: "My First YouTube Video"
- description: "..."
- tags: ["vlog", "daily life"]
- visibility: "public" | "unlisted" | "private"

Response (202 Accepted):
{
  "video_id": "dQw4w9WgXcQ",
  "upload_url": "https://upload.youtube.com/session/abc123",
  "status": "uploading",
  "estimated_processing_time": "10 minutes"
}
\`\`\`

### Resumable Upload (Critical for Large Files)

\`\`\`
1. POST /api/v1/videos/upload/init
   → Returns upload_session_id and upload_url
   
2. PUT {upload_url}
   Content-Range: bytes 0-524287/1048576
   → Upload first 512 KB chunk
   
3. PUT {upload_url}
   Content-Range: bytes 524288-1048575/1048576
   → Upload next 512 KB chunk
   
4. Server responds with 308 (Resume Incomplete) until complete
   Final response: 200 OK with video_id
\`\`\`

### Watch Video

\`\`\`
GET /api/v1/videos/{video_id}

Response:
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "channel": {
    "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
    "name": "Rick Astley",
    "subscribers": "3.2M"
  },
  "manifest_url": "https://youtube.com/manifest/dQw4w9WgXcQ.m3u8",
  "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
  "duration": 212,
  "views": 1400000000,
  "likes": 15000000,
  "published_at": "2009-10-25T06:57:33Z",
  "description": "...",
  "tags": ["music", "80s", "pop"]
}
\`\`\`

### Search Videos

\`\`\`
GET /api/v1/search?q=machine+learning&limit=20&page=1

Response:
{
  "results": [
    {
      "video_id": "abc123",
      "title": "Machine Learning Tutorial for Beginners",
      "thumbnail": "url",
      "channel": "Tech Academy",
      "views": "2.5M",
      "published_at": "2024-01-15"
    }
  ],
  "total_results": 15000000,
  "next_page_token": "xyz789"
}
\`\`\`

---

## Step 4: Database Schema

### Videos Table (PostgreSQL/Spanner)

\`\`\`sql
CREATE TABLE videos (
    video_id VARCHAR(11) PRIMARY KEY,  -- Base64 encoded (dQw4w9WgXcQ)
    user_id BIGINT NOT NULL,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    duration INT,  -- seconds
    manifest_url VARCHAR(500),
    thumbnail_urls JSON,  -- {default, medium, high, maxres}
    status VARCHAR(20),  -- uploading, processing, ready, failed
    visibility VARCHAR(10),  -- public, unlisted, private
    created_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP,
    view_count BIGINT DEFAULT 0,
    like_count INT DEFAULT 0,
    dislike_count INT DEFAULT 0,
    comment_count INT DEFAULT 0,
    INDEX idx_user_created (user_id, created_at),
    INDEX idx_published (published_at)
);
\`\`\`

### Watch History & Analytics (BigQuery/Cassandra)

\`\`\`sql
-- Time-series data for analytics
CREATE TABLE video_events (
    video_id VARCHAR(11),
    event_type VARCHAR(20),  -- view, like, share, watch_time
    user_id BIGINT,
    timestamp TIMESTAMP,
    metadata JSON,  -- {watch_duration, referrer, device_type}
    PRIMARY KEY (video_id, timestamp, user_id)
) PARTITION BY RANGE (timestamp);
\`\`\`

### Comments Table (Cassandra)

\`\`\`cql
CREATE TABLE comments (
    video_id TEXT,
    comment_id TIMEUUID,
    user_id BIGINT,
    text TEXT,
    parent_comment_id TIMEUUID,  -- For nested replies
    created_at TIMESTAMP,
    like_count INT,
    PRIMARY KEY (video_id, comment_id)
) WITH CLUSTERING ORDER BY (comment_id DESC);
\`\`\`

### Channels & Subscriptions

\`\`\`sql
CREATE TABLE channels (
    channel_id VARCHAR(20) PRIMARY KEY,
    user_id BIGINT UNIQUE,
    name VARCHAR(100),
    description TEXT,
    subscriber_count BIGINT DEFAULT 0,
    created_at TIMESTAMP
);

CREATE TABLE subscriptions (
    subscriber_id BIGINT,
    channel_id VARCHAR(20),
    subscribed_at TIMESTAMP,
    PRIMARY KEY (subscriber_id, channel_id)
);
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
      ▼                      ▼                      ▼
┌───────────┐        ┌───────────┐         ┌───────────┐
│  Upload   │        │   Watch   │         │  Search   │
│  Service  │        │  Service  │         │  Service  │
└─────┬─────┘        └─────┬─────┘         └─────┬─────┘
      │                    │                      │
      │                    │                      ▼
      │                    │              ┌──────────────┐
      │                    │              │Elasticsearch │
      │                    │              │  (Search)    │
      │                    │              └──────────────┘
      │                    │
      ▼                    ▼
┌────────────────┐  ┌───────────────┐
│ Google Cloud   │  │  CDN Network  │
│  Storage (GCS) │  │  (YouTube CDN)│
│  Raw uploads   │  │  300+ edges   │
└───────┬────────┘  └───────────────┘
        │
        ▼
┌────────────────────────────┐
│  Video Processing Pipeline │
│  (Transcoding, Thumbnails) │
└────────────────┬───────────┘
                 │
                 ▼
┌────────────────────────────┐
│   Encoded Video Storage    │
│   (GCS - Multi-region)     │
└────────────────────────────┘

        ┌────────────────┐
        │  Cloud Pub/Sub │
        │  (Event Queue) │
        └────────┬───────┘
                 │
         ┌───────┴────────┬──────────────┐
         ▼                ▼              ▼
    ┌─────────┐     ┌──────────┐  ┌──────────┐
    │Analytics│     │Copyright │  │  Recs    │
    │ Service │     │   ID     │  │ Service  │
    └─────────┘     └──────────┘  └──────────┘
\`\`\`

---

## Step 6: Video Upload Flow (Detailed)

**Step-by-Step Process**:

\`\`\`
1. USER UPLOADS VIDEO (Web/Mobile App)
   - User selects 1.5 GB video file (10 min, 1080p60)
   - App requests: POST /api/v1/videos/upload/init
   - Server generates upload_session_id and pre-signed GCS URL
   - Response: {upload_url, session_id, chunk_size: 10MB}

2. CHUNKED UPLOAD (Resumable)
   - App splits video into 150 chunks (1.5 GB / 10 MB)
   - Uploads chunks in parallel (5 concurrent)
   - Each chunk: PUT {upload_url}?chunk=1 with Content-Range header
   - Server tracks progress: Redis key "upload:session_123" = {chunks_uploaded: 45/150}
   - If network fails, app resumes from last successful chunk
   
3. UPLOAD COMPLETE
   - Final chunk upload returns 200 OK
   - Server writes metadata to database:
     INSERT INTO videos (video_id, user_id, title, status) VALUES (?, ?, ?, 'processing')
   - Publish event to Cloud Pub/Sub: "video_uploaded" {video_id, gcs_uri}
   - Return video_id to user: "Your video is processing..."
   - User sees video in "My Videos" with status "Processing"

4. VIDEO PROCESSING PIPELINE (Async)
   --- Worker 1: Transcoding ---
   - Worker consumes Pub/Sub message
   - Downloads raw video from GCS
   - Encodes using FFmpeg to multiple formats:
     * 4K 60fps (2160p) - if source is 4K
     * 1080p60 (8 Mbps)
     * 1080p30 (5 Mbps)
     * 720p30 (2.5 Mbps)
     * 480p30 (1 Mbps)
     * 360p30 (500 Kbps)
     * 240p30 (300 Kbps)
     * 144p30 (144 Kbps) - for 2G mobile
   - Splits into 10-second HLS segments
   - Uploads encoded files to GCS:
     gs://youtube-videos/dQw4w9WgXcQ/1080p60/segment_001.ts
     gs://youtube-videos/dQw4w9WgXcQ/1080p60/segment_002.ts
     ...
   - Generates master manifest (playlist.m3u8)
   - Time: 10-minute video takes ~20 minutes to encode all versions (parallel)
   
   --- Worker 2: Thumbnail Generation ---
   - Extracts frames at 0%, 25%, 50%, 75%, 100% of video
   - Generates 4 thumbnail sizes:
     * Default: 120×90
     * Medium: 320×180
     * High: 480×360
     * Max: 1280×720
   - Uses AI to select best thumbnail (face detection, brightness, composition)
   - Uploads to GCS/CDN
   
   --- Worker 3: Content ID (Copyright Detection) ---
   - Generates audio fingerprint and video fingerprint
   - Compares against Content ID database (100M+ copyrighted works)
   - If match found:
     * Flag video
     * Options: Block, Monetize (revenue goes to copyright owner), Track
     * Notify uploader
   - Time: 5-10 minutes
   
   --- Worker 4: Metadata Extraction ---
   - Extract metadata: resolution, framerate, codec, bitrate
   - Speech-to-text for automatic captions
   - Scene detection for chapter markers
   - Store in database

5. PROCESSING COMPLETE
   - Update database: UPDATE videos SET status='ready', manifest_url=? WHERE video_id=?
   - Publish "video_ready" event
   - Send notification to user: "Your video is live!"
   - Index in Elasticsearch for search
   - Pre-cache popular creator's videos to CDN
   - Video appears in subscriber feeds

6. TOTAL TIME
   - Upload: 1.5 GB at 10 Mbps = 20 minutes
   - Processing: 20-30 minutes
   - Total: Video live in 40-50 minutes
   - For small videos (< 1 minute), processing in 2-3 minutes
\`\`\`

---

## Step 7: Video Watch Flow (Detailed)

**User Clicks Play**:

\`\`\`
1. REQUEST VIDEO METADATA
   - GET /api/v1/videos/dQw4w9WgXcQ
   - Server queries database (cached in Redis):
     * Title, description, views, likes
     * Channel info
     * Manifest URL
     * Related videos (from recommendation service)
   - Response time: < 50ms (cached)

2. FETCH VIDEO MANIFEST
   - Player requests: GET https://youtube.com/dQw4w9WgXcQ/manifest.m3u8
   - CDN serves manifest (< 10ms, cached)
   - Manifest lists all available bitrates:
     #EXTM3U
     #EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080,FRAMERATE=60
     1080p60/playlist.m3u8
     #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720,FRAMERATE=30
     720p30/playlist.m3u8
     #EXT-X-STREAM-INF:BANDWIDTH=144000,RESOLUTION=256x144,FRAMERATE=30
     144p30/playlist.m3u8

3. START PLAYBACK (Adaptive Bitrate)
   - Player selects starting bitrate based on:
     * Network speed estimation (from previous videos)
     * Device capability (mobile vs desktop)
     * User preference (set quality to 1080p)
   - Default: Start with 480p or 720p for fast start
   - Request first segment: GET .../720p30/segment_001.ts
   - CDN serves segment (cached): < 100ms
   - Player begins playback at ~1.5 seconds

4. ADAPTIVE STREAMING
   - While playing segment_001, download segment_002
   - Measure bandwidth: 10 MB segment downloaded in 1 second = 80 Mbps
   - Decision: Switch to 1080p for next segment (requires 8 Mbps, we have 80 Mbps)
   - Continue measuring and adapting every 2-4 segments
   - Buffer ahead: Keep 20-30 seconds buffered

5. ANALYTICS TRACKING
   - Player sends heartbeat every 10 seconds:
     POST /api/v1/analytics/heartbeat
     {
       "video_id": "dQw4w9WgXcQ",
       "position": 45,  // seconds
       "bitrate": "1080p",
       "buffer_health": 28  // seconds buffered
     }
   - Server writes to Kafka → BigQuery
   - Aggregated for analytics dashboard:
     * Average view duration: 3:24
     * Audience retention graph
     * Traffic sources (YouTube search, suggested, external)

6. INCREMENT VIEW COUNT
   - View counted after 30 seconds of watch time (prevents spam)
   - Write to Kafka topic "view_events"
   - Consumer aggregates views in Redis:
     INCR views:dQw4w9WgXcQ
   - Batch sync to database every 10 minutes:
     UPDATE videos SET view_count = view_count + ? WHERE video_id = ?
   - View count on video page served from Redis (real-time)

7. RELATED VIDEOS SIDEBAR
   - Recommendation service queries:
     * Videos from same channel
     * Videos with similar tags
     * Collaborative filtering: Users who watched this also watched...
     * Trending in same category
   - Pre-computed recommendations cached in Redis
   - Personalized based on user watch history
\`\`\`

---

## Step 8: Search Implementation

**Architecture**:

\`\`\`
Videos Indexed in Elasticsearch:
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "description": "Official video...",
  "tags": ["music", "80s", "pop", "rick astley"],
  "channel_name": "Rick Astley",
  "transcript": "We're no strangers to love...",  // Auto-generated
  "published_at": "2009-10-25",
  "view_count": 1400000000,
  "engagement_score": 500000  // likes + comments + shares
}
\`\`\`

**Search Query**:

\`\`\`json
POST /videos/_search
{
  "query": {
    "multi_match": {
      "query": "machine learning tutorial",
      "fields": [
        "title^5",           // Title has 5x weight
        "tags^3",
        "description^2",
        "transcript",
        "channel_name^2"
      ],
      "type": "best_fields",
      "operator": "or"
    }
  },
  "function_score": {
    "functions": [
      {"field_value_factor": {"field": "view_count", "modifier": "log1p"}},
      {"field_value_factor": {"field": "engagement_score", "modifier": "log1p"}},
      {"gauss": {"published_at": {"scale": "365d", "decay": 0.5}}}  // Freshness
    ],
    "boost_mode": "sum"
  },
  "sort": [
    {"_score": "desc"},
    {"view_count": "desc"}
  ],
  "size": 20
}
\`\`\`

**Ranking Factors**:
1. **Relevance**: How well query matches title/description (TF-IDF, BM25)
2. **View count**: Popular videos ranked higher (logarithmic scaling)
3. **Engagement**: Likes, comments, shares (quality signal)
4. **Freshness**: Recent videos boosted (time decay function)
5. **Personalization**: User watch history, location, language
6. **Video quality**: Resolution, production value (ML model)
7. **Watch time**: Average percentage watched (retention)

**Autocomplete**:

\`\`\`json
POST /videos/_search
{
  "suggest": {
    "title-suggest": {
      "prefix": "mach",
      "completion": {
        "field": "title.completion",
        "size": 10,
        "contexts": {
          "language": ["en"]
        }
      }
    }
  }
}
\`\`\`

Returns: "machine learning", "machine learning tutorial", "machine shop", ...

---

## Step 9: Recommendation System

**YouTube's Recommendation Algorithm** (Simplified):

### 1. Candidate Generation

**Objective**: From 500M videos, select 100-1000 candidates

**Signals**:
- **Watch history**: Last 50 videos user watched
- **Search history**: Keywords user searched
- **Liked videos**: Videos user liked/commented on
- **Subscribed channels**: New uploads from subscriptions
- **Demographic**: Age, location, language
- **Context**: Time of day, device type

**Method**:
- **Collaborative Filtering**: Users similar to you watched these videos
  * Matrix factorization (user embeddings × video embeddings)
  * Train on billions of (user, video, watch) triples
- **Content-Based**: Videos similar to what you watched
  * Video embeddings (trained on co-watch patterns)
  * Find nearest neighbors in embedding space

### 2. Ranking

**Objective**: Rank 1000 candidates → Top 20 for homepage

**Features** (100s of features):
- **Video features**: 
  * Age, duration, view count, like ratio
  * Thumbnail click-through rate (CTR)
  * Channel authority score
- **User features**:
  * Watch time on similar videos
  * Engagement rate (like/comment rate)
  * Binge behavior (watching multiple videos in session)
- **Context features**:
  * Device type (mobile/desktop)
  * Time of day (different content for morning vs night)
  * Previously impressed (don't show again)

**Model**: Gradient Boosted Decision Trees (GBDT) or Deep Neural Network
- Trained to predict: **Expected watch time**
- Label: Actual seconds user watched
- Optimization: Maximize total watch time on YouTube

**Ranking Score**:
\`\`\`
score = P(click) × P(watch | click) × expected_watch_time
\`\`\`

Where:
- P(click): Probability user clicks video (predicted by CTR model)
- P(watch | click): Probability user watches (not immediate back)
- expected_watch_time: How long user will watch (regression model)

### 3. Re-Ranking & Diversity

- Remove duplicates (same video, same channel)
- Inject diversity: Don't show 10 gaming videos in a row
- Insert fresh content: New uploads from subscriptions
- Satisfy different intents: Trending, educational, entertainment mix

### 4. Real-Time Personalization

As user watches videos in current session:
- Update user vector in real-time
- Re-rank recommendations dynamically
- "Because you watched X" row appears

**Infrastructure**:
- Candidate generation: Batch job (every hour) → Store in Bigtable
- Ranking: Real-time (< 100ms) → TensorFlow Serving
- Caching: Recommendations cached in Redis (5-min TTL)

---

## Step 10: Live Streaming

**Architecture**:

\`\`\`
[Streamer] → [Ingest Server] → [Transcoder] → [CDN] → [Viewers]
\`\`\`

**Protocol**: RTMP (ingest) → HLS (delivery)

**Flow**:

\`\`\`
1. STREAMER SETUP
   - Streamer uses OBS (Open Broadcaster Software)
   - Configured with YouTube stream key
   - Stream URL: rtmp://a.rtmp.youtube.com/live2/{stream_key}

2. INGESTION
   - Streamer pushes RTMP stream to YouTube ingest servers
   - Ingest server located near streamer (low latency)
   - Raw stream: 1080p60 @ 6 Mbps

3. TRANSCODING (Real-Time)
   - Transcode to multiple bitrates in real-time:
     * 1080p60 (6 Mbps)
     * 720p60 (4.5 Mbps)
     * 720p30 (3 Mbps)
     * 480p30 (1.5 Mbps)
     * 360p30 (800 Kbps)
   - Generate HLS segments (2-6 second chunks)
   - Latency added: 6-20 seconds (HLS overhead)

4. CDN DISTRIBUTION
   - Segments pushed to CDN edges
   - Viewers request manifest, download segments
   - Adaptive bitrate same as on-demand

5. ULTRA-LOW LATENCY (Optional)
   - Use WebRTC instead of HLS
   - Latency: < 1 second (vs 6-20 sec for HLS)
   - Trade-off: Higher CPU cost, less scalable

6. CHAT & SUPER CHAT
   - WebSocket connection for live chat
   - Super Chat: Paid messages highlighted
   - Stored in Firestore, displayed in real-time
\`\`\`

**Scaling Live Events**:
- Popular stream: 100K+ concurrent viewers
- Pre-distribute stream to CDN edges proactively
- Use low-latency HLS (LL-HLS) for 3-5 second latency

---

## Step 11: Monetization & Analytics

### Ads Integration

**Video Ads**:
- Pre-roll (before video), mid-roll (during video), post-roll
- Skippable after 5 seconds
- Ad selection: Google Ad Exchange auction
- Targeting: Demographics, interests, context (video content)

**Ad Serving Flow**:
\`\`\`
1. Player requests video manifest
2. Manifest includes ad break markers: #EXT-X-CUE-OUT (30 seconds)
3. At ad break, player requests ad from Ad Server
4. Ad Server runs auction, returns winning ad
5. Player plays 30-second ad, then resumes video content
6. Player reports ad view: POST /api/ads/impression {ad_id, video_id}
7. Revenue split: Creator (55%), YouTube (45%)
\`\`\`

### Analytics Dashboard

**Metrics for Creators**:
- **Views**: Total views (filtered for bots)
- **Watch time**: Total minutes watched (key metric)
- **Audience retention**: Graph showing % watching at each timestamp
- **Click-through rate (CTR)**: Impressions → Clicks (thumbnail effectiveness)
- **Average view duration**: How long users watch
- **Revenue**: Estimated earnings from ads
- **Demographics**: Age, gender, location of viewers
- **Traffic sources**: YouTube search (40%), Suggested (50%), External (10%)

**Data Pipeline**:
\`\`\`
Player Events → Kafka → Streaming Job (Dataflow) → BigQuery
                              ↓
                      Aggregated Metrics
                              ↓
                      Served to Creator Dashboard
\`\`\`

---

## Step 12: Content ID (Copyright Protection)

**How Content ID Works**:

\`\`\`
1. COPYRIGHT OWNER UPLOADS REFERENCE FILE
   - Music label uploads song: "Song.mp3"
   - YouTube generates fingerprint:
     * Audio fingerprint: Spectrogram-based hash
     * Video fingerprint: Perceptual hash of keyframes
   - Store in Content ID database (100M+ files)

2. USER UPLOADS VIDEO WITH COPYRIGHTED MUSIC
   - Video uploaded, processed
   - Fingerprint generated for uploaded video
   - Compare against Content ID database
   - Match found: "Song.mp3" detected in video (30% of audio)

3. COPYRIGHT POLICY APPLIED
   - Policy set by copyright owner:
     * BLOCK: Video blocked in all countries
     * MONETIZE: Ads shown, revenue goes to copyright owner
     * TRACK: Video allowed, copyright owner gets analytics
   - Uploader notified: "Your video includes copyrighted content"

4. DISPUTE PROCESS
   - Uploader can dispute claim
   - Manual review by YouTube team
   - If valid (e.g., fair use), claim removed
\`\`\`

**Technology**:
- **Audio fingerprinting**: Chromaprint, Shazam-like algorithm
- **Video fingerprinting**: Perceptual hashing (resistant to compression, scaling)
- **Matching**: Approximate nearest neighbor search (billions of comparisons)
- **Scale**: Scans 500 hours uploaded/minute in real-time

---

## Step 13: Database Sharding Strategy

### Sharding Videos Table

**Shard by** \`video_id\` (hash-based):

\`\`\`
Shard = hash(video_id) % NUM_SHARDS

video_id "dQw4w9WgXcQ" → hash → shard 42
\`\`\`

**Why video_id?**
- Most queries: \`SELECT * FROM videos WHERE video_id = ?\` (single shard)
- User profile page: \`SELECT * FROM videos WHERE user_id = ?\` (scatter-gather acceptable, infrequent)

**Alternative**: Shard by \`(user_id, video_id)\` composite key
- User uploads stored together
- Good for "My Videos" page
- Trade-off: Global queries (trending, search) harder

**YouTube choice**: Shard by video_id, use secondary index for user_id queries

### Analytics Sharding

**BigQuery**: Partitioned by timestamp (automatic)
- Each day's data in separate partition
- Queries filtered by date very fast
- Old data archived to cold storage

---

## Step 14: Cost Optimization

### Storage Costs

**Video Storage**:
- 18 exabytes/year × $0.023/GB/month = $414M/month (!!)
- **Optimization**:
  * S3 Glacier for videos with < 100 views/year: 83% savings
  * Delete old, unpopular videos after 3 years (< 10 views)
  * Use lower bitrates for old videos (re-encode with better compression)
- **Result**: $200M/month (~50% savings)

### Bandwidth Costs

**CDN Costs**:
- 2.5 PB/day × $0.02/GB (negotiated rate) = $50M/day = $18B/year
- **Optimization**:
  * Google owns network infrastructure (peering agreements with ISPs)
  * Effective cost: $0.005-0.01/GB → $4.5B/year
  * 4x cost savings vs third-party CDN

### Encoding Costs

**Transcoding**:
- 4.3M videos/day × 10 min × $0.05/min (Google Transcoder API) = $2.15M/day = $785M/year
- **Optimization**:
  * Encode popular creators' videos to all formats immediately
  * Lazy encode long-tail videos (encode 480p, generate higher on-demand)
  * Use 50-70% savings

### Total Infrastructure Cost

Estimated: **$10-15B/year** for YouTube infrastructure
- Revenue: **$30B/year** (ads, Premium subscriptions)
- Profit: **$15-20B/year**

---

## Step 15: Advanced Optimizations

### 1. Smart Thumbnails

Generate 100+ thumbnail candidates per video:
- Extract frames at different timestamps
- Run ML model to score each thumbnail:
  * Face detection (human faces get more clicks)
  * Brightness and contrast (clear image)
  * Composition (rule of thirds)
  * Emotion detection (surprise, happiness)
- A/B test top 5 thumbnails with real users
- Select winner (highest CTR)

### 2. Preloading & Predictive Caching

**User behavior patterns**:
- 90% of users who watch video X also watch video Y (next in series)
- Preload first 10 seconds of Y while X is playing
- User experiences instant playback when clicking Y

**Implementation**:
\`\`\`
During playback of video X:
1. Recommendation API predicts next likely video: Y
2. Player silently downloads first segment of Y (in background)
3. Stores in browser cache
4. When user clicks Y, instant playback from cache
\`\`\```

### 3. Intelligent Quality Selection

**User-specific defaults**:
- Track user's typical behavior:
  * Always watches at 1080p → Default to 1080p
  * Always watches at 480p (slow network) → Default to 480p
- Save user preference in profile

**Device detection**:
- Mobile on cellular → Default to 360p (save data)
- Desktop on fiber → Default to 1080p
- Tablet on WiFi → Default to 720p

### 4. Watch Later & Offline Downloads

**Watch Later**:
- Bookmark video for later viewing
- Store list in user profile (Redis + SQL)
- Sync across devices

**Offline Downloads** (YouTube Premium):
- Download videos for offline viewing (airplanes, no connectivity)
- Encrypted with DRM (expires after 30 days)
- Store locally on device
- Track which videos downloaded (analytics)

---

## Step 16: Scaling Challenges

### Challenge 1: Hot Videos (Viral Content)

**Problem**: New viral video gets 10M views in 1 hour (100x normal)
**Solution**:
- CDN auto-scales (distributed across 300+ edges)
- Origin (GCS) sees minimal traffic (95% cache hit rate)
- Database view counter uses Redis (handles 1M+ writes/sec)
- If Redis overloaded, shard view counters by video_id

### Challenge 2: Live Event Spikes

**Problem**: 5M concurrent viewers for World Cup final
**Solution**:
- Pre-provision CDN capacity (predict spikes)
- Use adaptive bitrate aggressively (downgrade quality if needed)
- Multiple CDN providers (failover if one saturated)
- WebRTC for ultra-low latency (1-2 sec vs 10-20 sec HLS)

### Challenge 3: Copyright Abuse

**Problem**: Users re-upload copyrighted movies
**Solution**:
- Content ID scans all uploads automatically
- Match against 100M+ reference files
- Block or monetize based on copyright owner's policy
- Appeals process for false positives

### Challenge 4: Spam & Fake Views

**Problem**: Bots inflate view counts
**Solution**:
- View validation:
  * Require 30 seconds watch time
  * Check user agent (bot detection)
  * IP address reputation
  * Behavioral analysis (human vs bot patterns)
- Machine learning model to detect fake traffic
- Penalize videos with suspicious views (don't recommend)

---

## Interview Tips

### What to Clarify

1. **Upload vs Watch**: Which is more important? (Watch is 100x traffic)
2. **Features**: Live streaming? Ads? Recommendations?
3. **Scale**: YouTube-scale (billions of views/day) or startup scale?
4. **Copyright**: Need Content ID or simple DMCA takedowns?

### What to Emphasize

1. **Resumable uploads**: Critical for large video files (explain chunked upload)
2. **Async processing**: Upload and processing decoupled (pub/sub queue)
3. **CDN-first architecture**: 95%+ traffic served from edge (no origin overload)
4. **Adaptive bitrate**: Explain HLS manifests, bitrate selection algorithm
5. **Recommendation system**: High-level overview (collaborative filtering, ranking)
6. **View counting**: Redis counters with batch DB sync (explain trade-offs)
7. **Search**: Elasticsearch with multi-field search and ranking factors
8. **Sharding**: By video_id for most queries to hit single shard

### Common Mistakes

1. ❌ **Synchronous processing**: Upload blocks while encoding (terrible UX)
2. ❌ **No CDN**: Serving videos from origin (impossible at scale)
3. ❌ **Single bitrate**: Poor UX for slow networks
4. ❌ **SQL for analytics**: Use columnar store (BigQuery, Redshift)
5. ❌ **No resumable uploads**: Large file uploads fail and restart from beginning
6. ❌ **Real-time view counts in SQL**: Use Redis counters, batch sync
7. ❌ **Not considering copyright**: Content ID is critical for legal compliance

### Follow-up Questions

- **"How would you handle 4K/8K video uploads?"**
  - Encode to even more bitrates (4K = 25 Mbps, 8K = 50+ Mbps)
  - Serve only to compatible devices (detect capability)
  - Higher storage cost (offset by better video quality for premium users)

- **"What if Google's CDN goes down?"**
  - Multi-CDN strategy: YouTube CDN (primary) + CloudFront (backup)
  - DNS failover: Switch to backup CDN in 60 seconds
  - Graceful degradation: Serve lower bitrates if capacity limited

- **"How do you prevent users from downloading videos?"**
  - HLS segments are streamable, not downloadable (need special tools)
  - DRM encryption for premium content (Widevine, FairPlay)
  - Legal approach: DMCA takedowns for piracy sites

- **"How would you build YouTube Premium (ad-free subscription)?"**
  - User account flag: \`is_premium = true\`
  - Ad server checks flag before serving ads
  - Background downloads (offline viewing) enabled
  - Revenue split with creators (based on watch time, not ads)

- **"How do recommendations affect total watch time?"**
  - Bad recommendations: Users leave YouTube (low watch time)
  - Good recommendations: Users binge-watch (high watch time)
  - Optimization goal: Maximize watch time (not click-through rate)
  - Trade-off: Show clickbait (high CTR, low watch time) vs quality (low CTR, high watch time)
  - YouTube optimizes for watch time (better long-term engagement)

---

## Summary

**Core Components**:
1. **Upload Service**: Chunked resumable uploads to GCS
2. **Processing Pipeline**: Async transcoding (7 bitrates), thumbnail generation, Content ID
3. **Storage**: GCS (18 EB/year), multi-region replication
4. **CDN**: YouTube CDN (300+ edges, 95% cache hit rate)
5. **Streaming**: HLS adaptive bitrate streaming (1 PB/sec peak)
6. **Search**: Elasticsearch (500M videos indexed)
7. **Recommendations**: Collaborative filtering + deep learning ranking
8. **Analytics**: Kafka → BigQuery (time-series data)
9. **Monetization**: Google Ad Exchange integration
10. **Content ID**: Audio/video fingerprinting for copyright protection

**Key Design Decisions**:
- ✅ **Chunked resumable uploads**: 10 MB chunks, retry on failure
- ✅ **Async processing**: Upload returns immediately, processing in background (20-30 min)
- ✅ **HLS adaptive bitrate**: 7 bitrates (144p-4K), 10-second segments
- ✅ **CDN-first**: 95% traffic from edge, 5% from origin
- ✅ **Redis view counters**: Real-time counting, batch DB sync every 10 min
- ✅ **Elasticsearch search**: Multi-field search with ranking (relevance, popularity, freshness)
- ✅ **BigQuery analytics**: Time-series data, partitioned by timestamp
- ✅ **Content ID**: Automatic copyright detection on all uploads
- ✅ **Sharded by video_id**: Most queries hit single shard

**Capacity Handled**:
- **Uploads**: 4.3M videos/day (720K hours), 50 PB/day encoded
- **Views**: 1.5B videos/day (1B hours watched), 17K views/sec
- **Peak bandwidth**: 1 Pbps (petabit per second) via CDN
- **Storage**: 18 exabytes/year
- **Live streams**: 100K+ concurrent viewers per popular stream

**Costs (Estimated)**:
- **Storage**: $200M/month (optimized with Glacier)
- **Bandwidth**: $4.5B/year (own CDN infrastructure)
- **Encoding**: $400M/year (optimized with lazy encoding)
- **Total**: ~$10-15B/year operational cost

**Performance Metrics**:
- **Upload time**: 1.5 GB video uploads in 20 min (10 Mbps)
- **Processing time**: Video ready in 20-50 min (parallel encoding)
- **Video start time**: < 2 seconds globally (CDN edge + 144p/360p fast start)
- **Search latency**: < 100ms (Elasticsearch)
- **Recommendation latency**: < 100ms (cached in Redis)

This design handles **YouTube-scale traffic** (10% of global internet) with **sub-2-second video start times** globally, using a combination of Google's infrastructure, intelligent caching, adaptive bitrate streaming, and async processing pipelines. The system prioritizes user experience (instant playback, no buffering) while optimizing costs through CDN efficiency, smart encoding, and storage tiering.`,
};
