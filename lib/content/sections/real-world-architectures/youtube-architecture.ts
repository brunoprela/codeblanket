/**
 * YouTube Architecture Section
 */

export const youtubearchitectureSection = {
  id: 'youtube-architecture',
  title: 'YouTube Architecture',
  content: `YouTube is the world's largest video sharing platform with over 2.5 billion users watching 1 billion hours of video daily. Acquired by Google in 2006 for $1.65 billion, YouTube has evolved into a complex distributed system handling massive scale video uploads, processing, storage, and delivery. This section explores the architecture that powers YouTube.

## Overview

YouTube\'s scale and challenges:
- **2.5 billion users** worldwide
- **500+ hours of video** uploaded every minute
- **1 billion hours** watched daily
- **Petabytes of storage** for video content
- **Multi-codec support**: H.264, VP9, AV1
- **Global CDN**: Low-latency delivery worldwide

### Key Challenges

1. **Video processing**: Transcode to multiple formats/resolutions
2. **Storage**: Petabytes of video data
3. **Delivery**: Low-latency streaming globally
4. **Recommendations**: Surface relevant videos from billions
5. **Copyright**: Content ID system for copyright protection
6. **Monetization**: Ads, views, analytics at scale

---

## High-Level Architecture

\`\`\`
┌──────────────┐
│   Uploader   │ (Video Upload)
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  Upload Service          │
│  (Chunked Upload)        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Video Processing        │
│  (Transcoding,           │
│   Thumbnail Generation)  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Google Cloud Storage    │
│  (Master + Transcoded)   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Content Delivery        │
│  (YouTube CDN)           │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Viewer (Streaming)      │
└──────────────────────────┘
\`\`\`

---

## Core Components

### 1. Video Upload Pipeline

YouTube handles 500+ hours of video uploads every minute.

**Upload Flow**:

**1. Client-Side Preparation**:
- User selects video file
- Client validates: file size (<128 GB), format (MP4, AVI, MOV, etc.), duration (<12 hours for most users)
- Generate thumbnail locally (optional)

**2. Chunked Upload** (Resumable):
- Divide video into chunks (5-10 MB each)
- Upload chunks in parallel
- Each chunk gets unique URL
- Can resume if connection drops

\`\`\`
Video (2 GB) → Split into 400 chunks (5 MB each)
               → Upload chunks 1-100 (parallel)
               → Connection drops
               → Resume from chunk 101 (no re-upload)
\`\`\`

**3. Server-Side Storage**:
- Chunks uploaded to Google Cloud Storage (GCS)
- Object storage (S3-like)
- Stored temporarily (master file)

**4. Metadata Creation**:
- Generate video_id (unique identifier)
- Store metadata: title, description, tags, uploader, timestamp
- Store in Bigtable or Spanner

**Data Model** (Bigtable):
\`\`\`
Row Key: video_id
Columns:
  - metadata:title
  - metadata:description
  - metadata:uploader_id
  - metadata:upload_timestamp
  - metadata:duration
  - metadata:view_count
  - metadata:like_count
\`\`\`

**5. Trigger Processing**:
- Publish message to pub/sub queue: "video uploaded, video_id=123"
- Processing workers consume message
- Begin transcoding

**Upload Performance**:
- P50 upload time: <5 minutes (for 1 GB video)
- P99 upload time: <15 minutes
- Resumable upload reduces failures

---

### 2. Video Transcoding and Processing

After upload, videos are transcoded into multiple formats for different devices/bandwidths.

**Why Transcoding?**

- **Multiple resolutions**: 360p, 480p, 720p, 1080p, 1440p, 4K, 8K
- **Multiple codecs**: H.264 (widely supported), VP9 (better compression), AV1 (future-proof)
- **Adaptive bitrate streaming**: Client switches resolution based on bandwidth

**Transcoding Pipeline**:

**1. Job Scheduling**:
- Pub/Sub message received: "video_id=123, master_file=gs://bucket/master.mp4"
- Scheduler assigns to transcoding worker (distributed across datacenters)

**2. Download Master File**:
- Worker downloads master file from GCS
- May be stored temporarily on worker's local disk (fast I/O)

**3. Transcoding**:
- Use FFmpeg or custom encoder
- Generate multiple versions in parallel:
  - 4K (3840x2160) @ 15 Mbps (H.264)
  - 1080p (1920x1080) @ 8 Mbps (H.264)
  - 720p (1280x720) @ 5 Mbps (H.264)
  - 480p (854x480) @ 2.5 Mbps (H.264)
  - 360p (640x360) @ 1 Mbps (H.264)
  - (Repeat for VP9, AV1)

**4. Quality Control**:
- Automated checks: video corruption, audio sync, artifacts
- Machine learning models detect quality issues

**5. Upload Transcoded Files**:
- Upload to GCS
- Organized by video_id and resolution

\`\`\`
gs://youtube-videos/
  video_123/
    master.mp4
    1080p_h264.mp4
    720p_h264.mp4
    480p_h264.mp4
    1080p_vp9.webm
    720p_vp9.webm
\`\`\`

**6. Generate Thumbnails**:
- Extract frames at intervals (e.g., 1 frame per second)
- Select best thumbnail (ML model ranks frames by quality)
- Upload to GCS

**7. Update Metadata**:
- Mark video as "processed"
- Video now available for viewing

**Transcoding Performance**:
- 2-hour movie: Transcoded in ~30 minutes (massive parallelization)
- Distributed across 1000s of workers
- Priority: Popular channels get faster processing

**Cost Optimization**:
- Use preemptible VMs (spot instances) for non-urgent transcoding (70% cheaper)
- Transcode only popular resolutions initially (4K added later if needed)

---

### 3. Video Storage

YouTube stores petabytes of video data.

**Storage Strategy**:

**1. Google Cloud Storage (GCS)**:
- Object storage (S3-equivalent)
- Highly durable (11 nines)
- Multiple storage classes:
  - **Standard**: Frequently accessed (recent uploads)
  - **Nearline**: Accessed occasionally (videos with low views)
  - **Coldline**: Rarely accessed (old, unpopular videos)

**2. Storage Organization**:
\`\`\`
gs://youtube-videos/
  video_123/
    master.mp4
    transcoded/
      1080p_h264.mp4
      720p_h264.mp4
      ...
    thumbnails/
      thumb1.jpg
      thumb2.jpg
\`\`\`

**3. Deduplication**:
- Hash video file (SHA-256)
- If hash exists (duplicate), reuse existing video
- Reduces storage for re-uploads

**4. Compression**:
- VP9 codec: 30% smaller than H.264 (same quality)
- AV1 codec: 50% smaller than H.264 (same quality)
- Trade-off: Encoding time (AV1 is slow)

**Scale**:
- Petabytes of storage
- Cost: $0.02/GB/month (standard), $0.01/GB/month (nearline)
- YouTube's bill: Millions per month

**Optimization**:
- Move old videos to Coldline (cheaper storage)
- Delete least popular resolutions (keep only 480p for videos with <100 views/year)

---

### 4. Content Delivery Network (CDN)

YouTube delivers videos via a global CDN for low latency.

**YouTube CDN**:
- Built on Google\'s infrastructure
- Edge locations worldwide (1000s of PoPs)
- Co-located with ISPs for minimal hops

**How CDN Works**:

**1. Video Request**:
\`\`\`
User clicks video → YouTube backend
                         ↓
                    Determine closest edge location
                         ↓
                    Return CDN URL (e.g., video.googlevideo.com)
                         ↓
                    Client requests video from CDN
                         ↓
                    CDN serves from cache (cache hit) OR fetches from origin (cache miss)
\`\`\`

**2. Cache Strategy**:
- **Popular videos**: Cached at all edge locations (cache hit 99%+)
- **Unpopular videos**: Fetched from origin on-demand, cached temporarily
- **Cache eviction**: LRU (Least Recently Used)

**3. Adaptive Bitrate Streaming (ABR)**:
- Video divided into chunks (2-10 seconds each)
- Each chunk available at multiple resolutions
- Client measures bandwidth, requests appropriate resolution

\`\`\`
Manifest file (.mpd or .m3u8):
- Lists all chunks and available resolutions
- Client parses manifest, requests chunks sequentially

Client algorithm:
  if bandwidth > 10 Mbps:
      request 1080p chunk
  elif bandwidth > 5 Mbps:
      request 720p chunk
  else:
      request 480p chunk
\`\`\`

**4. Protocols**:
- **DASH** (Dynamic Adaptive Streaming over HTTP): Industry standard
- **HLS** (HTTP Live Streaming): Apple devices

**CDN Performance**:
- P50 latency: <50ms (video start time)
- P99 latency: <200ms
- Buffering rate: <1% of playback time

---

### 5. Recommendation System

YouTube's recommendation algorithm drives 70%+ of watch time.

**Goals**:
- **Maximize watch time**: Keep users engaged
- **Surface relevant content**: Personalized recommendations
- **Discover new creators**: Balance popular vs emerging

**Recommendation Types**:

**1. Homepage Recommendations**:
- Personalized grid of videos
- Based on: Watch history, subscriptions, likes, search history
- Updated in real-time

**2. Suggested Videos (Sidebar)**:
- Videos similar to currently watching video
- Based on: Video metadata, co-watch patterns (users who watched X also watched Y)

**3. Search Results**:
- Videos matching search query
- Ranked by: Relevance, watch time, recency

**Machine Learning Pipeline**:

**1. Candidate Generation**:
- From billions of videos, select ~1000 candidates
- Methods:
  - Collaborative filtering: Users similar to you watched X
  - Content-based: You watched genre Y, here are similar videos
  - Trending: Popular videos in your region

**2. Ranking**:
- Rank 1000 candidates by predicted watch time
- Features:
  - User features: Watch history, subscriptions, demographics
  - Video features: Title, description, tags, thumbnail, duration
  - Context features: Time of day, device, location
- Model: Deep neural network (TensorFlow)
- Output: Predicted watch time for each video

**3. Serving**:
- Return top 20 videos
- A/B testing: Different algorithms for different users

**Feedback Loop**:
- User watches video → Record engagement (watch time, likes, shares)
- Retrain model with new data (hourly/daily)
- Continuously improve recommendations

**Challenges**:

**1. Cold Start**:
- New user: No watch history
- Solution: Default recommendations (trending, popular)

**2. Filter Bubble**:
- Recommending only similar content
- Solution: Inject diversity (explore vs exploit)

**3. Clickbait**:
- Misleading thumbnails, titles
- Solution: Penalize videos with high click-through but low watch time

---

### 6. View Count and Analytics

YouTube tracks billions of video views and provides analytics to creators.

**View Counting**:

**Challenges**:
- **High volume**: 1 billion hours watched daily = billions of view events
- **Fraud detection**: Bots artificially inflating view counts
- **Real-time**: View count updated in real-time (or near real-time)

**Architecture**:

**1. Event Collection**:
- Client sends "view" event when video played for >30 seconds
- Event includes: video_id, user_id, timestamp, watch_duration, device

**2. Stream Processing**:
- Events streamed to Pub/Sub
- Apache Beam / Cloud Dataflow processes events
- Filter fraud (bots, repeated views from same user)

**3. Aggregation**:
- Count views per video (group by video_id)
- Store in Bigtable or Spanner

\`\`\`
Table: video_views
Row Key: video_id
Columns:
  - view_count:total
  - view_count:last_hour
  - view_count:last_day
\`\`\`

**4. Display**:
- View count fetched from database
- Cached in Memcached (TTL: 1 minute for popular videos, 10 minutes for unpopular)

**Fraud Detection**:
- Repeated views from same IP within 24 hours: Counted as 1 view
- Bot traffic: Machine learning models detect patterns (rapid requests, no mouse movement)
- Click farms: Geographic patterns, similar behavior

**Analytics Dashboard (YouTube Studio)**:

Creators get detailed analytics:
- **Views**: Total, by date, by geography
- **Watch time**: Total hours, average duration
- **Traffic sources**: Search, suggested videos, external websites
- **Audience demographics**: Age, gender, location
- **Engagement**: Likes, dislikes, comments, shares

**Data Pipeline**:
\`\`\`
View events → Pub/Sub → Beam/Dataflow → BigQuery (data warehouse)
                                              ↓
                                         Analytics API
                                              ↓
                                         YouTube Studio UI
\`\`\`

---

### 7. Content ID (Copyright Protection)

YouTube's Content ID system detects copyrighted content automatically.

**How Content ID Works**:

**1. Reference File Submission**:
- Copyright holders (studios, labels) upload reference files
- System generates "fingerprint" (audio/video signature)
- Store fingerprints in database

**2. Uploaded Video Fingerprinting**:
- When user uploads video, generate fingerprint
- Compare against reference database

**3. Match Detection**:
- If fingerprint matches (with similarity threshold), it's a match
- Copyright holder notified

**4. Actions**:
- **Block**: Video cannot be viewed
- **Monetize**: Ads run, revenue goes to copyright holder
- **Track**: Just track views, no action

**Fingerprinting Technology**:
- **Audio fingerprint**: Robust to pitch shift, speed changes, noise
- **Video fingerprint**: Detects video even if cropped, color-adjusted, mirrored

**Challenges**:
- **False positives**: Legitimate fair use flagged
- **Scalability**: Compare every upload against millions of reference files
- **Performance**: Fingerprint comparison must be fast (<1 minute)

**Scale**:
- 500+ hours uploaded per minute
- Millions of reference files
- Billions of comparisons per day

---

## Live Streaming

YouTube Live streams events in real-time (gaming, concerts, news).

**Live Streaming Architecture**:

**1. Streamer Setup**:
- Use OBS (Open Broadcaster Software) or similar
- Encode video/audio locally
- Stream via RTMP (Real-Time Messaging Protocol) to YouTube

**2. Ingest**:
- YouTube receives RTMP stream
- Buffer 5-10 seconds (for encoding)

**3. Transcoding (Real-Time)**:
- Transcode to multiple resolutions (same as VOD)
- Must be real-time (no delays)
- Use low-latency encoders

**4. Packaging**:
- Package into DASH/HLS chunks
- Upload to CDN

**5. Delivery**:
- Viewers request chunks from CDN
- Latency: 10-30 seconds (buffer + transcoding + delivery)

**Low-Latency Mode**:
- Reduce latency to <5 seconds
- Trade-off: Less buffering = more rebuffering

**Chat**:
- Real-time chat using WebSockets
- Messages stored in Spanner (distributed SQL)
- Thousands of messages per second for popular streams

---

## Search

YouTube search lets users find videos from billions available.

**Search Index** (Elasticsearch / Custom):

**Indexing**:
- Index video metadata: title, description, tags
- Extract keywords via NLP
- Store in inverted index (keyword → video IDs)

**Search Query**:
\`\`\`
User searches "machine learning tutorial"
    ↓
Tokenize: ["machine", "learning", "tutorial"]
    ↓
Query index: videos with all three keywords
    ↓
Retrieve ~10,000 matching videos
    ↓
Rank by relevance (title match, view count, recency, watch time)
    ↓
Return top 20
\`\`\`

**Ranking Factors**:
- **Keyword relevance**: Title/description match
- **View count**: Popular videos ranked higher
- **Watch time**: High retention ranked higher
- **Recency**: Recent uploads boosted
- **User personalization**: Match user interests

**Autocomplete**:
- Suggest queries as user types
- Based on: Popular searches, user's search history
- Powered by trie data structure

---

## Comments and Engagement

YouTube supports comments, likes, dislikes, shares, subscriptions.

**Comment System**:

**Data Model** (Spanner or Bigtable):
\`\`\`
Table: comments
Partition Key: video_id
Clustering Key: timestamp DESC
Columns: comment_id, user_id, text, like_count, reply_to (for nested replies)
\`\`\`

**Features**:
- **Nested replies**: Comments can have replies (tree structure)
- **Likes**: Users can like comments
- **Sorting**: Top comments, newest first
- **Moderation**: Creators can delete, pin, hide comments
- **Spam detection**: ML models flag spam, abusive comments

**Real-Time Updates**:
- WebSocket for live comment updates (live streams, premieres)
- Polling for regular videos (refresh every 30 seconds)

---

## Technology Stack

### Google Infrastructure

**1. Google Cloud Storage (GCS)**:
- Video storage (petabytes)
- Highly durable, scalable

**2. Bigtable**:
- NoSQL database for metadata, analytics
- High write throughput (millions of views/second)

**3. Spanner**:
- Distributed SQL for relational data (users, comments)
- Global consistency

**4. Pub/Sub**:
- Event streaming (uploads, views, likes)
- Decouples producers and consumers

**5. Dataflow / Apache Beam**:
- Stream and batch processing (analytics, fraud detection)

**6. BigQuery**:
- Data warehouse for analytics (petabyte-scale queries)

**7. Kubernetes (GKE)**:
- Container orchestration for microservices

**8. TensorFlow**:
- Machine learning (recommendations, content moderation)

---

## Key Lessons

### 1. Video Processing is Expensive

Transcoding to multiple formats/resolutions requires massive compute. Use spot instances and prioritize popular content.

### 2. Storage Costs Add Up

Petabytes of storage cost millions monthly. Use tiered storage (Standard, Nearline, Coldline) and delete unpopular resolutions.

### 3. CDN is Critical for Quality

Low-latency streaming requires global CDN. Cache popular content, fetch unpopular on-demand.

### 4. Recommendations Drive Engagement

70%+ of watch time comes from recommendations. Invest in ML for personalization.

### 5. Content Moderation at Scale

With 500+ hours uploaded per minute, automated moderation (Content ID, spam detection) is essential.

---

## Interview Tips

**Q: How would you design YouTube\'s video upload and processing pipeline?**

A: Chunked resumable upload: Divide video into 5 MB chunks, upload in parallel, store in GCS. Generate video_id, store metadata in Bigtable. Publish "upload complete" event to Pub/Sub. Transcoding workers consume event, download master file, transcode to multiple resolutions (1080p, 720p, 480p) and codecs (H.264, VP9) in parallel using FFmpeg. Upload transcoded files to GCS. Generate thumbnails (extract frames, ML selects best). Update metadata to "processed." Use preemptible VMs for cost savings. Prioritize popular channels. Handle failures with retries and dead-letter queues.

**Q: How does YouTube handle billions of video views?**

A: Stream view events to Pub/Sub (video_id, user_id, timestamp). Process with Dataflow: (1) Filter fraud (bots, repeated views from same IP). (2) Aggregate views per video. (3) Store in Bigtable (video_id → view_count). Cache in Memcached (TTL: 1 minute for popular videos). Display to users. For analytics, stream to BigQuery (data warehouse). Creators query via YouTube Studio API. Fraud detection: ML models detect bot patterns (rapid requests, no engagement). Scalability: Dataflow auto-scales, Bigtable handles millions of writes/second.

**Q: How would you design YouTube's recommendation system?**

A: Two-stage approach: (1) Candidate generation: From billions of videos, select ~1000 candidates using collaborative filtering (users like you watched X), content-based (similar to your history), and trending (popular in your region). (2) Ranking: Train deep neural network (TensorFlow) to predict watch time for each candidate. Features: user history, video metadata, context (time of day, device). Rank by predicted watch time, return top 20. Retrain model daily with new engagement data. Handle cold start (new users) with default recommendations. Inject diversity to avoid filter bubbles. Penalize clickbait (high click-through, low watch time).

---

## Summary

YouTube's architecture demonstrates building a video platform at massive scale:

**Key Takeaways**:

1. **Chunked upload**: Resumable uploads handle large files reliably
2. **Transcoding pipeline**: Parallel processing, multiple formats/resolutions, spot instances for cost
3. **Polyglot storage**: GCS for videos, Bigtable for metadata, Spanner for relational, BigQuery for analytics
4. **Global CDN**: Low-latency delivery, adaptive bitrate streaming
5. **ML recommendations**: Candidate generation + ranking, continuous retraining
6. **Content ID**: Automatic copyright detection via fingerprinting
7. **Stream processing**: Pub/Sub + Dataflow for views, analytics, fraud detection
8. **Tiered storage**: Standard, Nearline, Coldline for cost optimization

YouTube\'s success relies on Google's infrastructure (GCS, Bigtable, Spanner, TensorFlow) and sophisticated ML for recommendations and moderation.
`,
};
