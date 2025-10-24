/**
 * Design Netflix Section
 */

export const netflixSection = {
  id: 'netflix',
  title: 'Design Netflix',
  content: `Netflix is a video streaming platform serving 200+ million subscribers worldwide. The core challenges are: storing petabytes of video content, encoding for multiple devices and bandwidths, delivering with low latency globally via CDN, and providing personalized recommendations.

## Problem Statement

Design Netflix with:
- **Video upload & encoding**: Convert videos to multiple formats/bitrates
- **Content delivery**: Stream videos with < 2 second start time globally
- **Adaptive bitrate streaming**: Adjust quality based on bandwidth
- **Search & browse**: Find movies/shows
- **Recommendations**: Personalized content suggestions
- **User management**: Profiles, watch history, continue watching

**Scale**: 200M users, 10B hours watched/month, 100 PB video library

---

## Step 1: Requirements

### Functional

1. **Upload Videos**: Content providers upload movies/shows
2. **Encode Videos**: Transcode to multiple formats (1080p, 720p, 480p, etc.)
3. **Stream Videos**: Adaptive bitrate streaming (ABR)
4. **Search**: Find content by title, genre, actor
5. **Recommendations**: Personalized homepage
6. **Watch History**: Resume playback, track progress
7. **Profiles**: Multiple users per account

### Non-Functional

1. **Low Latency**: < 2sec video start time
2. **High Availability**: 99.99% uptime
3. **Scalable**: Support 200M concurrent users
4. **Global**: Low latency worldwide (CDN)
5. **Cost-Effective**: Optimize bandwidth costs

---

## Step 2: Capacity Estimation

**Users**: 200M subscribers, 20% concurrently streaming (peak) = 40M concurrent
**Video Library**: 10,000 titles × 2 hours avg × 10 versions (bitrates/formats) = 200K video files
**Storage**: 1 hour video @ 5 Mbps avg = 2.25 GB. 200K files × 2.25 GB = 450 TB → With all bitrates: ~10 PB
**Bandwidth**: 40M streams × 5 Mbps = 200 Tbps (managed by CDN)

---

## Step 3: High-Level Architecture

\`\`\`
Content Upload → Encoding Pipeline → S3 Storage → CDN → User

[Content Provider]
       ↓
[Upload Service] → S3 Raw Videos
       ↓
[Encoding Service] (AWS Elemental MediaConvert)
   → Generates: 4K, 1080p, 720p, 480p, 360p (HLS/DASH)
       ↓
[S3 Encoded Videos] → CloudFront CDN (Edge Locations)
       ↓
[User Devices] (Adaptive Bitrate Player)
\`\`\`

---

## Step 4: Video Encoding Pipeline

**Input**: Raw 4K video (100 GB, H.264, MP4)
**Process**:
1. Split into chunks (10-second segments)
2. Encode each chunk in parallel to multiple bitrates:
   - 4K (25 Mbps), 1080p (8 Mbps), 720p (5 Mbps), 480p (2.5 Mbps), 360p (1 Mbps)
3. Generate HLS/DASH manifest files
4. Upload to S3
**Time**: 100 GB video → 4 hours encoding (parallel workers)

**HLS Manifest** (playlist.m3u8):
\`\`\`
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080
1080p/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1280x720
720p/playlist.m3u8
\`\`\`

**Adaptive Bitrate Streaming (ABR)**: Player downloads manifest, starts with low bitrate, measures bandwidth, switches to higher bitrate if available. User experiences no buffering.

---

## Step 5: Content Delivery Network (CDN)

**Why CDN is Critical**:
- **Latency**: S3 origin (US-East-1) → User in India = 300ms. CDN edge (Mumbai) → User = 20ms.
- **Bandwidth Cost**: S3 egress: $0.09/GB. CDN: $0.02-0.05/GB (cheaper for popular content).
- **Origin Load**: Without CDN, S3 serves 200 Tbps (impossible). With CDN (95% hit rate), S3 serves 10 Tbps (manageable).

**Netflix CDN Strategy**:
- **Open Connect**: Netflix's custom CDN (appliances inside ISP networks)
- **Third-party CDNs**: CloudFront, Akamai, Fastly as fallback
- **Edge Servers**: Cache popular content at 200+ locations globally

**Cache Strategy**:
- Popular titles (Top 100): Pre-pushed to all edge locations (cache for 30+ days)
- Warm titles: Cached on first request (cache for 7 days)
- Cold titles: Served from origin

---

## Step 6: Video Streaming Flow

\`\`\`
1. User clicks "Play" on video
2. App requests: GET /api/videos/{video_id}/manifest.m3u8
3. API returns CDN URL: https://cdn.netflix.com/video123/manifest.m3u8
4. Player downloads manifest (lists all bitrate options)
5. Player starts with 360p (fast start), downloads 10-sec chunk
6. While playing, player measures bandwidth
7. If bandwidth > 5 Mbps, switch to 720p for next chunk
8. Smooth streaming continues, adapting to network conditions
\`\`\`

**Start Time Optimization**: 
- Manifest cached in CDN (< 10ms)
- First 360p chunk pre-loaded (< 500ms)
- Total start time: < 2 seconds

---

## Step 7: Database Schema

**Videos Table** (PostgreSQL):
\`\`\`sql
CREATE TABLE videos (
    video_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255),
    description TEXT,
    duration INT,  -- seconds
    genres JSON,   -- ["Action", "Drama"]
    manifest_url VARCHAR(500),
    thumbnail_url VARCHAR(500),
    release_date DATE
);
\`\`\`

**User Watch History** (Cassandra - time-series):
\`\`\`cql
CREATE TABLE watch_history (
    user_id BIGINT,
    video_id VARCHAR(50),
    watched_at TIMESTAMP,
    position INT,  -- seconds watched
    PRIMARY KEY (user_id, watched_at)
);
\`\`\`

**Recommendations** (Elasticsearch):
- Index videos by: genre, actors, tags, user ratings
- Collaborative filtering: "Users who watched X also watched Y"

---

## Step 8: Recommendations System

**High-Level**:
1. **User Profile**: Watch history, ratings, genres viewed
2. **Collaborative Filtering**: Matrix factorization (similar users)
3. **Content-Based**: Genre, actors, tags matching
4. **Trending**: Popular now (time-weighted views)
5. **Personalized Rows**: "Because you watched X", "Trending in Action"

**Implementation**: 
- Offline batch job (Spark) processes watch history daily
- Generates recommendations per user → Store in Redis
- API serves pre-computed recommendations (< 50ms)

**Real-time**: 
- User watches new video → Update user vector
- Re-compute recommendations incrementally (online learning)

---

## Step 9: Search

**Elasticsearch**:
\`\`\`json
POST /videos/_search
{
  "query": {
    "multi_match": {
      "query": "action thriller",
      "fields": ["title^3", "description", "genres", "actors"]
    }
  },
  "sort": [{"popularity": "desc"}],
  "size": 20
}
\`\`\`

**Autocomplete**: 
- Use Elasticsearch edge n-grams for prefix search
- User types "str" → Suggests "Stranger Things", "Strong Man"

---

## Step 10: Optimizations

**1. Pre-Encoding Popular Content**:
- Encode top 1000 titles to all formats proactively
- Encode long-tail content on-demand (lazy encoding)

**2. Intelligent Pre-fetching**:
- If user watches Episode 1, pre-fetch Episode 2 (90% likely to watch next)

**3. Compression**:
- Use H.265/HEVC (50% smaller than H.264, same quality)
- Trade-off: Higher encoding cost, lower bandwidth cost

**4. Thumbnail Generation**:
- Generate 100 thumbnails per video (every 10 seconds)
- A/B test which thumbnail drives more clicks

**5. Offline Viewing**:
- Allow downloads for mobile users
- DRM-encrypted, expires after 30 days

---

## Step 11: Cost Optimization

**Bandwidth** (Largest cost):
- 40M streams × 5 Mbps × 3 hours/day = 2.7 PB/day
- At $0.05/GB CDN cost = $138M/day → $50B/year (!)
- **Solution**: Open Connect (own CDN inside ISPs) → Reduces to $0.01/GB → $10B/year

**Storage**:
- 10 PB at $0.023/GB/month = $230K/month
- Use S3 Intelligent-Tiering for old content

**Encoding**:
- Parallel encoding on spot instances (70% cheaper)
- Encode once, serve millions of times (amortized cost)

---

## Step 12: Monitoring

**Metrics**:
- **Video Start Time**: p95 < 2 seconds
- **Rebuffer Rate**: < 0.5% (buffering mid-playback)
- **Bitrate Distribution**: % users at each quality
- **CDN Cache Hit Rate**: > 95%
- **Concurrent Streams**: Real-time dashboard

**Alerting**:
- Start time > 3 seconds → CDN performance issue
- Rebuffer rate > 1% → Origin overload or ISP congestion

---

## Trade-offs

**H.264 vs H.265**:
- H.265: 50% smaller, but slower encoding (2x CPU), not all devices support
- Netflix uses H.264 for compatibility, H.265 for 4K content

**Push vs Pull CDN**:
- **Push**: Pre-distribute popular content to edges (lower latency, higher storage cost)
- **Pull**: On-demand caching (higher latency on first request, lower storage cost)
- Netflix uses both: Push top 100, pull long-tail

---

## Interview Tips

**Clarify**:
1. Live streaming vs on-demand?
2. User-generated content (YouTube) vs curated (Netflix)?
3. Scale: How many users?

**Emphasize**:
1. **CDN**: Critical for global low-latency delivery
2. **Adaptive Bitrate**: Explain HLS/DASH manifests
3. **Encoding Pipeline**: Parallel multi-bitrate encoding
4. **Cost**: Bandwidth dominates costs at scale

**Common Mistakes**:
- Not using CDN (origin can't serve 200 Tbps)
- Single bitrate (poor UX on slow networks)
- Synchronous encoding (upload blocks for hours)
- Ignoring bandwidth costs (can bankrupt company)

---

## Summary

**Components**:
1. **Upload Service**: Receive content from providers
2. **Encoding Pipeline**: AWS Elemental MediaConvert (multi-bitrate HLS)
3. **S3 Storage**: 10 PB video library
4. **CDN**: Open Connect (custom) + CloudFront (95% cache hit)
5. **API Servers**: Serve metadata, recommendations
6. **Recommendation Engine**: Spark + collaborative filtering
7. **Elasticsearch**: Search & discovery

**Key Decisions**:
- ✅ Adaptive bitrate streaming (HLS/DASH)
- ✅ CDN-first architecture (95%+ traffic served from edge)
- ✅ Open Connect (custom CDN inside ISPs) for cost savings
- ✅ Parallel encoding pipeline (4 hours for 100 GB video)
- ✅ Pre-computed recommendations (served from Redis)

**Capacity**:
- 200M users, 40M peak concurrent streams
- 10 PB video library (all bitrates)
- 2.7 PB/day bandwidth (200 Tbps peak)
- < 2 second video start time globally

Netflix's architecture prioritizes **user experience** (low latency, no buffering) and **cost efficiency** (own CDN, intelligent caching) at massive global scale.`,
};
