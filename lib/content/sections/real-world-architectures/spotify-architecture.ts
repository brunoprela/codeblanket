/**
 * Spotify Architecture Section
 */

export const spotifyarchitectureSection = {
  id: 'spotify-architecture',
  title: 'Spotify Architecture',
  content: `Spotify is the world's largest music streaming platform with over 500 million users streaming 100 billion songs monthly. With a catalog of 100+ million tracks, Spotify's architecture must handle massive scale audio streaming, personalized recommendations, and real-time social features. This section explores the technical systems behind Spotify.

## Overview

Spotify's scale and challenges:
- **500+ million users** (200+ million paid subscribers)
- **100 million+ tracks** in catalog
- **100 billion+ songs** streamed monthly
- **Real-time**: Instant playback, live lyrics, social features
- **Personalization**: Discover Weekly, Daily Mixes, wrapped
- **Multi-platform**: Mobile, desktop, web, smart speakers, cars

### Key Challenges

1. **Audio delivery**: Low-latency streaming globally
2. **Recommendations**: Surface relevant music from 100M+ tracks
3. **Social features**: Friend activity, collaborative playlists, Blend
4. **Rights management**: Track royalties for millions of artists
5. **Scale**: Billions of streams per day

---

## Core Components

### 1. Audio Streaming and Delivery

Spotify streams audio with low latency and high quality.

**Audio Formats**:
- **Ogg Vorbis**: Default codec (better quality than MP3 at same bitrate)
- **AAC**: For iOS devices
- **Quality tiers**:
  - Free: 128 kbps (good quality)
  - Premium: 320 kbps (very high quality)
  - Lossless: Coming soon (CD quality, 1411 kbps)

**Streaming Protocol**:
- **HTTP streaming**: Chunked transfer
- **Adaptive bitrate**: Adjust quality based on bandwidth
- **Pre-fetching**: Download next song while current song playing

**Audio Storage and CDN**:

**Storage**:
- Audio files stored in **Google Cloud Storage (GCS)**
- Multiple encodings per track (128 kbps, 320 kbps, different formats)

**CDN**:
- Delivered via **Google Cloud CDN** and custom edge locations
- Edge locations cache popular tracks
- Cache hit rate: 90%+

**Playback Flow**:
\`\`\`
User clicks play → Spotify backend
                        ↓
                   Determine audio URL (CDN edge)
                        ↓
                   Return manifest (audio chunks)
                        ↓
                   Client requests chunks from CDN
                        ↓
                   CDN serves from cache (hit) or fetches from GCS (miss)
                        ↓
                   Client decodes and plays audio
\`\`\`

**Pre-fetching and Caching**:
- **Local cache**: Store recently played tracks on device (save bandwidth)
- **Pre-fetch next track**: Download next track in playlist (seamless playback)
- **Offline mode**: Download tracks for offline playback (Premium feature)

---

### 2. Music Catalog and Metadata

Spotify's catalog has 100+ million tracks with rich metadata.

**Metadata**:
- **Track**: Title, artist, album, duration, ISRC (International Standard Recording Code)
- **Artist**: Name, bio, images, genres, followers
- **Album**: Title, release date, cover art, tracks
- **Playlist**: Title, description, tracks, follower count
- **Lyrics**: Time-synced lyrics

**Storage**: **Cassandra** (NoSQL)

**Data Model**:
\`\`\`
Table: tracks
Row Key: track_id
Columns:
  - metadata:title
  - metadata:artist_id
  - metadata:album_id
  - metadata:duration
  - metadata:isrc
  - metadata:audio_url

Table: artists
Row Key: artist_id
Columns:
  - metadata:name
  - metadata:bio
  - metadata:genres
  - metadata:image_url
  - metadata:follower_count
\`\`\`

**Why Cassandra?**:
- High read throughput (billions of queries per day)
- Scalable (add nodes for capacity)
- Global distribution (multi-region replication)

**Metadata Updates**:
- New releases ingested daily (record labels upload via Spotify for Artists)
- Metadata enrichment: Genre tagging, mood classification (ML models)

---

### 3. Search

Spotify search allows users to find tracks, artists, albums, playlists, podcasts.

**Search Index**: **Elasticsearch**

**Indexing**:
- Index track metadata: title, artist name, album name, lyrics
- Extract keywords via NLP
- Store in inverted index (keyword → track IDs)

**Search Query**:
\`\`\`
User searches "shape of you"
    ↓
Tokenize: ["shape", "of", "you"]
    ↓
Query Elasticsearch: tracks with all keywords
    ↓
Retrieve ~10,000 matching tracks
    ↓
Rank by relevance (exact title match, popularity, recency)
    ↓
Return top 20
\`\`\`

**Ranking Factors**:
- **Exact match**: Title/artist exact match ranked highest
- **Popularity**: Stream count, monthly listeners
- **Recency**: Recent releases boosted
- **User personalization**: Match user's listening history

**Autocomplete**:
- Suggest as user types
- Powered by trie data structure
- Based on popular searches and user's history

---

### 4. Recommendation System

Spotify's recommendations drive discovery and engagement.

**Recommendation Types**:

**1. Discover Weekly** (Personalized playlist, refreshed Monday):
- Based on user's listening history
- Collaborative filtering: Users similar to you listened to X

**2. Daily Mixes** (Up to 6 playlists, updated daily):
- Clustered by genre/mood
- Mix of favorite tracks + new discoveries

**3. Release Radar** (New releases from followed artists):
- Updated Friday
- Tracks from artists user follows

**4. Spotify Radio** (Infinite playlist based on seed):
- User selects seed track/artist
- Generate infinite playlist of similar music

**5. Wrapped** (Year-end summary):
- Top artists, tracks, genres, listening stats
- Personalized recap, shareable

**Machine Learning Pipeline**:

**1. Data Collection**:
- Listen events (track_id, user_id, timestamp, duration, skipped, liked)
- User profiles (age, location, subscribed playlists)
- Track features (tempo, energy, danceability, valence)

**2. Feature Engineering**:
- **Audio features**: Extracted via signal processing (tempo, key, loudness, timbre)
- **Collaborative features**: User-track interactions, similar users
- **Content features**: Genre, mood, lyrics sentiment

**3. Models**:
- **Collaborative filtering**: Matrix factorization (user-track matrix)
- **Content-based**: Track similarity (cosine similarity of audio features)
- **Deep learning**: Neural networks (TensorFlow) for complex patterns
- **Contextual bandits**: Optimize for engagement (clicks, streams)

**4. Serving**:
- Pre-compute recommendations offline (batch jobs)
- Store in database (user_id → recommended_track_ids)
- API serves recommendations to clients
- Real-time adjustments based on recent listening

**Challenges**:

**1. Cold Start**:
- New user: No listening history
- Solution: Ask for favorite artists during onboarding, use popular tracks

**2. Filter Bubble**:
- Recommending only similar music
- Solution: Inject diversity (explore vs exploit)

**3. Scalability**:
- Billions of user-track pairs
- Solution: Distributed computing (Apache Spark for batch jobs)

---

### 5. Playlists and User Libraries

Playlists are core to Spotify's user experience.

**Playlist Types**:

**1. User-Created Playlists**:
- Users create, add tracks, share
- Public or private
- Collaborative (multiple users can edit)

**2. Spotify-Curated Playlists**:
- Created by Spotify editors
- Examples: Today's Top Hits, RapCaviar, Hot Country
- Updated frequently (daily/weekly)

**3. Algorithmic Playlists**:
- Generated by ML models
- Examples: Discover Weekly, Daily Mixes, Release Radar

**Data Model** (PostgreSQL):
\`\`\`sql
Table: playlists
- playlist_id (primary key)
- user_id (creator)
- title
- description
- is_public
- is_collaborative
- follower_count

Table: playlist_tracks
- playlist_id
- track_id
- position (order in playlist)
- added_at

Table: playlist_followers
- playlist_id
- user_id
- followed_at
\`\`\`

**Playlist Operations**:
- **Add track**: INSERT into playlist_tracks
- **Remove track**: DELETE from playlist_tracks
- **Reorder tracks**: UPDATE position
- **Follow playlist**: INSERT into playlist_followers

**Challenges**:

**1. Collaborative Playlists**:
- Multiple users editing simultaneously
- Conflict resolution (last-writer-wins for now, could use OT)

**2. Large Playlists**:
- Playlists with 10,000+ tracks
- Pagination for performance

**3. Playlist Recommendations**:
- Recommend playlists to users
- Based on listening history, followed playlists

---

### 6. Social Features

Spotify has social features for discovering music through friends.

**Social Features**:

**1. Friend Activity**:
- See what friends are listening to in real-time
- Desktop sidebar shows friends' current tracks

**2. Collaborative Playlists**:
- Multiple users can add/remove tracks
- Real-time sync across users

**3. Blend**:
- Create shared playlist with friend
- Mix of both users' tastes

**4. Sharing**:
- Share tracks, albums, playlists via URL
- Integrations: Instagram Stories, Twitter, WhatsApp

**Real-Time Architecture**:

**WebSocket**:
- Persistent connection for real-time updates
- Server pushes friend activity updates to client

**Data Flow**:
\`\`\`
User A plays track
    ↓
Client sends "now_playing" event to backend
    ↓
Backend updates User A's current track in database
    ↓
Backend notifies User A's friends via WebSocket
    ↓
Friends' clients display "User A is listening to Track X"
\`\`\`

**Storage** (Redis):
\`\`\`
Key: now_playing:user:123
Value: {track_id: "abc", timestamp: 1698158400}
TTL: 5 minutes (stale data auto-expires)
\`\`\`

---

### 7. Podcasts

Spotify has invested heavily in podcasts, acquiring Anchor, Gimlet, Parcast.

**Podcast Features**:
- 5 million+ podcasts
- Original content (exclusive shows)
- Video podcasts
- Interactive polls, Q&A

**Architecture**:

**Podcast Ingestion**:
- Creators upload via Anchor or Spotify for Podcasters
- Audio files stored in GCS
- Metadata (title, description, episode list) stored in database

**Podcast Playback**:
- Similar to music streaming
- Chunked HTTP streaming
- Resume playback across devices (sync position)

**Recommendations**:
- Separate recommendation pipeline (podcast-specific models)
- Based on: Listening history, subscribed shows, trending topics

**Monetization**:
- Ads inserted dynamically (dynamic ad insertion, DAI)
- Targeted based on user demographics, interests

---

### 8. Lyrics and Canvas

**Lyrics**:
- Time-synced lyrics (karaoke-style)
- Lyrics provided by Musixmatch
- Displayed in real-time as song plays

**Implementation**:
\`\`\`
Lyrics format (JSON):
{
  "track_id": "abc123",
  "lines": [
    {"time": 0, "text": "I'm in love with the shape of you"},
    {"time": 5000, "text": "We push and pull like a magnet do"},
    ...
  ]
}

Client:
- Fetch lyrics for current track
- Highlight line based on playback position
- Update every 100ms
\`\`\`

**Canvas** (Looping video art):
- Short looping videos for tracks
- Created by artists
- Displayed in Now Playing screen

---

## Technology Stack

### Backend

- **Microservices**: 100s of services
- **Java/Scala**: Primary backend languages
- **Python**: Data science, ML models
- **Google Cloud Platform (GCP)**: Primary cloud provider

### Data Storage

- **Cassandra**: Metadata (tracks, artists, albums)
- **PostgreSQL**: User data, playlists, subscriptions
- **Bigtable**: Analytics, event logs
- **Redis**: Caching, real-time data (now playing)

### Data Processing

- **Apache Kafka**: Event streaming (play events, user actions)
- **Apache Storm**: Real-time stream processing
- **Apache Spark**: Batch processing (recommendations, analytics)
- **Google Dataflow**: Managed stream/batch processing

### Machine Learning

- **TensorFlow**: Deep learning models
- **Apache Airflow**: ML pipeline orchestration
- **Luigi**: Workflow management (Spotify-built)

### Infrastructure

- **Kubernetes (GKE)**: Container orchestration
- **Istio**: Service mesh
- **gRPC**: Inter-service communication

---

## Key Lessons

### 1. Pre-Fetching Reduces Latency

Pre-fetching next track while current track plays enables seamless playback, zero interruption between tracks.

### 2. Offline Mode is Critical

Allowing users to download tracks for offline playback improves experience in low-connectivity scenarios (flights, commutes).

### 3. Personalization Drives Engagement

Discover Weekly, Daily Mixes, and Wrapped are hugely popular. Investment in ML for personalization pays off.

### 4. Social Features Amplify Discovery

Friend activity, collaborative playlists, Blend leverage social graph for music discovery.

### 5. Content Licensing is Complex

Rights management, royalty calculation, regional availability require sophisticated systems.

---

## Interview Tips

**Q: How would you design Spotify's audio streaming system?**

A: Use chunked HTTP streaming with adaptive bitrate. Store audio files in multiple encodings (128 kbps, 320 kbps) and formats (Ogg Vorbis, AAC) in GCS. Deliver via CDN (Google Cloud CDN + custom edge locations) with cache hit rate 90%+. Playback flow: User clicks play → Backend returns manifest (audio chunks) with CDN URLs → Client requests chunks sequentially → CDN serves from cache or fetches from origin. Pre-fetch next track while current track playing (seamless transition). Local cache stores recently played tracks (save bandwidth). Support offline mode (Premium users download tracks). Handle network changes with adaptive bitrate (switch quality based on bandwidth).

**Q: How does Spotify generate Discover Weekly?**

A: Use collaborative filtering + content-based filtering. Data: Billions of play events (user_id, track_id, timestamp, duration, skipped). Build user-track interaction matrix. Apply matrix factorization (SVD) to find latent factors (user preferences, track characteristics). For each user, find similar users (cosine similarity in latent space). Recommend tracks that similar users listened to but current user hasn't. Filter out already-heard tracks. Enrich with content-based features (audio features: tempo, energy, danceability from signal processing). Rank by predicted engagement. Generate weekly on Sunday night, deliver Monday morning. Continuously retrain model with new data. Handle cold start (new users) by asking favorite artists during onboarding.

**Q: How would you design Spotify's friend activity feature?**

A: Use WebSocket for real-time updates. When user plays track: (1) Client sends "now_playing" event to backend. (2) Backend updates user's current track in Redis (key: now_playing:user:123, value: {track_id, timestamp}, TTL: 5 minutes). (3) Backend looks up user's friends (social graph in PostgreSQL). (4) Backend pushes update to friends' WebSocket connections. (5) Friends' clients display "User X is listening to Track Y". Handle privacy: Users can opt out of sharing activity. Scale: Use pub/sub (Kafka) for high throughput. WebSocket servers subscribe to relevant topics (users they have connections to). Support offline: Store recent activity (last 30 minutes) for users who were offline.

---

## Summary

Spotify's architecture demonstrates building a music streaming platform at massive scale:

**Key Takeaways**:

1. **Chunked HTTP streaming**: Adaptive bitrate, pre-fetching, seamless playback
2. **CDN**: Global edge locations, cache popular tracks, 90%+ hit rate
3. **Recommendation ML**: Collaborative filtering + content-based, TensorFlow, continuous retraining
4. **Cassandra for metadata**: High read throughput, scalable, multi-region
5. **Social features**: WebSocket for real-time, Redis for now-playing state
6. **Playlists**: Core to UX, algorithmic (Discover Weekly), user-created, collaborative
7. **Offline mode**: Download for offline playback, critical for mobile
8. **Podcasts**: Separate pipeline, dynamic ad insertion, video support

Spotify's success comes from great audio delivery, ML-powered personalization, and social features that amplify music discovery.
`,
};
