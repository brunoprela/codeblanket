/**
 * Slack Architecture Section
 */

export const slackarchitectureSection = {
    id: 'slack-architecture',
    title: 'Slack Architecture',
    content: `Slack is a business communication platform that has become essential for modern workplace collaboration. With 18+ million daily active users across 750,000+ organizations, Slack's architecture must handle real-time messaging, file sharing, search, notifications, and integrations at massive scale while maintaining 99.99% uptime. This section explores the technical systems behind Slack.

## Overview

Slack's scale and challenges:
- **18+ million daily active users**
- **750,000+ organizations** (workspaces)
- **Billions of messages** sent daily
- **2.5+ billion monthly searches**
- **10+ million messages** sent per hour (peak)
- **99.99% uptime SLA** (50 minutes downtime per year max)
- **50+ milliseconds P50 latency** for message delivery

### Key Architectural Challenges

1. **Real-time messaging**: Instant message delivery to all participants
2. **Multi-tenancy**: Isolate 750K+ workspaces while sharing infrastructure
3. **Search**: Full-text search across billions of messages per workspace
4. **Presence**: Track online/offline status for millions of users
5. **File storage**: Handle billions of file uploads
6. **Integrations**: Support 2,400+ apps and bots
7. **Scalability**: Handle traffic spikes (Monday mornings, breaking news events)

---

## Evolution of Slack's Architecture

### Phase 1: Monolithic PHP Application (2013-2015)

Early Slack was built as a PHP monolith.

\`\`\`
Browser → Load Balancer → PHP App Servers (LAMP stack)
                               ↓
                         MySQL (single database)
                               ↓
                         Memcached (caching)
\`\`\`

**Technology Stack**:
- **PHP**: Web application (using HHVM for performance)
- **MySQL**: All data (users, workspaces, messages, channels)
- **Memcached**: Query result caching
- **WebSockets**: Real-time message delivery
- **Apache**: Web server

**Scaling Challenges** (2014-2015):
- MySQL write bottleneck (millions of messages/hour)
- Single database couldn't handle load
- Deployment slow (deploy entire monolith)
- Feature velocity limited (tight coupling)

---

### Phase 2: MySQL Sharding (2015-2017)

Slack sharded MySQL to handle write load.

**Sharding Strategy**: Shard by **workspace_id**

\`\`\`
Why shard by workspace_id?
- Most queries are workspace-specific
- All workspace data on same shard (data locality)
- No cross-shard queries for 95% of operations

Shard routing:
workspace_id → hash(workspace_id) % num_shards → Shard N
\`\`\`

**Vitess** (MySQL Sharding):

Slack adopted **Vitess** (created by YouTube/Google):
- Transparent sharding layer above MySQL
- Application thinks it's talking to single database
- Vitess routes queries to correct shard
- Supports online schema changes (no downtime)
- Horizontal scaling (add shards as data grows)

**Vitess Architecture**:
\`\`\`
Application → VTGate (query router)
                  ↓
         VTTablet (shard: workspace 1-10K)
         VTTablet (shard: workspace 10K-20K)
         VTTablet (shard: workspace 20K-30K)
              ↓
         MySQL Instances
\`\`\`

**Benefits**:
- Horizontal scalability (100+ shards)
- Online resharding (split hot shards)
- Connection pooling (reduce MySQL connections)
- Query rewriting (optimize for performance)

---

### Phase 3: Service-Oriented Architecture (2017-present)

Slack decomposed monolith into microservices.

**Core Services**:
- **Message Service**: Store and deliver messages
- **Channel Service**: Manage channels, memberships
- **User Service**: User profiles, authentication
- **Workspace Service**: Workspace settings, billing
- **Search Service**: Elasticsearch-based search
- **File Service**: File uploads, storage, serving
- **Notification Service**: Push notifications, email
- **Presence Service**: Online/offline status
- **Gateway Service**: WebSocket connections, message routing

**Communication**:
- **gRPC**: Inter-service communication (low latency, protocol buffers)
- **Kafka**: Event streaming (message events, audit logs)
- **Redis**: Pub/sub for real-time messaging

---

## Core Components

### 1. Real-Time Messaging Architecture

Slack's core feature is real-time messaging. Messages must appear instantly (<100ms).

**WebSocket Connection**:

Every Slack client maintains persistent WebSocket connection:

\`\`\`
Client (browser/mobile) → WebSocket → Gateway Server
                                           ↓
                                      Subscriptions
                                    (channels user is in)
\`\`\`

**Gateway Server**:
- Each server handles 10,000-50,000 WebSocket connections
- Written in Go (high concurrency, low memory)
- Stateful (knows which channels user is subscribed to)
- Auto-scaling (add servers during peak hours)

**Message Flow**:

\`\`\`
User A sends message to channel #engineering:

1. Client → WebSocket → Gateway Server A
2. Gateway A → Message Service (gRPC)
3. Message Service:
   - Validates message (permissions, content)
   - Generates message_id (unique, time-ordered)
   - Stores in MySQL (Vitess-sharded by workspace_id)
   - Indexes in Elasticsearch (for search)
   - Publishes to Redis Pub/Sub: channel:engineering:messages
4. All Gateway Servers subscribed to channel:engineering:messages
5. Gateway Servers push message to connected clients
   - Gateway A → User A (confirmation)
   - Gateway B → User B (real-time)
   - Gateway C → User C (real-time)
6. Clients display message

Latency breakdown:
- Client → Gateway: 10ms (network)
- Gateway → Message Service: 5ms (gRPC)
- Message Service processing: 20ms (DB write, Elasticsearch index)
- Redis Pub/Sub: 5ms (broadcast)
- Gateway → Client: 10ms (network)
Total: ~50ms P50, ~150ms P99
\`\`\`

**Scaling WebSocket Connections**:

**Challenge**: How to route messages to correct Gateway Server?

**Solution**: Redis Pub/Sub

\`\`\`
Gateway Servers subscribe to Redis channels:
- workspace:{id}:messages (all workspace messages)
- channel:{id}:messages (specific channel messages)
- user:{id}:direct_messages (DMs to user)

When message arrives:
1. Publish to Redis channel
2. All subscribed Gateway Servers receive message
3. Each Gateway checks: Do I have clients in this channel?
4. If yes, push to those clients via WebSocket
5. If no, ignore (no clients connected)
\`\`\`

**Presence and Heartbeats**:

\`\`\`
Client → Gateway: Heartbeat every 30 seconds
Gateway → Redis: SET presence:user:{id} = "online" (TTL: 60s)

If no heartbeat for 60s:
- Redis key expires → User marked offline
- Broadcast presence change to workspace members
\`\`\`

---

### 2. Message Storage

**Data Model** (MySQL via Vitess):

\`\`\`sql
Table: messages
- message_id (primary key, BIGINT)
  -- Generated by Snowflake-like ID generator
  -- Sortable by time: Higher ID = newer message
- workspace_id (shard key, indexed)
- channel_id (indexed)
- user_id (sender)
- text (VARCHAR 40,000)
  -- Slack messages limited to ~40K chars
- attachments (JSON)
  -- Files, images, links, embeds
- thread_ts (nullable, FK to parent message_id)
  -- For threaded conversations
- created_at (timestamp, indexed)
- updated_at (nullable, timestamp)
  -- For edited messages
- deleted_at (nullable, timestamp)
  -- Soft delete (messages not physically deleted immediately)

Indexes:
- PRIMARY KEY (message_id)
- INDEX (workspace_id, channel_id, created_at DESC)
  -- Most common query: Get recent messages in channel
- INDEX (workspace_id, user_id, created_at DESC)
  -- User's message history
- INDEX (workspace_id, thread_ts, created_at)
  -- Threaded conversations
\`\`\`

**Sharding Details**:

\`\`\`
Workspace 1-10,000 → Shard 1
Workspace 10,001-20,000 → Shard 2
...
Workspace 740,001-750,000 → Shard 75

Each shard is a MySQL instance (master + replicas)
All data for workspace on same shard (no distributed queries)
\`\`\`

**Message Retention**:

Different plans have different retention:
- **Free**: Last 90 days of messages
- **Pro**: Unlimited message history
- **Enterprise**: Unlimited + compliance exports

**Retention Implementation**:
\`\`\`
For Free workspaces:
- Background job runs nightly
- DELETE FROM messages 
  WHERE workspace_id IN (free_workspaces)
  AND created_at < NOW() - INTERVAL 90 DAY
  AND deleted_at IS NULL

- Also delete from Elasticsearch index
- Archive to S3 (cold storage) before deletion (compliance)
\`\`\`

---

### 3. Channels and Workspaces

**Multi-Tenancy Model**:

Each organization is a **workspace** (tenant):
- Isolated data (workspace 1 can't see workspace 2's data)
- Separate billing, settings, integrations
- Shared infrastructure (same servers, databases)

**Channel Types**:

**1. Public Channels**:
- Any workspace member can join
- Searchable within workspace
- Default: #general, #random

**2. Private Channels**:
- Invitation only
- Not searchable by non-members
- End-to-end encryption (optional, Enterprise Grid)

**3. Direct Messages (DMs)**:
- 1-on-1 conversations
- Or group DMs (up to 9 people)
- More private than channels

**Data Model**:

\`\`\`sql
Table: workspaces
- workspace_id (primary key)
- name (e.g., "Acme Corp")
- subdomain (e.g., "acme.slack.com")
- plan (free, pro, business, enterprise)
- created_at
- settings (JSON)

Table: channels
- channel_id (primary key)
- workspace_id (FK, indexed)
- name (e.g., "engineering", "general")
- type (public, private, dm, group_dm)
- topic (channel description)
- purpose (what channel is for)
- created_by (user_id)
- created_at
- is_archived (boolean)

Table: channel_members
- channel_id (FK, indexed)
- user_id (FK, indexed)
- workspace_id (denormalized for sharding)
- joined_at
- role (member, admin)
- notifications (all, mentions, nothing)

Composite index: (channel_id, user_id) for membership checks
Composite index: (workspace_id, user_id) for user's channels
\`\`\`

**Channel Membership Queries**:

\`\`\`sql
-- Get all channels user is in
SELECT c.* FROM channels c
JOIN channel_members cm ON c.channel_id = cm.channel_id
WHERE cm.user_id = ? AND c.workspace_id = ?

-- Get all members of channel
SELECT u.* FROM users u
JOIN channel_members cm ON u.user_id = cm.user_id
WHERE cm.channel_id = ? AND u.workspace_id = ?

-- Check if user in channel (permission check)
SELECT EXISTS(
  SELECT 1 FROM channel_members
  WHERE channel_id = ? AND user_id = ? AND workspace_id = ?
)
\`\`\`

---

### 4. Search

Slack's search is a key feature: Search across all messages, files, channels.

**Search Index**: **Elasticsearch**

**Why Elasticsearch?**
- Full-text search across billions of messages
- Fast (sub-100ms queries)
- Scalable (add nodes for capacity)
- Supports complex queries (filters, facets, highlighting)

**Index Structure**:

Each workspace has its own Elasticsearch index (tenant isolation):

\`\`\`json
Index: slack_workspace_12345
{
  "message_id": "1698158400123456",
  "workspace_id": "12345",
  "channel_id": "C123",
  "channel_name": "engineering",
  "user_id": "U456",
  "user_name": "John Doe",
  "text": "Deploying the new feature to production now",
  "attachments": [
    {"type": "file", "name": "deploy.sh"}
  ],
  "has_attachments": true,
  "has_links": true,
  "mentions": ["U789", "U101"],
  "timestamp": "2024-10-24T12:00:00Z",
  "thread_ts": null,
  "reactions": ["thumbsup", "rocket"]
}
\`\`\`

**Search Query Flow**:

\`\`\`
User searches: "deploy production"

1. Client → Search Service (REST API)
2. Search Service:
   - Identifies workspace_id from auth token
   - Identifies user's accessible channels (permission check)
   - Constructs Elasticsearch query:
   
{
  "index": "slack_workspace_12345",
  "query": {
    "bool": {
      "must": {
        "multi_match": {
          "query": "deploy production",
          "fields": ["text^2", "attachments.name"],
          "type": "best_fields"
        }
      },
      "filter": {
        "terms": {
          "channel_id": ["C123", "C456", "C789"]
          // Only channels user has access to
        }
      }
    }
  },
  "highlight": {
    "fields": {"text": {}}
  },
  "sort": [{"timestamp": "desc"}],
  "size": 20
}

3. Elasticsearch returns results
4. Fetch message metadata (user names, avatars) from cache/DB
5. Return to client with highlights
\`\`\`

**Search Features**:

**Filters**:
- **from:@john**: Messages from user John
- **in:#engineering**: Messages in #engineering channel
- **has:link**: Messages with links
- **has:file**: Messages with file attachments
- **before:2024-10-01**: Messages before date
- **after:2024-10-01**: Messages after date

**Autocomplete**:
- As user types, suggest completions
- Channels, users, common searches
- Powered by Elasticsearch suggest API

**Indexing Pipeline**:

\`\`\`
Message created:
1. Store in MySQL (durable)
2. Publish event to Kafka: message.created
3. Indexer service consumes Kafka
4. Index message in Elasticsearch
5. Message searchable within 1-5 seconds

Message edited:
1. Update in MySQL
2. Publish event: message.updated
3. Update Elasticsearch index

Message deleted:
1. Soft delete in MySQL (deleted_at = now)
2. Publish event: message.deleted
3. Remove from Elasticsearch index
\`\`\`

---

### 5. File Storage and Sharing

Slack handles billions of file uploads (images, documents, code snippets).

**File Upload Flow**:

\`\`\`
1. User uploads file (drag & drop or attach)
2. Client requests upload URL:
   POST /api/files.getUploadURLExternal
   Response: {
     "upload_url": "https://files.slack.com/upload/...",
     "file_id": "F123456"
   }

3. Client uploads file directly to S3 (presigned URL)
   - Chunked upload for large files (resumable)
   - Progress bar updates

4. Client notifies Slack backend: Upload complete
   POST /api/files.completeUploadExternal
   {
     "file_id": "F123456",
     "workspace_id": "W123",
     "channel_id": "C456"
   }

5. Backend:
   - Validates file uploaded to S3
   - Generates thumbnails (for images)
   - Virus scan (ClamAV)
   - Stores metadata in MySQL:
     * file_id, workspace_id, user_id
     * filename, size, mimetype
     * s3_key, thumbnail_url
     * created_at
   - Posts message to channel with file attachment
   - Indexes file in Elasticsearch (searchable by name)

6. Other users can download:
   - Click file → Request download URL
   - Backend generates presigned S3 URL (expires in 1 hour)
   - User downloads directly from S3
\`\`\`

**File Storage**:

\`\`\`
S3 organization:
s3://slack-files/
  workspace-12345/
    files/
      2024/10/24/
        F123456_original.png
        F123456_thumb_160.png
        F123456_thumb_360.png
        F123456_thumb_720.png
\`\`\`

**CDN**:
- Files served via CloudFront (AWS CDN)
- Edge caching (95%+ hit rate for popular files)
- HTTPS only (security)

**File Permissions**:
- Access controlled by channel membership
- Check: Is user in channel where file was shared?
- Enterprise: DLP (Data Loss Prevention) policies

---

### 6. Notifications

Slack notifications keep users engaged even when not active.

**Notification Types**:

**1. Push Notifications** (Mobile):
- New message in channel you're in
- @mention or direct message
- Keywords (user-defined)

**2. Email Notifications**:
- Digest emails (unread messages summary)
- @mentions when offline
- Configurable frequency

**3. Desktop Notifications**:
- Native OS notifications (macOS, Windows, Linux)
- Same triggers as push notifications

**Notification Service Architecture**:

\`\`\`
Message sent → Message Service
                   ↓
              Publish event: message.created (Kafka)
                   ↓
         Notification Service (consumer)
                   ↓
         Check: Should notify user?
           - Is user offline? (check presence in Redis)
           - Does user have notifications enabled for channel?
           - Is it an @mention or keyword match?
                   ↓
         If yes, send notification:
           - Push: APNs (iOS), FCM (Android)
           - Email: SendGrid/SES
           - Desktop: WebSocket to desktop app
\`\`\`

**Notification Preferences**:

Per-channel settings:
- **All messages**: Notify for every message
- **@mentions only**: Only @you or @channel
- **Nothing**: Mute channel

Global settings:
- **Do Not Disturb (DND)**: Schedule quiet hours (e.g., 10 PM - 8 AM)
- **Notification sound**: Choose sound or silent
- **Keywords**: Custom keywords trigger notifications

**Batching**:
- Don't send 100 push notifications for 100 messages
- Batch: "5 new messages in #engineering"
- Reduces notification fatigue

---

### 7. Threads

Threaded conversations reduce channel noise.

**How Threads Work**:

\`\`\`
Message A (parent):
  - Message ID: 1698158400000000
  - Thread ID: NULL (not in thread)

User replies to Message A:
  - Message B (child):
    - Message ID: 1698158460000000
    - Thread ID: 1698158400000000 (parent's ID)

Message C (another reply):
  - Message ID: 1698158520000000
  - Thread ID: 1698158400000000

Thread structure:
Message A (parent)
  ├─ Message B (reply 1)
  └─ Message C (reply 2)
\`\`\`

**Data Model**:

\`\`\`sql
Table: messages
- message_id
- thread_ts (timestamp of parent message)
  -- NULL for top-level messages
  -- Set for replies

Queries:
-- Get parent message
SELECT * FROM messages WHERE message_id = ?

-- Get all replies in thread
SELECT * FROM messages 
WHERE thread_ts = ? AND workspace_id = ?
ORDER BY created_at ASC

-- Count replies in thread
SELECT COUNT(*) FROM messages
WHERE thread_ts = ? AND workspace_id = ?
\`\`\`

**Thread Notifications**:
- User can "follow" thread (get notified of new replies)
- Auto-follow if user participates in thread
- Unfollow option available

---

### 8. Integrations and Apps

Slack supports 2,400+ apps (bots, integrations, workflows).

**Slack App Types**:

**1. Bots**:
- Automated users that respond to messages
- Examples: GitHub bot, Jira bot, Polly polls

**2. Slash Commands**:
- Custom commands: /giphy, /remind, /poll
- Trigger workflows or external APIs

**3. Interactive Components**:
- Buttons, menus, dialogs in messages
- User interaction triggers webhooks

**4. Workflows**:
- No-code automation (like Zapier)
- Trigger: New message, scheduled time
- Actions: Post message, create ticket, send email

**Integration Architecture**:

\`\`\`
Slack App → OAuth 2.0 → Slack API
                              ↓
                      Workspace grants permissions
                              ↓
                      App receives OAuth token
                              ↓
                      App makes API calls (with token)

Incoming Webhooks:
External Service → POST to webhook URL → Slack posts message

Outgoing Webhooks:
Slack message → If matches trigger → POST to external URL → External service responds
\`\`\`

**API Rate Limiting**:

\`\`\`
Per-app rate limits:
- Tier 1: 1 request/second
- Tier 2: 20 requests/second
- Tier 3: 50 requests/second
- Tier 4: 100+ requests/second (approved apps)

Implementation:
- Token bucket algorithm
- Tracked in Redis (per app token)
- HTTP 429 response when limit exceeded
- Retry-After header indicates wait time
\`\`\`

---

## Technology Stack

### Backend

- **PHP**: Original monolith (HHVM for performance)
- **Go**: Gateway servers (WebSocket handling)
- **Java**: Some services (search, analytics)
- **Python**: Data processing, ML models

### Data Storage

- **MySQL**: Primary database (via Vitess sharding)
- **Redis**: Caching, pub/sub, presence, rate limiting
- **Memcached**: Additional caching layer
- **Elasticsearch**: Search index (messages, files)
- **S3**: File storage (images, documents)

### Data Processing

- **Kafka**: Event streaming (message events, audit logs)
- **Spark**: Batch processing (analytics, ML training)
- **Flink**: Stream processing (real-time metrics)

### Infrastructure

- **AWS**: Primary cloud provider
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code
- **Consul**: Service discovery

### Monitoring

- **Datadog**: Metrics, logs, APM
- **PagerDuty**: On-call, alerting
- **Sentry**: Error tracking
- **Custom**: Internal monitoring tools

---

## Key Lessons from Slack Architecture

### 1. Vitess Enables MySQL Sharding

Vitess provides transparent sharding layer above MySQL. Applications don't need sharding logic. Online resharding, schema changes without downtime.

### 2. Redis Pub/Sub for Real-Time Broadcasting

WebSocket Gateway Servers subscribe to Redis channels. Messages published once, all subscribed gateways receive. Scales better than direct server-to-server communication.

### 3. Shard by Workspace ID

Most queries workspace-specific. Sharding by workspace_id keeps related data together, avoids cross-shard queries.

### 4. Elasticsearch for Full-Text Search

MySQL insufficient for full-text search at scale. Elasticsearch provides fast, scalable search with complex query support.

### 5. Multi-Tenancy Requires Isolation

Each workspace isolated (separate Elasticsearch index, access controls). Shared infrastructure (servers, databases) for cost efficiency.

### 6. Notifications Must Be Smart

Don't bombard users. Respect preferences, batch notifications, implement DND. Notification fatigue kills engagement.

---

## Interview Tips

**Q: How would you design Slack's real-time messaging system?**

A: Use WebSocket for persistent connections. Architecture: (1) Gateway servers (Go) handle WebSocket connections (10K-50K per server). (2) User sends message → Gateway forwards to Message Service (gRPC). (3) Message Service validates, stores in MySQL (Vitess-sharded by workspace_id), indexes in Elasticsearch. (4) Publish to Redis Pub/Sub (channel:messages). (5) All Gateway servers subscribed to channel, receive message. (6) Gateway servers push to connected WebSocket clients. (7) Clients display message. Handle scale: Horizontal Gateway scaling, Redis Pub/Sub for broadcast, Vitess sharding for storage. Latency: <50ms P50 (10ms network + 5ms Gateway→Service + 20ms DB write + 5ms Redis + 10ms Gateway→Client).

**Q: How does Slack implement search across billions of messages?**

A: Use Elasticsearch with per-workspace indexes. Index structure: messages indexed with text, user, channel, timestamp, attachments. Query flow: (1) User searches "deploy production." (2) Search Service identifies workspace_id from auth token. (3) Permission check: Get user's accessible channels. (4) Elasticsearch query with multi_match on text field, filtered by channel_ids user can access. (5) Sort by timestamp DESC, return top 20 with highlights. (6) Fetch user metadata (names, avatars) from cache. Indexing pipeline: Message created → Store in MySQL → Publish to Kafka → Indexer consumes Kafka → Index in Elasticsearch (1-5 second delay acceptable). Scalability: Add Elasticsearch nodes, shard index by date (older messages in different shards).

**Q: How does Slack handle multi-tenancy (750K+ workspaces)?**

A: Shard by workspace_id for data isolation and performance. MySQL: Vitess shards by workspace_id, all workspace data on same shard (data locality, no cross-shard queries). Elasticsearch: Separate index per workspace (tenant isolation, security). Gateway servers: Track user's workspace in session, route messages to correct shard. Billing: Per-workspace plans (free, pro, enterprise), tracked in workspaces table. API rate limiting: Per-workspace quotas in Redis. Benefits: (1) Isolation - workspace 1 can't see workspace 2's data. (2) Performance - related data together. (3) Scalability - add shards as workspaces grow. Trade-off: Cross-workspace features (Slack Connect) require special handling.

---

## Summary

Slack's architecture handles real-time business communication at massive scale:

**Key Takeaways**:

1. **Vitess-sharded MySQL**: Horizontal scaling, shard by workspace_id, online schema changes
2. **WebSocket + Redis Pub/Sub**: Real-time message delivery, Gateway servers subscribe to channels
3. **Elasticsearch per workspace**: Full-text search, tenant isolation, fast queries (<100ms)
4. **File storage**: S3 for storage, CloudFront CDN for delivery, presigned URLs for security
5. **Notifications**: Push (APNs/FCM), email, desktop, smart batching, respect preferences
6. **Multi-tenancy**: Workspace isolation at data layer, shared infrastructure for efficiency
7. **Integrations**: OAuth 2.0 for apps, webhooks for events, rate limiting per app
8. **Microservices**: Independent scaling, gRPC for communication, Kafka for events

Slack's success from reliable real-time delivery, comprehensive search, and developer-friendly integrations enabling ecosystem of 2,400+ apps.
`,
};
