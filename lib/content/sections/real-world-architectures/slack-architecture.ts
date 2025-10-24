/**
 * Slack Architecture Section
 */

export const slackarchitectureSection = {
    id: 'slack-architecture',
    title: 'Slack Architecture',
    content: `Slack is a business communication platform with 18+ million daily active users across 750,000+ organizations. Slack's architecture must handle real-time messaging, file sharing, search, and integrations at scale while ensuring high availability and low latency.

## Overview

**Scale**: 18M+ daily active users, 750K+ organizations, billions of messages daily, 99.99% uptime SLA

**Key Challenges**: Real-time messaging, message storage, search, presence, integrations, multi-tenancy

## Core Components

### 1. Real-Time Messaging

**WebSocket Connection**:
- Persistent WebSocket per client
- Server pushes messages instantly
- Connection pooling: Each server handles 10K+ connections

**Message Flow**:
\`\`\`
User A sends message to channel:
1. Client → WebSocket → API server
2. Store message in database (MySQL)
3. Publish to Redis pub/sub (channel:{id}:messages)
4. All clients subscribed to channel receive message
5. Display message in real-time
\`\`\`

**Architecture**:
\`\`\`
Client WebSocket → Gateway Servers (handle WebSocket connections)
                        ↓
                   Message Queue (Redis/RabbitMQ)
                        ↓
                   Workers (process messages, store in DB)
                        ↓
                   MySQL (sharded by workspace_id)
\`\`\`

---

### 2. Message Storage

**Database**: MySQL (Vitess-sharded)

**Data Model**:
\`\`\`sql
Table: messages
- message_id (primary key)
- workspace_id (shard key)
- channel_id (indexed)
- user_id
- text
- attachments (JSON)
- thread_id (for threaded replies)
- timestamp
- edited_at
- deleted (soft delete)

Index: (workspace_id, channel_id, timestamp)
\`\`\`

**Vitess** (MySQL sharding):
- Shard by workspace_id (all workspace data on same shard)
- Horizontal scaling
- Online schema changes
- Query routing

**Message Retention**:
- Free plans: 90 days
- Paid plans: Unlimited
- Archived messages moved to cold storage (S3)

---

### 3. Search

**Elasticsearch** for full-text search across messages, files, channels.

**Index Structure**:
\`\`\`json
{
  "message_id": "123",
  "workspace_id": "ws_456",
  "channel_id": "ch_789",
  "user_id": "u_999",
  "text": "Let's deploy the new feature tomorrow",
  "timestamp": 1698158400,
  "has_attachments": true,
  "mentions": ["u_111", "u_222"],
  "reactions": ["thumbsup", "rocket"]
}
\`\`\`

**Search Query**:
\`\`\`
User searches: "deploy feature"
- Full-text search on text field
- Filter by workspace_id (security)
- Optional filters: channel, date range, from user
- Rank by relevance + recency
- Highlight matches
\`\`\`

**Indexing**:
- New messages indexed in real-time (via message queue)
- Bulk indexing for historical messages
- Per-workspace indexes (isolation)

---

### 4. File Sharing

**File Upload**:
\`\`\`
1. User uploads file
2. Chunked upload to S3
3. Generate presigned URL
4. Store metadata in MySQL (file_id, workspace_id, uploader, size)
5. Post message with file attachment
\`\`\`

**File Download**:
- Presigned S3 URLs (temporary, authenticated)
- CDN (CloudFront) for popular files
- Access control: Check user's workspace membership

**File Types**:
- Images: Generate thumbnails
- Documents: Preview generation (PDFs, Office docs)
- Code snippets: Syntax highlighting

---

### 5. Presence (Online Status)

**Presence System**:
\`\`\`
User online:
- WebSocket heartbeat every 30 seconds
- Store in Redis: presence:{user_id} = "online"
- TTL: 60 seconds (auto-expires if no heartbeat)

User inactive (away):
- No activity for 5 minutes → "away"

User offline:
- WebSocket disconnects → "offline"
\`\`\`

**Broadcast Presence**:
- User's status changes → Publish to Redis pub/sub
- All workspace members receive update
- Update UI (green/yellow/gray dot)

---

### 6. Channels and Workspaces

**Multi-Tenancy**:
- Each organization is a workspace
- Workspace data isolated (separate DB shard)
- Channels within workspace

**Channel Types**:
- **Public**: All workspace members can join
- **Private**: Invitation only
- **Direct Messages (DMs)**: 1-on-1 or groups

**Data Model**:
\`\`\`sql
Table: workspaces
- workspace_id (primary key)
- name
- created_at

Table: channels
- channel_id (primary key)
- workspace_id (FK)
- name
- type (public, private, dm)
- members (array or join table)

Table: workspace_members
- workspace_id
- user_id
- role (admin, member, guest)
- joined_at
\`\`\`

---

### 7. Integrations and Apps

**Slack Apps**:
- Bots, slash commands, interactive components
- OAuth for authentication
- Webhooks for posting messages
- APIs for reading/writing data

**Architecture**:
\`\`\`
External App → Slack API
                  ↓
             Validate OAuth token
                  ↓
             Check permissions (scopes)
                  ↓
             Execute action (post message, read channel)
\`\`\`

**Rate Limiting**:
- Per app: 1 req/second (tier 1), 20 req/second (tier 2)
- Per workspace: Higher limits
- Token bucket algorithm (Redis)

---

### 8. Notifications

**Push Notifications**:
- Message in channel user hasn't read → Push notification
- @mentions → Always notify
- Keywords: User-defined (e.g., "urgent", "bug")

**Email Notifications**:
- Digest emails (daily summary)
- Immediate for @mentions (if offline)

**Notification Preferences**:
- Per-channel: All messages, @mentions only, nothing
- Do Not Disturb (DND): Schedule quiet hours
- Mobile vs desktop: Different settings

---

### 9. Threads

**Threaded Conversations**:
- Reply to message → Creates thread
- All replies grouped under parent message
- Reduces channel noise

**Data Model**:
\`\`\`sql
Table: messages
- message_id
- thread_id (nullable, FK to parent message_id)
- is_thread_parent (boolean)

Queries:
- Get thread: WHERE thread_id = ? ORDER BY timestamp
- Get parent: WHERE message_id = thread_id
\`\`\`

---

## Technology Stack

**Backend**: PHP (original), migrating to Go and Scala
**Data**: MySQL (Vitess for sharding), Redis, Memcached
**Search**: Elasticsearch
**Storage**: S3 (files), CloudFront (CDN)
**Messaging**: Redis pub/sub, RabbitMQ
**Infrastructure**: AWS, Kubernetes
**Monitoring**: Datadog, PagerDuty

---

## Key Lessons

1. **WebSocket** for real-time messaging, persistent connections
2. **Vitess** enables MySQL sharding by workspace (horizontal scaling)
3. **Redis pub/sub** for message broadcasting to WebSocket clients
4. **Elasticsearch** for full-text search across messages, files
5. **Multi-tenancy**: Workspace isolation at database level (sharding)

---

## Interview Tips

**Q: How would you design Slack's real-time messaging system?**

A: Use WebSocket for persistent connections. Architecture: (1) Gateway servers handle WebSocket connections (10K+ per server). (2) User sends message → Gateway forwards to message queue (Redis/RabbitMQ). (3) Workers consume queue, store message in MySQL (Vitess-sharded by workspace_id). (4) Publish to Redis pub/sub channel (channel:{id}:messages). (5) Gateway servers subscribed to pub/sub, push message to relevant WebSocket clients. (6) Clients display message in real-time. Handle scale: Horizontal scaling of gateways and workers. Persistence: MySQL for messages, Elasticsearch for search. Presence: Heartbeats (30s intervals), Redis with TTL (60s). Message ordering: Timestamp-based, server generates timestamp (avoid clock skew).

---

## Summary

Slack's architecture handles real-time business communication at scale: WebSocket for instant messaging, Vitess-sharded MySQL for message storage, Redis pub/sub for broadcasting, Elasticsearch for search, S3 for files. Success from reliable real-time delivery, multi-tenant isolation, and comprehensive search across conversations.
`,
};

