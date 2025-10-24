/**
 * Design Facebook Messenger Section
 */

export const facebookMessengerSection = {
  id: 'facebook-messenger',
  title: 'Design Facebook Messenger',
  content: `Facebook Messenger is a real-time messaging platform supporting 1-on-1 chats, group messaging, read receipts, typing indicators, and media sharing. The core challenge is delivering messages instantly with high availability and strong consistency guarantees.

## Problem Statement

Design a messaging service that supports:
- **1-on-1 messaging**: Private conversations between two users
- **Group chat**: Conversations with multiple participants
- **Real-time delivery**: Messages delivered within milliseconds
- **Read receipts**: Show when messages are read
- **Typing indicators**: Show when users are typing
- **Media sharing**: Photos, videos, voice messages
- **Message history**: Persistent storage and retrieval
- **Online/offline status**: User presence tracking

**Scale**: 1 billion users, 100 billion messages/day

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Send Message**: User sends text/media to another user or group
2. **Receive Message**: Real-time delivery to online users
3. **Message History**: Retrieve past conversations
4. **Read Receipts**: Track message delivery and read status
5. **Typing Indicator**: Show when users are typing
6. **Group Chat**: Up to 256 participants per group
7. **Online Status**: Show user online/offline/last seen
8. **Push Notifications**: Notify offline users

### Non-Functional Requirements

1. **Low Latency**: < 100ms message delivery for online users
2. **High Availability**: 99.99% uptime
3. **Strong Consistency**: No message loss or duplication
4. **Scalable**: Handle billions of messages per day
5. **Global**: Low latency worldwide
6. **Secure**: End-to-end encryption (optional)

### Out of Scope

- Video calls
- Stories
- Payment integration

---

## Step 2: Capacity Estimation

### Traffic

**Messages**:
- 100B messages/day = ~1.16M messages/sec
- Peak: ~2M messages/sec

**Connections**:
- 1B users, 10% concurrently active = 100M concurrent WebSocket connections
- Each connection: ~5 KB memory = 500 GB total connection state

### Storage

**Messages**:
- 100B messages/day
- Average message: 200 bytes (text + metadata)
- 100B × 200 bytes = 20 TB/day
- 5 years: 20 TB × 365 × 5 = 36.5 PB

**Media**:
- 20% of messages include media (avg 500 KB)
- 20B media messages/day × 500 KB = 10 PB/day
- Store in S3, serve via CDN

### Bandwidth

**Message throughput**:
- 1.16M messages/sec × 200 bytes = 232 MB/sec

**WebSocket bandwidth**:
- 100M connections × 1 KB/sec (heartbeats, presence) = 100 GB/sec

---

## Step 3: System API Design

### WebSocket API (Primary)

**Connect**:
\`\`\`
WS /ws/connect?user_id=123&token=xyz

Server → Client:
{
  "type": "connected",
  "user_id": 123,
  "session_id": "abc123"
}
\`\`\`

**Send Message**:
\`\`\`
Client → Server:
{
  "type": "send_message",
  "message_id": "msg_123",
  "conversation_id": "conv_456",
  "recipient_ids": [789],
  "text": "Hello!",
  "timestamp": 1698160000
}

Server → Client (Acknowledgment):
{
  "type": "ack",
  "message_id": "msg_123",
  "status": "delivered",
  "server_timestamp": 1698160001
}

Server → Recipient:
{
  "type": "new_message",
  "message_id": "msg_123",
  "sender_id": 123,
  "text": "Hello!",
  "timestamp": 1698160001
}
\`\`\`

**Read Receipt**:
\`\`\`
Client → Server:
{
  "type": "mark_read",
  "message_id": "msg_123"
}

Server → Sender:
{
  "type": "message_read",
  "message_id": "msg_123",
  "reader_id": 789
}
\`\`\`

### REST API (Fallback)

**Get Conversation History**:
\`\`\`
GET /api/v1/conversations/{conv_id}/messages?limit=50&before={msg_id}

Response:
{
  "messages": [
    {
      "message_id": "msg_123",
      "sender_id": 123,
      "text": "Hello!",
      "timestamp": 1698160000,
      "status": "read"
    }
  ]
}
\`\`\`

---

## Step 4: Database Schema

### Messages Table (Cassandra)

\`\`\`cql
CREATE TABLE messages (
    conversation_id TEXT,
    message_id TIMEUUID,
    sender_id BIGINT,
    text TEXT,
    media_urls LIST<TEXT>,
    timestamp TIMESTAMP,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
\`\`\`

**Why Cassandra?**
- Optimized for write-heavy workloads (1M+ writes/sec)
- Time-series data (messages ordered by time)
- Horizontal scaling to petabytes
- Multi-datacenter replication

### Conversations Table

\`\`\`sql
CREATE TABLE conversations (
    conversation_id VARCHAR(50) PRIMARY KEY,
    participant_ids JSON,  -- [123, 456, 789]
    conversation_type ENUM('direct', 'group'),
    created_at TIMESTAMP,
    last_message_id VARCHAR(50),
    last_message_timestamp TIMESTAMP
);
\`\`\`

### User Connections (Redis)

\`\`\`
Key: user:{user_id}:connections
Value: Set of WebSocket connection IDs
\`\`\`

---

## Step 5: High-Level Architecture

\`\`\`
┌──────────────┐
│Mobile/Web App│
└──────┬───────┘
       │ WebSocket
       ▼
┌──────────────────────────────────────┐
│    WebSocket Gateway (Stateful)      │
│  100M concurrent connections         │
│  Partitioned by user_id hash         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│    Message Service (Stateless)       │
│  Validation, routing, persistence    │
└───┬──────────────────────┬───────────┘
    │                      │
    ▼                      ▼
┌───────────┐     ┌────────────────┐
│   Kafka   │     │ Cassandra DB   │
│  (Queue)  │     │   (Messages)   │
└─────┬─────┘     └────────────────┘
      │
      ▼
┌────────────────┐
│ Delivery Worker│
└────────────────┘
\`\`\`

---

## Step 6: Message Send Flow

**1-on-1 Message Flow**:
\`\`\`
1. Alice sends message to Bob via WebSocket
2. WebSocket Gateway receives message
3. Forward to Message Service
4. Message Service:
   a. Validate message (auth, spam check)
   b. Generate server_timestamp
   c. Write to Cassandra (conversation_id, message)
   d. Publish to Kafka topic: "user:bob:messages"
   e. Return ACK to Alice (delivered to server)
5. Delivery Worker consumes from Kafka
6. Check Redis: Is Bob online?
   → Online: Get Bob's WebSocket connection IDs
   → Send message via WebSocket
   → Bob receives message, sends read receipt
7. If Bob offline:
   → Store in inbox (Redis list)
   → Send push notification
   → Bob fetches on next connection
\`\`\`

### Group Message Flow

**Challenge**: Send to 100 participants efficiently

\`\`\`
1. Alice sends message to Group (100 members)
2. Message Service:
   a. Write message once to Cassandra
   b. Publish 100 events to Kafka (one per member)
   c. Kafka partitions distribute load
3. Delivery Workers (100 parallel):
   - Each worker handles subset of recipients
   - Check online status
   - Deliver via WebSocket or queue for offline
4. Fanout complete in < 100ms
\`\`\`

---

## Step 7: WebSocket Gateway Design

**Challenge**: Maintain 100M concurrent WebSocket connections

### Connection Management

**Stateful Gateway Servers**:
- Each gateway server handles 100K connections
- Need 1,000 gateway servers total
- Connections partitioned by \`user_id % NUM_GATEWAYS\`

**Connection State (per server)**:
\`\`\`typescript
class WebSocketGateway {
    connections: Map<user_id, WebSocket[]> = new Map();
    
    onConnect(userId: number, ws: WebSocket) {
        // Store connection
        if (!this.connections.has(userId)) {
            this.connections.set(userId, []);
        }
        this.connections.get(userId).push(ws);
        
        // Register in Redis (for routing)
        redis.sadd(\`user:\${userId}:gateways\`, this.serverId);
    }
    
    onDisconnect(userId: number, ws: WebSocket) {
        // Remove connection
        const userConns = this.connections.get(userId);
        userConns.splice(userConns.indexOf(ws), 1);
        
        if (userConns.length === 0) {
            redis.srem(\`user:\${userId}:gateways\`, this.serverId);
        }
    }
}
\`\`\`

### Message Routing

**How to route message to correct gateway?**

\`\`\`
1. Message Service needs to deliver to Bob (user_id=789)
2. Query Redis: SMEMBERS user:789:gateways
3. Result: [gateway-5, gateway-12] (Bob has 2 devices connected)
4. Send RPC to gateway-5 and gateway-12
5. Gateways deliver to Bob's WebSocket connections
\`\`\`

---

## Step 8: Read Receipts & Typing Indicators

### Read Receipts

**Flow**:
\`\`\`
1. Bob receives message, displays it
2. Bob sends: {type: "mark_read", message_id: "msg_123"}
3. WebSocket Gateway forwards to Message Service
4. Message Service:
   a. Update message status in Cassandra (async, batch)
   b. Send read receipt to Alice via WebSocket
5. Alice sees "Read" indicator
\`\`\`

**Optimization**: Batch read receipts every 5 seconds to reduce DB writes.

### Typing Indicators

**Flow**:
\`\`\`
1. Alice starts typing
2. Client sends: {type: "typing_start", conversation_id: "conv_123"}
3. Gateway broadcasts to other participants (Bob) immediately
4. No database write (ephemeral state)
5. After 3 seconds of inactivity:
   {type: "typing_stop"}
\`\`\`

**Why no DB?** Typing is transient - no need to persist.

---

## Step 9: Online/Offline Status

**Presence Tracking**:

\`\`\`
1. User connects → WebSocket Gateway
2. Gateway updates Redis:
   SET user:123:online "true" EX 300  (5-min TTL)
3. Heartbeat every 60 seconds resets TTL
4. If disconnect or heartbeat stops:
   - Redis key expires after 5 minutes
   - User shown as offline
5. Query presence:
   GET user:123:online → "true"/"false"
   GET user:123:last_seen → timestamp
\`\`\`

**Last Seen**:
\`\`\`
On disconnect:
SET user:123:last_seen {current_timestamp}
\`\`\`

---

## Step 10: Message History & Pagination

**Query Conversation**:
\`\`\`sql
SELECT * FROM messages
WHERE conversation_id = 'conv_123'
AND message_id < {last_message_id}
ORDER BY message_id DESC
LIMIT 50;
\`\`\`

**Cassandra Benefits**:
- Partition by conversation_id (all messages for conversation on same node)
- Clustering by message_id (time-ordered)
- Fast range queries for pagination

**Cursor-based Pagination**:
\`\`\`
GET /messages?conversation_id=123&before=msg_xyz&limit=50
\`\`\`

---

## Step 11: Consistency & Reliability

### At-Least-Once Delivery

**Guarantees**:
1. Message written to Cassandra (durable)
2. Kafka retries delivery on failure
3. Recipient may receive duplicate (client deduplication via message_id)

**Client Handling**:
\`\`\`typescript
const seenMessages = new Set<string>();

function onMessage(message: Message) {
    if (seenMessages.has(message.message_id)) {
        return; // Duplicate, ignore
    }
    seenMessages.add(message.message_id);
    displayMessage(message);
}
\`\`\`

### Message Ordering

**Within Conversation**: Cassandra clustering ensures order.
**Across Shards**: Use TIMEUUID (timestamp + UUID) for total ordering.

---

## Step 12: Optimizations

### 1. Message Compression

Compress message batches over WebSocket (gzip):
- Reduces bandwidth by 60-70%
- Critical for mobile users

### 2. Connection Pooling

Reuse WebSocket connections for multiple conversations:
- Single WebSocket for all user's chats
- Multiplexing via conversation_id

### 3. Offline Message Inbox

Store offline messages in Redis list:
\`\`\`
LPUSH inbox:user_123 {message_json}
LTRIM inbox:user_123 0 999  (keep last 1000)
\`\`\`

On reconnect, drain inbox:
\`\`\`
LRANGE inbox:user_123 0 -1
DEL inbox:user_123
\`\`\`

---

## Trade-offs

### WebSocket vs HTTP Long Polling

**WebSocket** (Chosen):
- ✅ True real-time (< 10ms delivery)
- ✅ Lower overhead (no repeated handshakes)
- ❌ Stateful (requires connection management)

**HTTP Long Polling**:
- ✅ Stateless, simpler
- ❌ Higher latency (100-500ms)
- ❌ More overhead (repeated requests)

### Kafka vs Direct Delivery

**Kafka** (Chosen):
- ✅ Reliability (durable queue)
- ✅ Decouples sender from delivery
- ❌ Adds latency (~50ms)

**Direct**:
- ✅ Lower latency
- ❌ Message loss if recipient gateway is down

---

## Interview Tips

### What to Clarify

1. **Scale**: How many users? Messages per day?
2. **Features**: Group chat? Read receipts? Media?
3. **Latency**: Real-time requirement (< 100ms)?
4. **Consistency**: Can messages be lost/duplicated?

### What to Emphasize

1. **WebSocket for real-time**: Explain bidirectional communication
2. **Cassandra for messages**: Write-heavy, time-series data
3. **Kafka for reliability**: Async delivery queue
4. **Stateful gateway**: Connection management, routing

### Common Mistakes

1. ❌ Using HTTP polling for real-time (too slow)
2. ❌ SQL database for messages (doesn't scale to petabytes)
3. ❌ Synchronous delivery (blocks sender)
4. ❌ Not handling offline users

### Follow-up Questions

- "How do you handle message encryption?"
- "How would you implement disappearing messages (24h expiry)?"
- "What if a gateway server crashes with 100K connections?"
- "How do you detect and prevent spam?"

---

## Summary

**Core Components**:
1. **WebSocket Gateway**: Handle 100M concurrent connections (stateful)
2. **Message Service**: Validation, persistence, routing (stateless)
3. **Kafka**: Reliable async message delivery queue
4. **Cassandra**: Store 36.5 PB of messages (write-optimized)
5. **Redis**: Connection routing, presence, offline inbox
6. **Push Notification Service**: Alert offline users

**Key Decisions**:
- ✅ WebSocket for real-time bidirectional communication
- ✅ Cassandra for write-heavy time-series message storage
- ✅ Kafka for reliable async delivery
- ✅ Partitioned stateful gateways for connection management
- ✅ At-least-once delivery with client deduplication

**Capacity**:
- 100 billion messages/day (1.16M/sec)
- 100 million concurrent WebSocket connections
- 36.5 PB message storage over 5 years
- < 100ms real-time delivery for online users

This design handles **WhatsApp/Messenger-scale messaging** with **sub-100ms latency** and **99.99% availability** using battle-tested real-time communication patterns.`,
};
