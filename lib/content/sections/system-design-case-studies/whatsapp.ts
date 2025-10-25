/**
 * Design WhatsApp Section
 */

export const whatsappSection = {
  id: 'whatsapp',
  title: 'Design WhatsApp',
  content: `WhatsApp is a global messaging platform with 2 billion users, handling 100 billion messages per day. The key challenges are: real-time message delivery, end-to-end encryption, supporting text/media/voice messages, group chat, and maintaining extreme reliability with minimal infrastructure (famous for running with 50 engineers for 900M users).

## Problem Statement

Design WhatsApp with:
- **Send Messages**: Text, images, videos, voice messages
- **1-on-1 Chat**: Private conversations
- **Group Chat**: Up to 256 participants
- **End-to-End Encryption**: Messages encrypted, only sender/receiver can read
- **Message Delivery**: Delivered even if recipient offline
- **Read Receipts**: Blue checkmarks when read
- **Status Updates**: 24-hour ephemeral stories
- **Voice/Video Calls**: Real-time communication

**Scale**: 2B users, 100B messages/day, 100M concurrent users

---

## Step 1: Requirements

### Functional Requirements
1. **Send/Receive Messages**: Text, media (photos, videos, voice notes)
2. **Delivery Status**: Single checkmark (sent), double checkmark (delivered), blue checkmarks (read)
3. **Group Messaging**: Create groups, add/remove members, group admin
4. **Media Sharing**: Photos (10 MB), Videos (100 MB), Voice (10 min)
5. **End-to-End Encryption**: Signal Protocol
6. **Offline Messages**: Store and deliver when user comes online
7. **Multi-Device**: Sync across phone, web, desktop
8. **Voice/Video Calls**: Real-time P2P or relayed calls

### Non-Functional Requirements
1. **High Availability**: 99.99% uptime
2. **Low Latency**: < 100ms message delivery
3. **Strong Consistency**: No message loss or duplication
4. **Scalable**: 100B messages/day
5. **End-to-End Encrypted**: Zero-knowledge encryption
6. **Efficient**: Minimal battery/data usage

---

## Step 2: Capacity Estimation

**Messages**: 100B/day = 1.16M messages/sec average, 3M peak
**Users**: 2B total, 100M concurrent
**Storage**: 100B messages/day × 1 KB avg = 100 TB/day
**Media**: 20% of messages = 20B media/day × 500 KB = 10 PB/day

---

## Step 3: Architecture (Simplified WhatsApp Approach)

WhatsApp is famous for simplicity and efficiency. Key decisions:
- **Erlang**: Concurrency, fault tolerance, millions of connections per server
- **FreeBSD**: Minimalist, high-performance OS
- **Single Server Architecture**: Initially ran on single servers (vertical scaling)
- **Eventually Consistent**: Message ordering can have minor delays

\`\`\`
[Client Device]
     ↓ TCP/WebSocket (custom protocol)
[WhatsApp Gateway Servers]
     ↓
[Message Router]
     ↓
[Message Queue (per user)]
     ↓
[Recipient Device]

Storage: Cassandra (messages, metadata)
Cache: Redis (online users, message queues)
Media: S3 (photos, videos)
\`\`\`

---

## Step 4: Message Flow

**Send Message Flow**:
\`\`\`
1. Alice sends message to Bob
2. Client encrypts message with Bob\'s public key (E2E encryption)
3. Send to WhatsApp server via persistent TCP connection
4. Server validates sender, checks Bob's status
5. If Bob online:
   - Push message directly to Bob's device
   - Bob\'s device decrypts with private key
   - Send delivery acknowledgment
6. If Bob offline:
   - Store encrypted message in Bob's inbox (Redis/Cassandra)
   - Send push notification
   - Bob retrieves when online
7. Delivery confirmations:
   - Single check: Message sent to server
   - Double check: Message delivered to Bob's device
   - Blue checks: Bob read the message
\`\`\`

---

## Step 5: End-to-End Encryption (Signal Protocol)

**Key Exchange** (one-time setup):
\`\`\`
1. Alice and Bob both generate key pairs:
   - Identity key (long-term)
   - Signed pre-key (medium-term, rotated weekly)
   - One-time pre-keys (ephemeral, 100 generated)

2. Alice wants to message Bob (first time):
   - Alice\'s device requests Bob's public keys from server
   - Server sends: Bob's identity key, signed pre-key, one-time pre-key
   - Alice\'s device performs Diffie-Hellman key exchange
   - Generates shared secret (symmetric key)

3. Alice encrypts message with shared secret
   - Server never sees plaintext or shared secret
   - Only Bob's device can decrypt (has private keys)
\`\`\`

**Ratcheting** (forward secrecy):
- After each message, keys are rotated (ratcheted)
- Even if current key compromised, past messages stay secure
- This is Double Ratchet Algorithm (core of Signal Protocol)

**Group Encryption**:
- Each participant has pairwise encryption with sender
- Sender encrypts message N times (once per recipient)
- Efficient sender-side encryption tree for large groups

---

## Step 6: Message Storage

**Database: Cassandra**

\`\`\`cql
CREATE TABLE messages (
    user_id BIGINT,
    conversation_id TEXT,
    message_id TIMEUUID,
    sender_id BIGINT,
    encrypted_content BLOB,
    media_url TEXT,
    timestamp TIMESTAMP,
    status VARCHAR(20),  -- sent, delivered, read
    PRIMARY KEY (user_id, conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
\`\`\`

**Why Cassandra?**
- Write-heavy (1M+ writes/sec)
- Time-series data (messages ordered by time)
- Linear scalability
- Multi-datacenter replication

**Message Retention**:
- Messages stored on device (primary storage)
- Server stores for 30 days as backup (for new device setup)
- After 30 days, deleted from server (reduces storage costs)

---

## Step 7: Group Messaging

**Group Structure**:
\`\`\`
- Group ID
- Admin (s): Can add/remove members, change settings
- Members: Up to 256 participants
- Group metadata: Name, icon, created date
\`\`\`

**Send to Group**:
\`\`\`
1. Alice sends message to group (256 members)
2. Client encrypts message 256 times (one per member's public key)
3. Send to server with recipient list
4. Server fans out to all members:
   - Online members: Push immediately
   - Offline members: Store in inbox
5. Each member's device decrypts with own private key
\`\`\`

**Optimization** (Sender Keys):
- Instead of encrypting 256 times, use sender key
- Encrypt message once with group key
- Share group key with members (encrypted individually)
- Reduces encryption overhead for large groups

---

## Step 8: Media Sharing

**Upload Flow**:
\`\`\`
1. Alice selects photo (5 MB)
2. Client compresses to 2 MB (balance quality vs size)
3. Encrypt photo with recipient's key
4. Generate thumbnail (50 KB, encrypted)
5. Upload to S3 via pre-signed URL
6. Send message with media_url: s3://whatsapp-media/abc123.enc
7. Bob\'s device downloads encrypted file
8. Decrypt with shared key
9. Display photo
\`\`\`

**Media Storage**:
- S3 (or equivalent) for blob storage
- CDN (CloudFront) for fast global delivery
- Encrypted at rest (E2E encryption + S3 encryption)
- Auto-delete after 30 days (like messages)

**Voice Messages**:
- Record audio, compress with Opus codec
- ~1 MB per minute
- Same encryption/upload flow as photos

---

## Step 9: Voice/Video Calls

**Signaling** (call setup):
\`\`\`
1. Alice initiates call to Bob
2. Send CALL_OFFER via WhatsApp servers (encrypted)
3. Bob's device rings, Bob accepts
4. Exchange ICE candidates (IP addresses, ports)
5. Establish P2P connection (WebRTC)
\`\`\`

**Media Transmission**:
- **Direct P2P** (if possible): Lowest latency, best quality
- **Relayed via TURN servers** (if NAT/firewall blocks P2P):
  - WhatsApp TURN servers relay media
  - Encrypted with SRTP (Secure RTP)

**Protocols**:
- WebRTC (standard for browser/app real-time communication)
- STUN/TURN for NAT traversal
- Opus for audio codec (excellent compression/quality)
- VP8/H.264 for video codec

---

## Step 10: Multi-Device Support

**Challenge**: Messages encrypted with device-specific keys. How to sync across devices?

**WhatsApp Multi-Device 2.0** (2021):
\`\`\`
1. Primary device (phone) remains master
2. Secondary devices (web, desktop) register with phone
3. Phone shares encryption keys with secondary devices (once)
4. Each device can send/receive messages independently
5. Devices sync message history when phone online
\`\`\`

**Before Multi-Device 2.0**:
- Phone had to be online for web/desktop to work
- All messages routed through phone
- Phone went to sleep → Web disconnected (poor UX)

---

## Step 11: Status Updates (Stories)

**24-Hour Ephemeral Content**:
\`\`\`
1. Alice posts status (photo + caption)
2. Encrypt with viewer list keys
3. Upload to S3 with TTL metadata: expires_at = now + 24 hours
4. Notify contacts: "Alice posted a status"
5. Contacts view status (decrypt with their key)
6. After 24 hours:
   - S3 lifecycle policy deletes media
   - Database record marked expired
   - Status disappears from UI
\`\`\`

**View Tracking**:
- Track who viewed status
- Store in Redis (ephemeral data, 24-hour TTL)
- After 24 hours, view data deleted

---

## Step 12: Optimizations

**1. Message Compression**:
- Text compressed with gzip (70% reduction)
- Media already compressed (JPEG, H.264)

**2. Batching**:
- Batch multiple messages before sending
- Reduces TCP overhead
- Every 100ms or 10 messages, whichever comes first

**3. Connection Pooling**:
- Persistent TCP connections (not HTTP request/response)
- Reduces handshake overhead
- Enables instant push notifications

**4. Adaptive Media Quality**:
- On slow networks, send lower resolution photos
- Progressive JPEG (loads top-to-bottom)
- Videos: Adaptive bitrate streaming

---

## Step 13: Infrastructure & Scale

**WhatsApp's Famous Efficiency**:
- 50 engineers for 900M users (2014)
- Erlang: Single server handles 2M+ connections
- FreeBSD: Tuned for network performance
- Minimal features: No ads, no complex features
- Focus on reliability and speed

**Scaling Strategy**:
- Vertical scaling first (more CPU/RAM per server)
- Horizontal scaling when vertical exhausted
- Sharding by user_id (all user's data on one shard)

**Infrastructure Costs**:
- $1 per user per year (2014 numbers)
- Revenue: $1/year subscription (discontinued, now free with FB ads)

---

## Step 14: Reliability

**Message Delivery Guarantees**:
1. **At-Least-Once Delivery**: Message sent until acknowledged
2. **Idempotency**: Duplicate messages detected by message_id
3. **Retry Logic**: Exponential backoff (1s, 2s, 4s, 8s, ...)
4. **Offline Queue**: Store up to 1000 undelivered messages per user

**High Availability**:
- Multi-region deployment (US, EU, Asia)
- Active-active (all regions serve traffic)
- If one region down, traffic routes to others

**Monitoring**:
- Message delivery rate (target: 99.9%)
- Latency p99 (target: <100ms)
- Connection success rate (target: 99%)

---

## Step 15: Comparison with Messenger (Facebook)

**WhatsApp Philosophy**: Simple, reliable, privacy-focused
**Messenger Philosophy**: Feature-rich, social integration

| Feature | WhatsApp | Messenger |
|---------|----------|-----------|
| E2E Encryption | Default, always | Optional (Secret Conversations) |
| Infrastructure | Erlang, simple | Complex microservices |
| Features | Minimal | Stories, games, bots, payments |
| Team Size | 50 engineers (2014) | 1000+ engineers |
| Focus | Reliability, privacy | Features, engagement |

---

## Interview Tips

**Clarify**:
1. Scale: 2B users or 100M?
2. E2E encryption required?
3. Multi-device support?
4. Media sharing limits?

**Emphasize**:
1. **E2E Encryption**: Signal Protocol overview
2. **Message Queue**: Per-user inbox for offline delivery
3. **Erlang**: Explain why (concurrency, fault tolerance)
4. **Simple Architecture**: WhatsApp\'s famous efficiency

**Common Mistakes**:
- Over-engineering with microservices (WhatsApp uses monolithic approach)
- Ignoring encryption (critical for WhatsApp)
- Not handling offline message delivery
- Assuming HTTP REST (WhatsApp uses persistent TCP/WebSocket)

---

## Summary

**Core Components**:
1. **Persistent TCP Connections**: Erlang handles millions per server
2. **Message Router**: Routes messages to recipient queues
3. **User Message Queue**: Per-user inbox (Redis + Cassandra)
4. **E2E Encryption**: Signal Protocol (Double Ratchet)
5. **Media Storage**: S3 + CloudFront CDN
6. **Voice/Video**: WebRTC P2P or TURN relay
7. **Multi-Device**: Independent encryption keys per device

**Key Decisions**:
- ✅ Erlang + FreeBSD for efficiency
- ✅ Signal Protocol for E2E encryption
- ✅ Persistent connections (not HTTP)
- ✅ Simple, focused feature set
- ✅ Messages deleted after 30 days (storage savings)
- ✅ At-least-once delivery with idempotency

**Capacity**:
- 2B users, 100B messages/day
- 100M concurrent connections
- 1M+ messages/sec processing
- 10 PB/day media storage

WhatsApp's design prioritizes **reliability, privacy, and efficiency** over feature richness, enabling incredible scale with minimal infrastructure.`,
};
