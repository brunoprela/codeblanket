/**
 * WhatsApp Architecture Section
 */

export const whatsapparchitectureSection = {
  id: 'whatsapp-architecture',
  title: 'WhatsApp Architecture',
  content: `WhatsApp is the world's most popular messaging app with over 2 billion users exchanging 100 billion messages daily. Acquired by Facebook (now Meta) in 2014 for $19 billion, WhatsApp is notable for achieving massive scale with a small engineering team. This section explores the minimalist yet highly efficient architecture behind WhatsApp.

## Overview

WhatsApp\'s impressive scale and efficiency:
- **2+ billion users** worldwide
- **100 billion messages** per day
- **2 billion voice/video calls** daily
- **Small team**: 50 engineers for 900 million users (at acquisition)
- **High availability**: 99.9%+ uptime
- **End-to-end encryption**: All messages encrypted by default

### Key Architectural Principles

1. **Minimalism**: Keep it simple, avoid complexity
2. **Erlang/OTP**: Built on Erlang for massive concurrency
3. **Stateless servers**: Easy to scale horizontally
4. **Efficient protocols**: Custom binary protocol (less bandwidth)
5. **No ads, no tracking**: Privacy-focused

---

## Core Components

### 1. Real-Time Messaging

WhatsApp delivers messages in real-time with minimal latency.

**Message Flow**:

\`\`\`
User A sends message to User B:

1. Client encrypts message (end-to-end encryption)
2. Send to WhatsApp server via persistent connection
3. Server routes to User B's connected server
4. Deliver to User B's device
5. User B decrypts message

Latency: <100ms (if both users online)
\`\`\`

**Persistent Connection**:
- **WebSocket-like protocol** (custom binary protocol)
- Persistent TCP connection between client and server
- Keeps connection alive with heartbeats (every 30 seconds)
- If connection drops, reconnect automatically

**Why Persistent Connection?**
- Instant delivery (no polling needed)
- Low latency (<100ms)
- Battery-efficient (vs frequent polling)

---

### 2. Erlang/OTP for Concurrency

WhatsApp is built on **Erlang**, a programming language designed for telecom systems.

**Why Erlang?**

**1. Massive Concurrency**:
- Handle millions of concurrent connections per server
- Lightweight processes (not OS threads)
- Each connection = one Erlang process
- Single server: 2 million+ concurrent connections

**2. Fault Tolerance**:
- "Let it crash" philosophy
- Supervisor trees restart failed processes
- Isolated failures (one user's connection crash doesn't affect others)

**3. Hot Code Swapping**:
- Update code without stopping servers
- Zero-downtime deployments

**4. Proven in Telecom**:
- Erlang used in Ericsson telecom switches (99.9999999% uptime)
- Battle-tested for high availability

**Architecture**:
\`\`\`
Client → Load Balancer → Erlang Chat Server
                             ↓
                        Connection Process (Erlang process per connection)
                             ↓
                        Message Router
                             ↓
                        Recipient\'s Connection Process
                             ↓
                        Recipient's Client
\`\`\`

**Erlang Process**:
- Lightweight (2-3 KB memory each)
- 2 million processes per server (4 million connections = 4 million processes)
- Isolated (crash in one process doesn't affect others)

**Example Code** (simplified):
\`\`\`erlang
-module (connection).
-behaviour (gen_server).

%% Each user connection is a gen_server process

handle_message(Message, State) ->
    EncryptedMessage = Message#message.encrypted_content,
    RecipientPid = find_recipient_process(Message#message.to),
    RecipientPid ! {new_message, EncryptedMessage},
    {noreply, State}.
\`\`\`

---

### 3. Stateless Servers

WhatsApp servers are **stateless** (mostly).

**What is Stateless?**
- Server doesn't store user state between requests
- Connection state stored in memory (ephemeral)
- Persistent data stored in database

**Benefits**:
- **Easy to scale**: Add more servers, no state migration
- **High availability**: If server crashes, users reconnect to different server
- **Load balancing**: Route users to any server

**Routing**:
- User connects to any chat server
- Server registers connection in routing table (user_id → server_ip)
- When message arrives for user, look up routing table, forward to correct server

**Data Model** (routing table in Redis):
\`\`\`
Key: user:123:connection
Value: {server_ip: "10.0.1.5", connected_at: timestamp}
TTL: 60 seconds (if no heartbeat, connection considered dead)
\`\`\`

---

### 4. Message Storage

WhatsApp stores messages temporarily for delivery, then deletes.

**Storage Strategy**:

**1. Online Users (Connected)**:
- Message delivered immediately
- Not stored on server (end-to-end encrypted, server can't read)

**2. Offline Users**:
- Store message temporarily (encrypted)
- Deliver when user comes online
- Delete after delivery (or 30 days, whichever comes first)

**Why Not Permanent Storage?**
- **Privacy**: Messages are temporary, not stored indefinitely
- **Cost**: Less storage, lower costs
- **Regulatory**: Easier to comply with privacy laws (GDPR)

**Database**: **Mnesia** (Erlang\'s built-in distributed database)

**Data Model**:
\`\`\`
Table: messages
- message_id
- from_user_id
- to_user_id
- encrypted_content
- timestamp
- delivered (boolean)
\`\`\`

**Message Lifecycle**:
\`\`\`
1. User A sends message to User B (offline)
2. Store in Mnesia (encrypted)
3. User B comes online
4. Deliver message to User B
5. User B sends ACK (acknowledgment)
6. Delete message from Mnesia
\`\`\`

**Backup**: Messages also stored on client devices (local SQLite database).

---

### 5. End-to-End Encryption (E2EE)

WhatsApp uses **Signal Protocol** for end-to-end encryption.

**Key Properties**:
- Only sender and recipient can read messages
- WhatsApp servers cannot decrypt messages
- Forward secrecy (past messages safe even if keys compromised)

**How E2EE Works**:

**1. Key Exchange**:
- When User A and User B first chat, exchange public keys
- Use Diffie-Hellman key exchange
- Establish shared secret key

**2. Message Encryption**:
- Encrypt message with shared secret key (AES-256)
- Send encrypted message to server
- Server relays encrypted message (cannot decrypt)
- Recipient decrypts with shared secret key

**3. Ratcheting**:
- Keys rotate frequently (every message)
- "Double Ratchet" algorithm (Signal Protocol)
- Ensures forward secrecy (compromised keys don't expose past messages)

**Example**:
\`\`\`
User A → User B (first message):
1. A generates key pair (A_private, A_public)
2. B generates key pair (B_private, B_public)
3. Exchange public keys via server
4. A derives shared_secret from A_private + B_public
5. B derives same shared_secret from B_private + A_public
6. A encrypts message: encrypted_msg = AES256(message, shared_secret)
7. A sends encrypted_msg to server
8. Server relays to B
9. B decrypts: message = AES256_decrypt (encrypted_msg, shared_secret)
10. Keys rotate for next message
\`\`\`

**Key Storage**:
- Private keys stored on device (never leave device)
- Public keys registered with server
- Server acts as key directory (user_id → public_key)

---

### 6. Group Chat

Group chats support up to 256 members.

**Challenges**:
- Message must reach all members
- Some members online, others offline
- End-to-end encryption with multiple recipients

**Approach**:

**Sender-Side Fanout**:
- Client encrypts message separately for each recipient
- Send N encrypted messages to server (one per recipient)
- Server delivers to each recipient

\`\`\`
User A sends message to Group (B, C, D):
1. A encrypts message for B: enc_msg_B = encrypt (message, shared_secret_AB)
2. A encrypts message for C: enc_msg_C = encrypt (message, shared_secret_AC)
3. A encrypts message for D: enc_msg_D = encrypt (message, shared_secret_AD)
4. A sends [enc_msg_B, enc_msg_C, enc_msg_D] to server
5. Server delivers enc_msg_B to B, enc_msg_C to C, enc_msg_D to D
\`\`\`

**Why Sender-Side Fanout?**
- Maintains end-to-end encryption (server can't decrypt)
- Each recipient has unique shared key with sender

**Challenge**: For 256-member group, sender must encrypt 256 times!

**Optimization**:
- Use Sender Keys (group key shared among members)
- Encrypt once with group key, send to all members
- Group key rotates when member added/removed

---

### 7. Voice and Video Calls

WhatsApp supports voice and video calls (2 billion+ calls per day).

**Architecture**:

**1. Peer-to-Peer** (When Possible):
- Direct connection between caller and recipient
- No server relaying media (lower latency, less bandwidth)
- Use WebRTC (standard protocol)

**2. Relayed** (When P2P Fails):
- If direct connection fails (NAT, firewall), relay through server
- TURN servers (Traversal Using Relays around NAT)
- Lower quality but better than no call

**Signaling**:
\`\`\`
User A calls User B:
1. A sends "call" message to B via WhatsApp messaging infrastructure
2. B receives call notification
3. B accepts call
4. Exchange ICE candidates (IP addresses, ports) via WhatsApp server
5. Establish P2P connection (if possible) or fallback to relay
6. Media flows: A ↔ B (directly or via relay)
\`\`\`

**Codecs**:
- **Opus** for audio (low latency, high quality)
- **VP8/VP9** for video (efficient compression)
- Adaptive bitrate (adjust quality based on bandwidth)

**End-to-End Encryption**:
- Audio/video encrypted using SRTP (Secure Real-time Transport Protocol)
- Same Signal Protocol keys as messaging

---

### 8. Status (Stories)

WhatsApp Status is similar to Instagram Stories (24-hour ephemeral content).

**Architecture**:

**Storage**:
- Media (photos/videos) stored in WhatsApp servers
- Metadata (who posted, when, viewed by) in database

**Data Model**:
\`\`\`
Table: statuses
- status_id
- user_id
- media_url (S3)
- created_at
- expires_at (24 hours later)

Table: status_views
- status_id
- viewer_user_id
- viewed_at
\`\`\`

**Auto-Deletion**:
- Background job deletes statuses after 24 hours
- Media deleted from S3
- Metadata deleted from database

**Privacy**:
- User controls who can view status (all contacts, selected contacts, except...)

---

## Technology Stack

### Core Technologies

**1. Erlang/OTP**:
- Main programming language
- Handles messaging, connections, routing

**2. FreeBSD**:
- Operating system (chosen for performance, stability)
- Custom kernel tuning for high concurrency

**3. Mnesia**:
- Erlang\'s distributed database
- Stores temporary messages, user sessions

**4. XMPP (Modified)**:
- Originally used XMPP protocol (Extensible Messaging and Presence Protocol)
- Heavily customized for WhatsApp's needs
- Now custom binary protocol

**5. Protobuf**:
- Binary serialization format
- Smaller message size than JSON (saves bandwidth)

---

### Infrastructure

**Before Facebook Acquisition**:
- **Own datacenters**: Not cloud-based
- **Minimal infrastructure**: 50 engineers managed infrastructure

**After Facebook Acquisition** (2014):
- **Facebook's infrastructure**: Leveraged Facebook\'s datacenters, expertise
- **Shared services**: Storage, load balancing, monitoring
- **Integration**: Shared user IDs, contacts with Facebook

---

## Scaling Strategies

### 1. Erlang's Concurrency

Single server handles 2+ million concurrent connections.

**How?**
- Lightweight Erlang processes (2-3 KB each)
- Efficient scheduling (Erlang VM)
- Non-blocking I/O

**Comparison**:
- Traditional thread-per-connection: 1 MB per connection → 1000 connections per server
- Erlang process-per-connection: 2 KB per connection → 2 million connections per server

---

### 2. Stateless Architecture

Easy to scale horizontally:
- Add more servers → Distribute load
- No state migration needed
- Load balancer routes users to any server

---

### 3. Efficient Protocol

Custom binary protocol reduces bandwidth:
- Messages serialized with Protobuf (compact)
- Compression (gzip)
- Example: "Hello" in JSON (40 bytes) vs Protobuf (10 bytes)

**Benefit**: Lower bandwidth → Lower costs, faster delivery

---

### 4. Minimal Features

WhatsApp focused on core messaging, avoided feature bloat:
- No ads, no games, no news feed
- Simple UI
- Fast development

**Philosophy**: "Do one thing well"

---

## Key Lessons

### 1. Erlang Enables Massive Concurrency

Erlang's lightweight processes and fault tolerance allowed WhatsApp to handle 2 million connections per server, achieving scale with small team.

### 2. Stateless Architecture Simplifies Scaling

No server-side state makes it easy to add servers, improves availability, simplifies load balancing.

### 3. End-to-End Encryption is Feasible at Scale

WhatsApp proved E2EE works at massive scale (2 billion users), setting standard for privacy.

### 4. Minimalism and Focus

By avoiding feature bloat and focusing on core messaging, WhatsApp achieved reliability and simplicity.

### 5. Small Team, Big Impact

50 engineers for 900 million users (at acquisition) shows power of right technology choices (Erlang) and focus.

---

## Interview Tips

**Q: How does WhatsApp handle 2 billion users with a small team?**

A: (1) Erlang/OTP: Lightweight processes handle millions of concurrent connections per server with built-in fault tolerance. Single server handles 2+ million connections. (2) Stateless servers: Easy to scale horizontally, no state migration. (3) Efficient protocol: Custom binary protocol (Protobuf) reduces bandwidth, lowers costs. (4) Minimalism: Focus on core messaging, avoid feature bloat, reduces complexity. (5) Persistent connections: WebSocket-like protocol with heartbeats, instant message delivery. (6) Mnesia: Erlang\'s distributed database for temporary message storage, routing tables. Result: Small team manages massive scale.

**Q: How does WhatsApp implement end-to-end encryption?**

A: Uses Signal Protocol. Key exchange: When users first chat, exchange public keys (Diffie-Hellman). Derive shared secret key. Encryption: Sender encrypts message with AES-256 using shared secret, sends encrypted message to server. Server relays encrypted message (cannot decrypt). Recipient decrypts with shared secret. Ratcheting: Keys rotate every message (Double Ratchet algorithm) for forward secrecy. Group chats: Sender-side fanout (encrypt separately for each member) or Sender Keys (shared group key). Voice/video calls: SRTP encryption with same keys. Key storage: Private keys never leave device, public keys registered with server. Result: Only sender and recipient can read messages, server is blind.

**Q: How does WhatsApp deliver messages to offline users?**

A: Store temporarily in Mnesia (Erlang's distributed database). When User A sends message to offline User B: (1) Server checks routing table (Redis): User B offline (no connected server). (2) Store encrypted message in Mnesia (message_id, from, to, encrypted_content, timestamp). (3) When User B comes online, establish connection, server checks Mnesia for pending messages. (4) Deliver messages to User B. (5) User B sends ACK (acknowledgment). (6) Server deletes messages from Mnesia. Messages deleted after delivery or 30 days (privacy). Messages also stored on client devices (local SQLite) for backup.

---

## Summary

WhatsApp's architecture demonstrates achieving massive scale with minimalism and right technology choices:

**Key Takeaways**:

1. **Erlang/OTP**: Massive concurrency (2M+ connections per server), fault tolerance, hot code swapping
2. **Stateless servers**: Easy horizontal scaling, high availability, simple load balancing
3. **End-to-end encryption**: Signal Protocol, all messages encrypted, privacy-first
4. **Persistent connections**: WebSocket-like protocol, instant delivery, battery-efficient
5. **Temporary storage**: Messages stored only for delivery (30 days max), then deleted
6. **Efficient protocol**: Custom binary protocol (Protobuf), reduces bandwidth
7. **Minimalism**: Focus on core messaging, avoid feature bloat, small team
8. **Group chat**: Sender-side fanout or Sender Keys for E2EE with multiple recipients

WhatsApp's success shows that right technology (Erlang), focus (core messaging), and simplicity enable massive scale with small team.
`,
};
