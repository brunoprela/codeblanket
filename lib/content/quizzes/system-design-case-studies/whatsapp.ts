/**
 * Quiz questions for Design WhatsApp section
 */

export const whatsappQuiz = [
    {
        id: 'q1',
        question:
            'Explain how WhatsApp implements end-to-end encryption using the Signal Protocol. Walk through the key exchange process and explain why even WhatsApp servers cannot read messages.',
        sampleAnswer:
            'SIGNAL PROTOCOL E2E ENCRYPTION: (1) KEY GENERATION: Each device generates keys: Identity key pair (long-term, stored securely), Signed pre-key pair (rotated weekly), 100 one-time pre-key pairs (ephemeral, used once). Public keys uploaded to WhatsApp servers. Private keys never leave device. (2) KEY EXCHANGE (Alice → Bob first time): Alice requests Bob\'s public keys from server. Server returns: Bob\'s identity public key, signed pre-key, one one-time pre-key. Alice performs 4 Diffie-Hellman operations (X3DH protocol): DH1 = Alice identity × Bob signed pre-key. DH2 = Alice ephemeral × Bob identity. DH3 = Alice ephemeral × Bob signed pre-key. DH4 = Alice ephemeral × Bob one-time pre-key. Combine all DH outputs → master secret. Derive symmetric encryption key from master secret (AES-256). (3) MESSAGE ENCRYPTION: Alice encrypts message with symmetric key. Encrypted blob sent to WhatsApp servers. Server routes to Bob (still encrypted). Bob\'s device uses private keys to derive same master secret → decrypts message. (4) WHY SERVERS CAN\'T READ: WhatsApp server only sees public keys and encrypted blob. Master secret derived from private keys (never sent to server). Without private keys, mathematically infeasible to decrypt (256-bit security). (5) FORWARD SECRECY (Ratcheting): After each message, keys "ratchet" forward (new DH exchange). Even if current key compromised, past messages remain secure. KEY INSIGHT: E2E encryption means even service provider cannot decrypt - only sender and recipient devices have private keys.',
        keyPoints: [
            'X3DH key exchange: 4 Diffie-Hellman operations derive shared secret',
            'Private keys never leave device, only public keys on server',
            'Encrypted message sent to server - server cannot decrypt',
            'Double Ratchet: Keys change after each message (forward secrecy)',
            'WhatsApp zero-knowledge encryption: Cannot read user messages',
        ],
    },
    {
        id: 'q2',
        question:
            'WhatsApp runs with remarkable efficiency (50 engineers for 900M users in 2014). Explain the technology choices (Erlang, FreeBSD) and architecture decisions that enabled this scale.',
        sampleAnswer:
            'WHATSAPP EFFICIENCY SECRETS: (1) ERLANG: Designed for telecom switches (99.999% uptime). Lightweight processes: 1M+ concurrent connections per server (vs 10K for traditional servers). Actor model: Each user connection = isolated process (crash doesn\'t affect others). Hot code swapping: Update code without downtime. Excellent for I/O-bound workloads (messaging). (2) FREEBSD: Minimalist OS (vs Linux bloat), tuned kernel for network performance. Efficient TCP stack, low memory overhead. WhatsApp tuned: Increase file descriptor limits to 2M per process. Optimize network buffers for high throughput. (3) SIMPLE ARCHITECTURE: Monolithic application (not microservices - less operational overhead). Persistent TCP connections (not HTTP REST - reduces handshake overhead). No complex features (no ads, games, bots - focus on core messaging). Vertical scaling first (powerful servers with 256GB RAM, 32 cores). (4) MINIMAL STORAGE: Messages deleted after 30 days (reduces storage from PB to TB). Only metadata stored long-term, media in S3 with lifecycle policies. Client-side storage primary (device has full message history). (5) OPERATIONAL DISCIPLINE: Few engineers → Careful about adding complexity. Automated deployment, monitoring. On-call rotation with runbooks. RESULT: $1 per user per year operational cost. 1 engineer per 18M users (2014). COMPARISON: Facebook (Messenger): 1000+ engineers, complex microservices, 1 engineer per 2M users. WhatsApp: Simpler, more reliable, 10x more efficient. KEY INSIGHT: Minimalism and right technology choices (Erlang for concurrency, FreeBSD for performance) enable incredible efficiency.',
        keyPoints: [
            'Erlang: 1M+ connections per server, lightweight processes, fault tolerance',
            'FreeBSD: Tuned kernel for network performance',
            'Monolithic architecture: Simpler than microservices',
            'Messages deleted after 30 days: Reduces storage 100x',
            'Focus on core features: No bloat, extreme reliability',
        ],
    },
    {
        id: 'q3',
        question:
            'Design the offline message delivery system. User Bob is offline for 3 days. Alice sends 50 messages. How do you store, deliver, and ensure reliability when Bob comes online?',
        sampleAnswer:
            'OFFLINE MESSAGE DELIVERY: (1) STORAGE: Alice sends message while Bob offline. WhatsApp server receives encrypted message (cannot decrypt). Store in Bob\'s inbox: Redis: LPUSH inbox:user_bob message_1 (fast, in-memory). Cassandra: INSERT INTO offline_messages (user_id, message_id, encrypted_content, sender, timestamp) (persistent backup). Track count: INCR offline_count:user_bob (now 50 after 50 messages). (2) PUSH NOTIFICATION: Send push notification to Bob\'s device: "50 new messages from Alice". Badge count updated on app icon. (3) BOB COMES ONLINE: Bob opens WhatsApp, connects to server. Server detects connection: Redis HSET online_users user_bob {connection_id, timestamp}. (4) MESSAGE DELIVERY: Query offline messages: Cassandra: SELECT * FROM offline_messages WHERE user_id = Bob ORDER BY timestamp. Push messages to Bob\'s device in order (FIFO). Bob\'s device decrypts and displays. (5) ACKNOWLEDGMENT: Bob\'s device sends ACK for each message. Server deletes from Redis/Cassandra after ACK: LREM inbox:user_bob message_1. If device crashes during delivery, unacked messages remain for retry. (6) LARGE BACKLOG HANDLING: If 10,000 messages: Send in batches of 100. Paginate: Send batch 1, wait for ACK, send batch 2. Avoid overwhelming client device (memory, battery). (7) FAILURE SCENARIOS: If delivery fails (network dropped): Messages remain in Redis. Next connection attempt resumes delivery. Messages retained for 30 days. After 30 days: Delete (assume Bob switched phones/uninstalled). (8) DUPLICATE PREVENTION: Each message has unique message_id (TIMEUUID). Client tracks received message_ids (Bloom filter). If duplicate received (retry), ignore based on message_id. KEY INSIGHT: Offline inbox is persistent queue (Redis for speed, Cassandra for durability) with at-least-once delivery + client deduplication.',
        keyPoints: [
            'Store in Redis (fast) + Cassandra (durable) offline inbox',
            'Send push notification with badge count',
            'Deliver messages in batches when user connects',
            'ACK-based: Delete only after client acknowledges receipt',
            'Retain for 30 days, then delete (storage limits)',
        ],
    },
];

