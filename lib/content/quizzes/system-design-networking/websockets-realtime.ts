/**
 * Quiz questions for WebSockets & Real-Time Communication section
 */

export const websocketsrealtimeQuiz = [
  {
    id: 'chat-websocket-design',
    question:
      "Design a real-time chat system like Slack that needs to handle 10 million concurrent users. Explain your WebSocket architecture, how you'd scale it, how you'd handle message delivery guarantees, and what happens when a user is offline. Discuss trade-offs in your design.",
    sampleAnswer: `**System Requirements**:
- 10 million concurrent users
- Real-time message delivery (<100ms latency)
- Message history/persistence
- Offline message delivery
- Typing indicators, read receipts
- File sharing

**Architecture Overview**:

\`\`\`
Clients (10M)
    ↓
[CDN] → Static assets
    ↓
[Load Balancer] → WebSocket Gateway Cluster (1000 servers)
    ↓
[Message Queue (Kafka)] → Message Broker
    ↓
[Application Services Cluster]
    ↓
[Database (Cassandra)] → Message Storage
[Cache (Redis)] → Online users, presence
[Object Storage (S3)] → Files
\`\`\`

**Component Design**:

**1. WebSocket Gateway Layer**

*Purpose*: Handle WebSocket connections only, no business logic

*Scaling*:
- Each server handles 10,000 concurrent connections
- 10M users ÷ 10k per server = 1,000 servers needed
- Optimize: Use Erlang/Elixir (Discord's approach) or Go for high concurrency

*Implementation*:
\`\`\`javascript
class WebSocketGateway {
  constructor() {
    this.connections = new Map(); // userId → ws connection
    this.kafka = new KafkaProducer();
    this.redis = new Redis();
  }
  
  async handleConnection(ws, userId) {
    // Store connection
    this.connections.set(userId, ws);
    
    // Update presence in Redis
    await this.redis.sadd('online_users', userId);
    
    // Publish presence update
    this.kafka.publish('presence', {
      type: 'online',
      userId
    });
    
    ws.on('message', async (msg) => {
      await this.handleMessage(userId, msg);
    });
    
    ws.on('close', async () => {
      this.connections.delete(userId);
      await this.redis.srem('online_users', userId);
      
      this.kafka.publish('presence', {
        type: 'offline',
        userId
      });
    });
  }
  
  async handleMessage(userId, message) {
    // Parse message
    const data = JSON.parse(message);
    
    // Add metadata
    data.senderId = userId;
    data.timestamp = Date.now();
    data.messageId = uuid();
    
    // Publish to Kafka for processing
    await this.kafka.publish('messages', data);
  }
  
  async sendToUser(userId, message) {
    const ws = this.connections.get(userId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }
}
\`\`\`

**2. Message Queue (Kafka)**

*Why Kafka*:
- High throughput (millions of messages/sec)
- Durability (messages persisted)
- Replay capability (for offline users)
- Multiple consumers can process independently

*Topics*:
- \`messages\`: Chat messages
- \`presence\`: Online/offline updates
- \`typing\`: Typing indicators
- \`read-receipts\`: Message read status

**3. Message Processing Service**

*Responsibilities*:
- Validate messages
- Spam detection
- Store in database
- Fan out to recipients
- Handle offline delivery

\`\`\`javascript
class MessageProcessor {
  async processMessage(msg) {
    // 1. Validate and sanitize
    if (!this.isValid(msg)) {
      return;
    }
    
    // 2. Persist to database
    await this.db.insert('messages', {
      id: msg.messageId,
      channelId: msg.channelId,
      senderId: msg.senderId,
      content: msg.content,
      timestamp: msg.timestamp
    });
    
    // 3. Get channel members
    const members = await this.getChannelMembers(msg.channelId);
    
    // 4. Fan out to online members
    const deliveryStatus = await Promise.all(
      members.map(userId => this.deliverToUser(userId, msg))
    );
    
    // 5. Store for offline members
    const offlineUsers = members.filter((_, i) => !deliveryStatus[i]);
    if (offlineUsers.length > 0) {
      await this.storeOfflineMessages(offlineUsers, msg);
    }
    
    // 6. Send acknowledgment to sender
    await this.sendAck(msg.senderId, msg.messageId);
  }
  
  async deliverToUser(userId, msg) {
    // Check if user is online
    const isOnline = await this.redis.sismember('online_users', userId);
    if (!isOnline) {
      return false;
    }
    
    // Find which gateway server has the user
    const serverId = await this.redis.hget('user_connections', userId);
    
    // Publish to that server's topic
    await this.kafka.publish(\`gateway-\${serverId}\`, {
      type: 'deliver',
      userId,
      message: msg
    });
    
    return true;
  }
}
\`\`\`

**4. Offline Message Handling**

*Strategy 1: Push notifications*
- Send push notification to mobile device
- User opens app, connects, receives queued messages

*Strategy 2: Message inbox*
\`\`\`javascript
// When user comes online
async function onUserConnect(userId) {
  // Fetch undelivered messages
  const messages = await db.query(
    'SELECT * FROM offline_messages WHERE userId = ? ORDER BY timestamp',
    [userId]
  );
  
  // Deliver all messages
  for (const msg of messages) {
    await sendToUser(userId, msg);
  }
  
  // Clear offline queue
  await db.delete('offline_messages', { userId });
}
\`\`\`

*Storage*:
- DynamoDB: \`{ userId, messageId, message, timestamp }\`
- TTL: Auto-delete after 30 days
- Query pattern: \`userId\` as partition key, \`timestamp\` as sort key

**5. Message Delivery Guarantees**

*At-least-once delivery*:

\`\`\`javascript
// Client-side
class ReliableWebSocket {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.pendingMessages = new Map(); // messageId → message
    this.ackTimeout = 5000; // 5 seconds
  }
  
  send(message) {
    const messageId = uuid();
    message.id = messageId;
    
    this.ws.send(JSON.stringify(message));
    
    // Set timeout for acknowledgment
    const timer = setTimeout(() => {
      console.log('No ack received, resending');
      this.send(message);
    }, this.ackTimeout);
    
    this.pendingMessages.set(messageId, {
      message,
      timer
    });
  }
  
  handleAck(ack) {
    const pending = this.pendingMessages.get(ack.messageId);
    if (pending) {
      clearTimeout(pending.timer);
      this.pendingMessages.delete(ack.messageId);
    }
  }
}
\`\`\`

*Idempotency*:
\`\`\`javascript
// Server-side: Deduplicate messages
const processedMessages = new Set();

async function handleMessage(msg) {
  if (processedMessages.has(msg.messageId)) {
    // Already processed, send ack again
    sendAck(msg.senderId, msg.messageId);
    return;
  }
  
  // Process message
  await storeMessage(msg);
  await deliverMessage(msg);
  
  // Mark as processed
  processedMessages.add(msg.messageId);
  
  // Send ack
  sendAck(msg.senderId, msg.messageId);
}
\`\`\`

**6. Presence and Typing Indicators**

*Presence*:
\`\`\`javascript
// Redis sorted set: score = last seen timestamp
await redis.zadd('presence', Date.now(), userId);

// Get users active in last 5 minutes
const activeUsers = await redis.zrangebyscore(
  'presence',
  Date.now() - 300000,
  Date.now()
);
\`\`\`

*Typing indicators*:
- Don't persist (ephemeral)
- Publish to Kafka topic with 2-second TTL
- Only deliver to online channel members
- Throttle: Max 1 update per second per user

**7. File Sharing**

*Flow*:
1. Client uploads file to S3 (direct upload with presigned URL)
2. Client sends message with file URL
3. Server validates file (virus scan)
4. Deliver message with file link

\`\`\`javascript
// Get presigned URL for upload
app.post('/api/upload-url', async (req, res) => {
  const { filename, contentType } = req.body;
  
  const key = \`files/\${uuid()}/\${filename}\`;
  const url = await s3.getSignedUrl('putObject', {
    Bucket: 'chat-files',
    Key: key,
    ContentType: contentType,
    Expires: 300 // 5 minutes
  });
  
  res.json({ uploadUrl: url, fileKey: key });
});

// Client uploads directly to S3
// Then sends message with file reference
\`\`\`

**Scaling Considerations**:

*Horizontal Scaling*:
- WebSocket gateways: Stateless (except connections), easy to scale
- Add more servers behind load balancer
- Sticky sessions based on userId hash

*Geographic Distribution*:
- Deploy in multiple regions (US-East, US-West, EU, Asia)
- Users connect to nearest region
- Cross-region message delivery via global Kafka cluster

*Connection Distribution*:
\`\`\`
Load Balancer strategy: Least connections
- Route new connection to server with fewest connections
- More even distribution than round-robin
\`\`\`

**Database Design**:

*Cassandra schema*:
\`\`\`sql
-- Messages table
CREATE TABLE messages (
  channel_id UUID,
  timestamp TIMEUUID,
  message_id UUID,
  sender_id UUID,
  content TEXT,
  PRIMARY KEY (channel_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Query pattern: Get recent messages in channel
SELECT * FROM messages 
WHERE channel_id = ? 
ORDER BY timestamp DESC 
LIMIT 50;
\`\`\`

*Why Cassandra*:
- Writes scale linearly (important for high message volume)
- Partition by channel_id (good distribution)
- Time-series data (messages naturally ordered by time)
- High availability (no single point of failure)

**Trade-offs**:

**1. Message Delivery: At-least-once vs Exactly-once**

*Chose at-least-once*:
- Pro: Simpler, more reliable
- Con: Possible duplicates (handle with idempotency)
- Exactly-once is very complex in distributed systems

**2. Gateway Layer: Thick vs Thin**

*Chose thin*:
- Gateway only handles connections, no business logic
- Pro: Scales independently, simpler
- Con: Extra network hop through Kafka

**3. Presence: Real-time vs Eventually consistent**

*Chose eventually consistent*:
- Update presence every 30 seconds (heartbeat)
- Pro: Reduces load, good enough for chat
- Con: User might show online for 30 seconds after disconnect

**4. Database: SQL vs NoSQL**

*Chose Cassandra (NoSQL)*:
- Pro: Better write scalability, natural fit for time-series
- Con: Limited query flexibility (can't do full-text search)
- Mitigation: Use Elasticsearch for search

**Monitoring**:

*Key Metrics*:
- Active WebSocket connections per server
- Messages per second
- Message delivery latency (p50, p99)
- Connection duration
- Reconnection rate
- Offline message queue size

*Alerts*:
- Connection count > 9,000 per server (approaching limit)
- Message latency > 200ms
- Reconnection rate > 5%

**Expected Performance**:
- 10M concurrent users on 1000 servers (10k per server)
- <100ms message delivery latency (p99)
- 99.9% message delivery success
- Cost: ~$50k/month infrastructure (AWS/GCP)`,
    keyPoints: [
      'Separate WebSocket gateway layer from application logic for independent scaling',
      'Use Kafka for message distribution, durability, and offline message replay',
      'Implement at-least-once delivery with client acknowledgments and idempotency',
      'Store offline messages in DynamoDB/similar with userId as partition key',
      'Use Redis for presence tracking and online user lookup',
      'Scale to 10M users with ~1000 gateway servers (10k connections each)',
      'Cassandra for message storage (good for time-series, write-heavy workload)',
      'Geographic distribution: Deploy in multiple regions, route to nearest',
    ],
  },
  {
    id: 'websocket-fallback',
    question:
      'Your application uses WebSockets for real-time updates, but you discover that 15% of your users are behind corporate firewalls that block WebSocket connections. Design a fallback strategy that maintains functionality for these users while minimizing code complexity. Discuss the trade-offs.',
    sampleAnswer: `**Problem Analysis**:

Corporate firewalls often block WebSocket because:
- Uses \`Upgrade\` header (suspicious to some firewalls)
- Long-lived connections (timeouts)
- UDP-like behavior over TCP (some DPI systems block)
- Non-standard port (if not 80/443)

**Fallback Strategy Design**:

**Tier 1: WebSocket (85% of users)**
- Best performance, lowest latency
- Full duplex communication
- Try this first

**Tier 2: HTTP/2 Server-Sent Events (SSE) (12% of users)**
- One-way server → client
- Client uses regular HTTP POST for client → server
- Works through most firewalls
- Built-in auto-reconnect

**Tier 3: Long Polling (3% of users)**
- Works everywhere (just HTTP)
- Highest latency, most overhead
- Last resort

**Implementation**:

**1. Connection Detection & Fallback**

\`\`\`javascript
class AdaptiveTransport {
  constructor(url) {
    this.url = url;
    this.transport = null;
    this.connect();
  }
  
  async connect() {
    // Try WebSocket first
    try {
      this.transport = await this.tryWebSocket();
      console.log('Using WebSocket');
      return;
    } catch (error) {
      console.log('WebSocket failed:', error);
    }
    
    // Try SSE
    try {
      this.transport = await this.trySSE();
      console.log('Using SSE');
      return;
    } catch (error) {
      console.log('SSE failed:', error);
    }
    
    // Fall back to long polling
    this.transport = this.useLongPolling();
    console.log('Using long polling');
  }
  
  tryWebSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connection timeout'));
      }, 5000);
      
      ws.onopen = () => {
        clearTimeout(timeout);
        resolve(new WebSocketTransport(ws));
      };
      
      ws.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
    });
  }
  
  trySSE() {
    return new Promise((resolve, reject) => {
      const sse = new EventSource(\`\${this.url}/sse\`);
      
      const timeout = setTimeout(() => {
        sse.close();
        reject(new Error('SSE connection timeout'));
      }, 5000);
      
      sse.onopen = () => {
        clearTimeout(timeout);
        resolve(new SSETransport(sse, this.url));
      };
      
      sse.onerror = (error) => {
        clearTimeout(timeout);
        sse.close();
        reject(error);
      };
    });
  }
  
  useLongPolling() {
    return new LongPollingTransport(this.url);
  }
  
  send(data) {
    this.transport.send(data);
  }
  
  onMessage(callback) {
    this.transport.onMessage(callback);
  }
}
\`\`\`

**2. WebSocket Transport (Primary)**

\`\`\`javascript
class WebSocketTransport {
  constructor(ws) {
    this.ws = ws;
    this.messageHandlers = [];
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    };
  }
  
  send(data) {
    this.ws.send(JSON.stringify(data));
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.ws.close();
  }
}
\`\`\`

**3. SSE Transport (Fallback 1)**

\`\`\`javascript
class SSETransport {
  constructor(sse, baseUrl) {
    this.sse = sse;
    this.baseUrl = baseUrl;
    this.messageHandlers = [];
    
    this.sse.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    };
  }
  
  async send(data) {
    // Use regular HTTP POST for client → server
    await fetch(\`\${this.baseUrl}/message\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.sse.close();
  }
}
\`\`\`

**Server-side SSE** (Node.js):
\`\`\`javascript
app.get('/sse', (req, res) => {
  // Set SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no' // Disable nginx buffering
  });
  
  // Send initial connection message
  res.write('data: {"type":"connected"}\\n\\n');
  
  // Register client for push updates
  const clientId = uuid();
  clients.set(clientId, res);
  
  // Heartbeat (keep connection alive)
  const heartbeat = setInterval(() => {
    res.write(': heartbeat\\n\\n');
  }, 30000);
  
  // Cleanup on disconnect
  req.on('close', () => {
    clearInterval(heartbeat);
    clients.delete(clientId);
  });
});

// Send message to all SSE clients
function broadcastSSE(message) {
  const data = \`data: \${JSON.stringify(message)}\\n\\n\`;
  clients.forEach(client => {
    client.write(data);
  });
}
\`\`\`

**4. Long Polling Transport (Fallback 2)**

\`\`\`javascript
class LongPollingTransport {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.messageHandlers = [];
    this.polling = true;
    this.startPolling();
  }
  
  async startPolling() {
    while (this.polling) {
      try {
        const response = await fetch(\`\${this.baseUrl}/poll\`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        
        if (response.ok) {
          const messages = await response.json();
          messages.forEach(msg => {
            this.messageHandlers.forEach(handler => handler(msg));
          });
        }
      } catch (error) {
        console.error('Polling error:', error);
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  }
  
  async send(data) {
    await fetch(\`\${this.baseUrl}/message\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.polling = false;
  }
}
\`\`\`

**Server-side long polling**:
\`\`\`javascript
const messageQueues = new Map(); // userId → queue of messages

app.get('/poll', async (req, res) => {
  const userId = req.user.id;
  
  // Check if messages available
  const queue = messageQueues.get(userId) || [];
  
  if (queue.length > 0) {
    // Send immediately
    res.json(queue);
    messageQueues.set(userId, []);
  } else {
    // Hold connection until message arrives (max 30 seconds)
    const timeout = setTimeout(() => {
      res.json([]);
    }, 30000);
    
    // Register callback for when message arrives
    pendingRequests.set(userId, {
      res,
      timeout
    });
  }
});

// When message arrives
function deliverMessage(userId, message) {
  // Check if user has pending long poll request
  const pending = pendingRequests.get(userId);
  
  if (pending) {
    // Deliver immediately
    clearTimeout(pending.timeout);
    pending.res.json([message]);
    pendingRequests.delete(userId);
  } else {
    // Queue for next poll
    const queue = messageQueues.get(userId) || [];
    queue.push(message);
    messageQueues.set(userId, queue);
  }
}
\`\`\`

**5. Unified Application Interface**

\`\`\`javascript
// Application code doesn't know which transport is used
const transport = new AdaptiveTransport('wss://example.com/socket');

// Send messages (works with all transports)
transport.send({
  type: 'chat',
  message: 'Hello'
});

// Receive messages (works with all transports)
transport.onMessage((message) => {
  console.log('Received:', message);
});
\`\`\`

**Trade-offs Analysis**:

**1. Code Complexity**

*Increased*:
- Three transport implementations
- Connection detection logic
- Server must support all three protocols

*Mitigated by*:
- Shared interface (AdaptiveTransport)
- Transport-specific logic encapsulated
- Use libraries (socket.io handles this automatically)

**2. Server Resources**

*WebSocket*:
- 1 connection per user
- Minimal overhead after handshake

*SSE*:
- 1 connection per user (server → client)
- + HTTP POST for client → server
- Slightly more overhead

*Long Polling*:
- 1 active HTTP request per user constantly
- Much higher overhead
- More server resources (worker threads/processes)

*Cost*:
- 100k WebSocket users: 100 servers
- 100k long polling users: 300 servers
- 15% long polling = +30% server cost

**3. Latency**

*WebSocket*: 10-50ms
*SSE*: 20-100ms (includes HTTP POST roundtrip for sending)
*Long Polling*: 100-500ms (worst case: timeout + new request)

*Impact*:
- Chat feels slightly less real-time for long polling users
- Acceptable trade-off vs no functionality

**4. Battery Usage (Mobile)**

*WebSocket*: Best (single persistent connection)
*SSE*: Good (one persistent connection + occasional POST)
*Long Polling*: Poor (constant HTTP requests)

*Mitigation*:
- Increase polling interval on mobile (5-10 seconds)
- Use adaptive polling (poll faster during active use)

**5. Firewall Compatibility**

| Transport | Corporate Firewall | Hotel WiFi | Mobile Network | China Great Firewall |
|-----------|-------------------|------------|----------------|---------------------|
| WebSocket | 85% | 95% | 98% | 60% |
| SSE | 95% | 99% | 99% | 80% |
| Long Polling | 100% | 100% | 100% | 95% |

**Advanced Optimization: Adaptive Polling**

For long polling, don't poll constantly:

\`\`\`javascript
class AdaptiveLongPolling {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.pollInterval = 30000; // Start with 30 seconds
    this.active = false;
  }
  
  async poll() {
    const messages = await fetch(\`\${this.baseUrl}/poll\`);
    
    if (messages.length > 0) {
      // Activity detected, poll faster
      this.pollInterval = 1000; // 1 second
      this.deliverMessages(messages);
    } else {
      // No activity, slow down (exponential backoff)
      this.pollInterval = Math.min(this.pollInterval * 1.5, 30000);
    }
    
    setTimeout(() => this.poll(), this.pollInterval);
  }
  
  onUserInteraction() {
    // User is typing, poll faster
    this.pollInterval = 1000;
  }
}
\`\`\`

**Monitoring & Analytics**:

Track transport usage:
\`\`\`javascript
analytics.track('transport_type', {
  type: 'websocket', // or 'sse' or 'long_polling'
  userAgent: navigator.userAgent,
  region: userRegion
});
\`\`\`

Use this data to:
- Understand firewall blocking patterns
- Optimize server resource allocation
- Identify problematic networks/regions

**Recommendation: Use socket.io**

Socket.io library handles all this automatically:
\`\`\`javascript
// Client
const socket = io('https://example.com');
// Automatically tries: WebSocket → long polling

// Server
const io = require('socket.io')(server);
io.on('connection', (socket) => {
  // Same code for all transports
});
\`\`\`

**Final Architecture**:

\`\`\`
Client
  ↓
[Auto-detect best transport]
  ↓
├─ WebSocket (85%)
├─ SSE + HTTP POST (12%)
└─ Long Polling (3%)
  ↓
Server (handles all three)
  ↓
[Application logic (transport-agnostic)]
\`\`\`

**Expected Results**:
- 100% of users can use the application (vs 85% with WebSocket only)
- +20-30% server cost (for fallback support)
- Slightly higher latency for 15% of users (acceptable)
- Minimal code complexity increase (if using libraries)

This provides universal compatibility while maintaining good performance for the majority.`,
    keyPoints: [
      'Implement tiered fallback: WebSocket → SSE → Long Polling',
      'Auto-detect best available transport on connection',
      'SSE works for most firewalls, provides one-way server → client with HTTP POST for reverse',
      'Long Polling is last resort, works everywhere but higher latency and server cost',
      'Use unified interface so application code is transport-agnostic',
      'Monitor transport usage to understand blocking patterns',
      'Consider socket.io library which handles fallback automatically',
      'Trade-off: +20-30% server cost for 100% compatibility vs 85% with WebSocket only',
    ],
  },
  {
    id: 'websocket-security',
    question:
      'Explain the security vulnerabilities specific to WebSocket connections and how you would mitigate them. Include authentication, authorization, rate limiting, and protection against common attacks. How would you handle connection hijacking?',
    sampleAnswer: `**WebSocket-Specific Security Challenges**:

Unlike HTTP, WebSocket:
- Long-lived connections (more time for attacks)
- Bidirectional (attack surface in both directions)
- Bypasses some traditional security tools (HTTP-only firewalls, proxies)
- Less mature security ecosystem than HTTP

**1. Authentication Vulnerabilities**

**Problem 1: Credentials in URL**

\`\`\`javascript
// VULNERABLE
const ws = new WebSocket('wss://example.com/socket?token=secret123');
\`\`\`

*Why bad*:
- Tokens logged in server logs
- Visible in proxy logs
- Sent in HTTP headers during upgrade (cleartext if not WSS)
- Could be leaked in referrer headers

**Solution: Authenticate after connection**

\`\`\`javascript
// Client
const ws = new WebSocket('wss://example.com/socket');

ws.onopen = () => {
  // Send auth token as first message
  ws.send(JSON.stringify({
    type: 'auth',
    token: localStorage.getItem('authToken')
  }));
};

// Server
ws.on('message', async (message) => {
  const data = JSON.parse(message);
  
  if (!ws.authenticated) {
    if (data.type !== 'auth') {
      ws.close(4001, 'Authentication required');
      return;
    }
    
    // Validate token
    try {
      const user = await validateToken(data.token);
      ws.authenticated = true;
      ws.userId = user.id;
      ws.send(JSON.stringify({ type: 'auth_success' }));
    } catch (error) {
      ws.close(4001, 'Invalid token');
    }
    return;
  }
  
  // Handle other messages
  handleMessage(ws, data);
});
\`\`\`

**Problem 2: Session Fixation**

*Attack*: Attacker creates WebSocket connection, gets session ID, tricks victim into using it.

**Solution: Bind session to user**
\`\`\`javascript
// Generate new session ID after authentication
ws.on('auth_success', () => {
  const oldSessionId = ws.sessionId;
  ws.sessionId = generateNewSessionId();
  
  // Invalidate old session
  sessions.delete(oldSessionId);
  sessions.set(ws.sessionId, { userId: ws.userId });
});
\`\`\`

**Problem 3: Token Expiration**

Long-lived WebSocket connections can outlive token expiration.

**Solution: Periodic re-authentication**
\`\`\`javascript
// Client: Periodically refresh token
setInterval(async () => {
  const newToken = await refreshAuthToken();
  ws.send(JSON.stringify({
    type: 'refresh_auth',
    token: newToken
  }));
}, 15 * 60 * 1000); // Every 15 minutes

// Server: Check token expiry
setInterval(() => {
  if (tokenExpired(ws.token)) {
    ws.close(4001, 'Token expired');
  }
}, 60 * 1000); // Check every minute
\`\`\`

**2. Authorization Vulnerabilities**

**Problem: Insufficient authorization checks**

\`\`\`javascript
// VULNERABLE
ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // No authorization check!
  if (data.type === 'read_chat') {
    const messages = getMessages(data.channelId);
    ws.send(JSON.stringify(messages));
  }
});
\`\`\`

**Solution: Authorize every action**
\`\`\`javascript
ws.on('message', async (message) => {
  const data = JSON.parse(message);
  
  // Always check authorization
  if (data.type === 'read_chat') {
    // Verify user has access to this channel
    const hasAccess = await checkChannelAccess(ws.userId, data.channelId);
    if (!hasAccess) {
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Access denied'
      }));
      return;
    }
    
    const messages = getMessages(data.channelId);
    ws.send(JSON.stringify(messages));
  }
});
\`\`\`

**Best Practice: Authorization middleware**
\`\`\`javascript
async function requireChannelAccess(ws, channelId) {
  const hasAccess = await checkChannelAccess(ws.userId, channelId);
  if (!hasAccess) {
    throw new Error('Access denied');
  }
}

// Use in handlers
ws.on('message', async (message) => {
  try {
    const data = JSON.parse(message);
    
    if (data.type === 'read_chat') {
      await requireChannelAccess(ws, data.channelId);
      const messages = getMessages(data.channelId);
      ws.send(JSON.stringify(messages));
    }
  } catch (error) {
    ws.send(JSON.stringify({ type: 'error', message: error.message }));
  }
});
\`\`\`

**3. Rate Limiting**

**Problem: DoS via message flooding**

\`\`\`javascript
// Attacker sends 10,000 messages per second
for (let i = 0; i < 10000; i++) {
  ws.send('spam');
}
\`\`\`

**Solution 1: Per-connection rate limiting**
\`\`\`javascript
class RateLimiter {
  constructor(maxMessages, windowMs) {
    this.maxMessages = maxMessages; // e.g., 100
    this.windowMs = windowMs; // e.g., 1000ms
    this.counters = new Map(); // connectionId → [timestamps]
  }
  
  check(connectionId) {
    const now = Date.now();
    
    if (!this.counters.has(connectionId)) {
      this.counters.set(connectionId, []);
    }
    
    const timestamps = this.counters.get(connectionId);
    
    // Remove old timestamps
    const recent = timestamps.filter(t => now - t < this.windowMs);
    
    if (recent.length >= this.maxMessages) {
      return false; // Rate limit exceeded
    }
    
    recent.push(now);
    this.counters.set(connectionId, recent);
    
    return true;
  }
}

const rateLimiter = new RateLimiter(100, 1000); // 100 messages per second

ws.on('message', (message) => {
  if (!rateLimiter.check(ws.id)) {
    ws.send(JSON.stringify({ type: 'error', message: 'Rate limit exceeded' }));
    // Optionally close connection after repeated violations
    ws.violationCount = (ws.violationCount || 0) + 1;
    if (ws.violationCount > 5) {
      ws.close(4008, 'Rate limit exceeded');
    }
    return;
  }
  
  handleMessage(message);
});
\`\`\`

**Solution 2: Per-user rate limiting (across all connections)**
\`\`\`javascript
// Use Redis for distributed rate limiting
async function checkRateLimit(userId) {
  const key = \`ratelimit:\${userId}\`;
  const count = await redis.incr(key);
  
  if (count === 1) {
    // Set expiry on first request
    await redis.expire(key, 1); // 1 second window
  }
  
  return count <= 100; // Max 100 messages per second
}

ws.on('message', async (message) => {
  const allowed = await checkRateLimit(ws.userId);
  if (!allowed) {
    ws.send(JSON.stringify({ type: 'error', message: 'Rate limit exceeded' }));
    return;
  }
  
  handleMessage(message);
});
\`\`\`

**4. Cross-Site WebSocket Hijacking (CSWSH)**

**Attack**: Malicious site opens WebSocket to your server using victim's cookies.

\`\`\`html
<!-- evil.com -->
<script>
  // Opens WebSocket to victim.com
  // Browser automatically sends cookies!
  const ws = new WebSocket('wss://victim.com/socket');
  ws.onmessage = (e) => {
    // Steal victim's messages
    sendToAttacker(e.data);
  };
</script>
\`\`\`

**Solution: Check Origin header**
\`\`\`javascript
wss.on('connection', (ws, req) => {
  const origin = req.headers.origin;
  
  // Whitelist of allowed origins
  const allowedOrigins = [
    'https://example.com',
    'https://app.example.com'
  ];
  
  if (!allowedOrigins.includes(origin)) {
    ws.close(4003, 'Origin not allowed');
    return;
  }
  
  // Continue with connection
});
\`\`\`

**Additional protection: Custom header**
\`\`\`javascript
// Client: Add custom header during upgrade
const ws = new WebSocket('wss://example.com/socket', {
  headers: {
    'X-Custom-Auth': 'your-secret-token'
  }
});

// Server: Verify custom header
wss.on('connection', (ws, req) => {
  const customAuth = req.headers['x-custom-auth',];
  if (customAuth !== expectedToken) {
    ws.close(4003, 'Invalid auth header');
    return;
  }
});
\`\`\`

**5. Message Injection**

**Problem: Unsanitized user input**

\`\`\`javascript
// VULNERABLE
ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // Send to all users without sanitization
  broadcast({
    type: 'chat',
    message: data.message, // Could contain XSS payload
    user: data.user
  });
});
\`\`\`

**Solution: Validate and sanitize**
\`\`\`javascript
const sanitizeHtml = require('sanitize-html');

ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // Validate structure
  if (!data.type || !data.message) {
    ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format' }));
    return;
  }
  
  // Validate message type
  const allowedTypes = ['chat', 'typing', 'read',];
  if (!allowedTypes.includes(data.type)) {
    ws.send(JSON.stringify({ type: 'error', message: 'Invalid message type' }));
    return;
  }
  
  // Sanitize HTML
  const cleanMessage = sanitizeHtml(data.message, {
    allowedTags: ['b', 'i', 'em', 'strong',],
    allowedAttributes: {}
  });
  
  // Validate length
  if (cleanMessage.length > 1000) {
    ws.send(JSON.stringify({ type: 'error', message: 'Message too long' }));
    return;
  }
  
  broadcast({
    type: 'chat',
    message: cleanMessage,
    user: ws.userId, // Use server-side userId, not client-provided
    timestamp: Date.now() // Server-generated timestamp
  });
});
\`\`\`

**6. Connection Hijacking**

**Attack Scenario**: Attacker intercepts WebSocket connection and sends commands as victim.

**Mitigation 1: Use WSS (WebSocket Secure)**
\`\`\`javascript
// Always use wss:// (not ws://)
const ws = new WebSocket('wss://example.com/socket');
\`\`\`

*Why*:
- Encrypts all traffic (like HTTPS)
- Prevents man-in-the-middle attacks
- Certificate validation

**Mitigation 2: Message signatures**
\`\`\`javascript
// Client: Sign each message
function sendSecure(ws, data) {
  const message = JSON.stringify(data);
  const signature = hmacSHA256(message, userSecret);
  
  ws.send(JSON.stringify({
    message,
    signature
  }));
}

// Server: Verify signature
ws.on('message', (payload) => {
  const { message, signature } = JSON.parse(payload);
  
  const userSecret = getUserSecret(ws.userId);
  const expectedSignature = hmacSHA256(message, userSecret);
  
  if (signature !== expectedSignature) {
    ws.close(4004, 'Invalid signature');
    return;
  }
  
  handleMessage(JSON.parse(message));
});
\`\`\`

**Mitigation 3: Mutual TLS (mTLS)**

For high-security applications (B2B, financial):
\`\`\`javascript
const https = require('https');
const fs = require('fs');

const server = https.createServer({
  cert: fs.readFileSync('server-cert.pem'),
  key: fs.readFileSync('server-key.pem'),
  ca: fs.readFileSync('ca-cert.pem'),
  requestCert: true,
  rejectUnauthorized: true
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  // Client certificate validated by TLS
  const clientCert = req.socket.getPeerCertificate();
  console.log('Client:', clientCert.subject.CN);
});
\`\`\`

**7. Denial of Service (DoS)**

**Attack 1: Connection exhaustion**
\`\`\`javascript
// Attacker opens 10,000 connections
for (let i = 0; i < 10000; i++) {
  new WebSocket('wss://example.com/socket');
}
\`\`\`

**Solution: Connection limits**
\`\`\`javascript
const connectionsPerIP = new Map();

wss.on('connection', (ws, req) => {
  const ip = req.socket.remoteAddress;
  
  const count = connectionsPerIP.get(ip) || 0;
  
  // Max 10 connections per IP
  if (count >= 10) {
    ws.close(4009, 'Too many connections from this IP');
    return;
  }
  
  connectionsPerIP.set(ip, count + 1);
  
  ws.on('close', () => {
    connectionsPerIP.set(ip, connectionsPerIP.get(ip) - 1);
  });
});
\`\`\`

**Attack 2: Large message DoS**
\`\`\`javascript
// Send 100MB message
ws.send('x'.repeat(100 * 1024 * 1024));
\`\`\`

**Solution: Message size limit**
\`\`\`javascript
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    // Limit message size to 1MB
    if (message.length > 1024 * 1024) {
      ws.close(4010, 'Message too large');
      return;
    }
    
    handleMessage(message);
  });
});
\`\`\`

**8. Monitoring & Alerting**

\`\`\`javascript
// Track suspicious activity
class SecurityMonitor {
  constructor() {
    this.violations = new Map(); // userId → violation count
  }
  
  recordViolation(userId, type) {
    const key = \`\${userId}:\${type}\`;
    const count = this.violations.get(key) || 0;
    this.violations.set(key, count + 1);
    
    // Alert if threshold exceeded
    if (count > 10) {
      alertSecurity(\`User \${userId} has \${count} \${type} violations\`);
      
      // Optionally ban user
      if (count > 50) {
        banUser(userId);
      }
    }
  }
}

const monitor = new SecurityMonitor();

// Use in handlers
ws.on('message', (message) => {
  if (!rateLimiter.check(ws.id)) {
    monitor.recordViolation(ws.userId, 'rate_limit');
    return;
  }
  
  try {
    const data = JSON.parse(message);
    handleMessage(data);
  } catch (error) {
    monitor.recordViolation(ws.userId, 'invalid_message');
  }
});
\`\`\`

**Security Checklist**:

✅ Use WSS (encrypted), not WS  
✅ Authenticate after connection (not in URL)  
✅ Validate Origin header (prevent CSWSH)  
✅ Authorize every action  
✅ Validate and sanitize all messages  
✅ Implement rate limiting (per connection and per user)  
✅ Limit message size  
✅ Limit connections per IP  
✅ Implement heartbeat and connection timeout  
✅ Use server-generated data (timestamps, user IDs)  
✅ Monitor and alert on suspicious activity  
✅ Regularly rotate tokens/secrets  
✅ Log security events (failed auth, rate limits)  

**Defense-in-Depth**: Multiple layers of security make the system resilient to attacks even if one layer is breached.`,
    keyPoints: [
      'Never put credentials in WebSocket URL; authenticate after connection establishment',
      'Check Origin header to prevent Cross-Site WebSocket Hijacking (CSWSH)',
      'Implement rate limiting at both connection and user level to prevent DoS',
      'Always validate and sanitize messages; use server-generated data (userId, timestamp)',
      'Use WSS (encrypted) always; consider mTLS for high-security applications',
      "Authorize every action; don't assume authenticated user has access to all resources",
      'Limit connection count per IP and message size to prevent resource exhaustion',
      'Monitor violations and alert on suspicious activity; implement automatic banning',
    ],
  },
];
