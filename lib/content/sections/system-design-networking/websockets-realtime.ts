/**
 * WebSockets & Real-Time Communication Section
 */

export const websocketsrealtimeSection = {
  id: 'websockets-realtime',
  title: 'WebSockets & Real-Time Communication',
  content: `WebSockets enable real-time, bidirectional communication between client and server. Understanding when and how to use WebSockets is essential for building modern interactive applications.

## The Problem with HTTP for Real-Time

Traditional HTTP is **request-response**: Client initiates, server responds.

**Problem**: How do you push updates from server to client?

### **Attempt 1: Polling**

Client repeatedly asks server for updates:
\`\`\`
Every 5 seconds:
  Client → Server: "Any updates?"
  Server → Client: "No" (or "Yes, here's data")
\`\`\`

**Issues**:
- ❌ Wasteful (mostly "no" responses)
- ❌ Latency (up to 5 seconds delay)
- ❌ High server load (many unnecessary requests)

### **Attempt 2: Long Polling**

Client asks, server holds connection until update available:
\`\`\`
Client → Server: "Any updates?"
[Server holds connection open]
[Update becomes available]
Server → Client: "Yes, here's data"
Client → Server: "Any updates?" (immediately reconnect)
\`\`\`

**Better, but**:
- ❌ Still request-response pattern
- ❌ Connection overhead (HTTP headers every time)
- ❌ Complicated server-side (hold many connections)

### **Attempt 3: Server-Sent Events (SSE)**

Server pushes updates over single HTTP connection:
\`\`\`
Client → Server: "Start streaming updates"
Server → Client: [Data stream, one-way]
\`\`\`

**Better for one-way**, but:
- ❌ One-way only (server → client)
- ❌ Client must use HTTP for sending data
- ❌ Limited browser connection limits (6 per domain)

---

## WebSocket Solution

**WebSocket** is a **full-duplex** protocol over a single TCP connection.

**Full-duplex** = Both parties can send data simultaneously, anytime.

\`\`\`
Client ←→ Server
  Both can send messages anytime
  Single persistent TCP connection
  Low overhead (no HTTP headers per message)
\`\`\`

---

## WebSocket Protocol

### How WebSocket Works

**1. Starts as HTTP (Upgrade Handshake)**

Client initiates with HTTP request:
\`\`\`
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
\`\`\`

Server responds:
\`\`\`
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
\`\`\`

**Status 101**: Switching from HTTP to WebSocket protocol

**2. Connection Upgraded**

After handshake, same TCP connection now used for WebSocket protocol.

**3. Full-Duplex Communication**

\`\`\`
Client → Server: "Hello"
Server → Client: "Hi there"
Client → Server: "How are you?"
Server → Client: "I'm good"
... anytime, either direction ...
\`\`\`

---

## WebSocket Frame Structure

WebSocket sends data in **frames** (not HTTP messages):

\`\`\`
Frame structure:
[FIN|RSV|Opcode|Mask|PayloadLen|MaskingKey|Payload]

FIN: Is this the final fragment?
Opcode: 
  0x1 = Text frame
  0x2 = Binary frame
  0x8 = Connection close
  0x9 = Ping
  0xA = Pong
Mask: Is payload masked? (client→server: yes, server→client: no)
Payload: Actual data
\`\`\`

**Overhead**: 2-14 bytes per frame (vs 100s of bytes for HTTP)

---

## WebSocket vs HTTP Comparison

| Feature | HTTP | WebSocket |
|---------|------|-----------|
| **Direction** | Request-response | Full-duplex |
| **Latency** | High (new request each time) | Low (persistent connection) |
| **Overhead** | High (headers every request) | Low (small frame headers) |
| **Server push** | Not native (need workarounds) | Native support |
| **Connection** | Short-lived | Long-lived |
| **Use case** | Request data on demand | Real-time bidirectional updates |

---

## When to Use WebSockets

Use WebSockets when:
- **Real-time updates** needed
- **Bidirectional** communication
- **Low latency** critical
- **High message frequency**

### Use Cases

**1. Chat Applications**
- Messages sent both directions
- Low latency expected
- High message frequency
- Example: Slack, Discord, WhatsApp Web

**2. Live Notifications**
- Server pushes notifications to client
- Immediate delivery expected
- Example: Facebook notifications, Twitter feed updates

**3. Collaborative Editing**
- Multiple users editing same document
- Changes must propagate immediately
- Example: Google Docs, Figma

**4. Live Sports Scores**
- Server pushes score updates
- Multiple updates per second
- Low latency critical

**5. Trading Platforms**
- Stock prices update in real-time
- Orders submitted immediately
- Example: Robinhood, E*TRADE

**6. Online Gaming**
- Player actions sent both directions
- Sub-100ms latency required
- High message frequency (60+ per second)

**7. IoT Dashboards**
- Sensor data streams to dashboard
- Control commands sent back
- Example: Smart home apps

---

## When NOT to Use WebSockets

**Use HTTP/REST instead** when:
- Occasional data fetching
- One-time requests
- Can tolerate latency
- Need caching (HTTP caches well, WebSocket doesn't)

**Examples**:
- Fetching user profile (occasional)
- Submitting a form (one-time)
- Loading product catalog (can cache)

---

## WebSocket Implementation

### Client-Side (JavaScript)

\`\`\`javascript
// Connect to WebSocket server
const ws = new WebSocket('wss://example.com/socket');

// Connection opened
ws.addEventListener('open', (event) => {
  console.log('Connected to server');
  ws.send('Hello Server!');
});

// Receive message
ws.addEventListener('message', (event) => {
  console.log('Message from server:', event.data);
  
  // Parse JSON if needed
  const data = JSON.parse (event.data);
  console.log (data);
});

// Handle errors
ws.addEventListener('error', (event) => {
  console.error('WebSocket error:', event);
});

// Connection closed
ws.addEventListener('close', (event) => {
  console.log('Disconnected:', event.code, event.reason);
  
  // Reconnect logic
  setTimeout(() => {
    console.log('Reconnecting...');
    // Recreate connection
  }, 5000);
});

// Send message
ws.send('Hello');
ws.send(JSON.stringify({ type: 'chat', message: 'Hello' }));

// Close connection
ws.close();
\`\`\`

### Server-Side (Node.js with ws library)

\`\`\`javascript
const WebSocket = require('ws');

// Create WebSocket server
const wss = new WebSocket.Server({ port: 8080 });

// Track connected clients
const clients = new Set();

wss.on('connection', (ws) => {
  console.log('Client connected');
  clients.add (ws);
  
  // Send welcome message
  ws.send(JSON.stringify({ type: 'welcome', message: 'Connected!' }));
  
  // Receive message from client
  ws.on('message', (message) => {
    console.log('Received:', message);
    
    // Broadcast to all clients
    clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send (message);
      }
    });
  });
  
  // Client disconnected
  ws.on('close', () => {
    console.log('Client disconnected');
    clients.delete (ws);
  });
  
  // Handle errors
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
  
  // Heartbeat (detect broken connections)
  ws.isAlive = true;
  ws.on('pong', () => {
    ws.isAlive = true;
  });
});

// Heartbeat interval
setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      return ws.terminate();
    }
    
    ws.isAlive = false;
    ws.ping();
  });
}, 30000);
\`\`\`

---

## Scaling WebSockets

### Challenge: Stateful Connections

Unlike HTTP (stateless), WebSocket connections are **stateful**:
- Server holds connection state
- Can't easily load balance

**Problem**:
\`\`\`
User A connects to Server 1
User B connects to Server 2

User A sends message to User B
→ Server 1 doesn't know User B is on Server 2
\`\`\`

### Solution 1: Sticky Sessions

Route user to same server every time:

\`\`\`
Load Balancer (sticky by user ID)
    ↓                ↓
 Server 1         Server 2
 (Users A, C)     (Users B, D)
\`\`\`

**Pros**:
- Simple
- No coordination needed

**Cons**:
- Uneven load distribution
- Can't survive server restart

### Solution 2: Message Broker (Recommended)

Use pub/sub system to coordinate between servers:

\`\`\`
Server 1 ← User A
  ↓
Redis Pub/Sub
  ↓
Server 2 ← User B
\`\`\`

**Flow**:
1. User A sends message on Server 1
2. Server 1 publishes to Redis channel
3. Server 2 (subscribed to channel) receives message
4. Server 2 sends to User B

**Implementation** (Node.js):
\`\`\`javascript
const Redis = require('ioredis');
const pub = new Redis();
const sub = new Redis();

// Subscribe to channel
sub.subscribe('chat');

// Receive from Redis, send to local clients
sub.on('message', (channel, message) => {
  const data = JSON.parse (message);
  
  // Send to all local WebSocket clients
  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send (message);
    }
  });
});

// Receive from WebSocket client, publish to Redis
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    // Publish to Redis
    pub.publish('chat', message);
  });
});
\`\`\`

**Pros**:
- Scales horizontally
- Servers are stateless (can restart)
- Even load distribution

**Cons**:
- Added complexity
- Redis is single point of failure (use Redis Cluster)

### Solution 3: Dedicated WebSocket Layer

Separate WebSocket servers from application logic:

\`\`\`
Clients
  ↓
WebSocket Servers (handle connections only)
  ↓
Message Queue (Kafka/RabbitMQ)
  ↓
Application Servers (business logic)
  ↓
Database
\`\`\`

**Benefits**:
- Optimize WebSocket servers for connections (10k+ per server)
- Scale application servers independently
- Better resource utilization

---

## Connection Management

### Heartbeat / Ping-Pong

Detect broken connections:

\`\`\`javascript
// Server sends ping every 30 seconds
setInterval(() => {
  ws.ping();
}, 30000);

// If no pong received within timeout, connection is dead
ws.on('pong', () => {
  // Connection still alive
});
\`\`\`

**Why needed**:
- TCP doesn't immediately detect broken connections
- Router/firewall timeouts
- Mobile networks frequently disconnect

### Automatic Reconnection (Client-Side)

\`\`\`javascript
class ReconnectingWebSocket {
  constructor (url) {
    this.url = url;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.connect();
  }
  
  connect() {
    this.ws = new WebSocket (this.url);
    
    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectDelay = 1000; // Reset delay
    };
    
    this.ws.onclose = () => {
      console.log(\`Disconnected. Reconnecting in \${this.reconnectDelay}ms\`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay);
      
      // Exponential backoff
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxReconnectDelay
      );
    };
    
    this.ws.onerror = (error) => {
      console.error('Error:', error);
      this.ws.close();
    };
  }
  
  send (data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send (data);
    } else {
      console.warn('WebSocket not connected');
    }
  }
}
\`\`\`

### Message Queuing (Client-Side)

Queue messages if connection drops:

\`\`\`javascript
class QueuedWebSocket {
  constructor (url) {
    this.queue = [];
    this.ws = new WebSocket (url);
    
    this.ws.onopen = () => {
      // Flush queue
      while (this.queue.length > 0) {
        this.ws.send (this.queue.shift());
      }
    };
  }
  
  send (data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send (data);
    } else {
      // Queue for later
      this.queue.push (data);
    }
  }
}
\`\`\`

---

## Security Considerations

### 1. Authentication

**Don't put credentials in URL**:
\`\`\`javascript
// Bad
new WebSocket('wss://example.com/socket?token=secret123');
\`\`\`

**Use initial handshake**:
\`\`\`javascript
// Good
const ws = new WebSocket('wss://example.com/socket');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: localStorage.getItem('token')
  }));
};

// Server validates token before accepting messages
\`\`\`

### 2. Message Validation

Always validate messages:
\`\`\`javascript
ws.on('message', (message) => {
  try {
    const data = JSON.parse (message);
    
    // Validate structure
    if (!data.type || !data.payload) {
      throw new Error('Invalid message format');
    }
    
    // Validate type
    const allowedTypes = ['chat', 'typing', 'presence'];
    if (!allowedTypes.includes (data.type)) {
      throw new Error('Invalid message type');
    }
    
    // Process message
    handleMessage (data);
  } catch (error) {
    console.error('Invalid message:', error);
    ws.close(1008, 'Invalid message');
  }
});
\`\`\`

### 3. Rate Limiting

Prevent abuse:
\`\`\`javascript
const rateLimits = new Map();

ws.on('message', (message) => {
  const userId = ws.userId;
  const now = Date.now();
  
  if (!rateLimits.has (userId)) {
    rateLimits.set (userId, []);
  }
  
  const timestamps = rateLimits.get (userId);
  
  // Remove old timestamps (older than 1 minute)
  timestamps = timestamps.filter (t => now - t < 60000);
  
  // Check limit (max 100 messages per minute)
  if (timestamps.length >= 100) {
    ws.close(1008, 'Rate limit exceeded');
    return;
  }
  
  timestamps.push (now);
  rateLimits.set (userId, timestamps);
  
  // Process message
  handleMessage (message);
});
\`\`\`

### 4. Use WSS (WebSocket Secure)

Always use \`wss://\` (encrypted) in production, not \`ws://\`.

---

## WebSocket Alternatives

### Server-Sent Events (SSE)

**When to use**:
- One-way communication (server → client only)
- Simple updates
- Built-in auto-reconnect
- Easier to implement

**Example**:
\`\`\`javascript
// Client
const eventSource = new EventSource('/events');

eventSource.onmessage = (event) => {
  console.log('Update:', event.data);
};

// Server (Node.js)
res.writeHead(200, {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive'
});

setInterval(() => {
  res.write(\`data: \${JSON.stringify({ time: Date.now() })}\\n\\n\`);
}, 1000);
\`\`\`

**Pros**:
- Simpler than WebSocket
- Auto-reconnect built-in
- Works over HTTP (easier with proxies)

**Cons**:
- One-way only
- Less efficient than WebSocket
- Browser connection limits (6 per domain)

### gRPC Streaming

**When to use**:
- Microservices communication
- Bidirectional streaming
- Need strong typing (Protocol Buffers)

**Pros**:
- Efficient binary protocol
- Strong typing
- Code generation

**Cons**:
- More complex setup
- Not browser-native (need grpc-web)

---

## Real-World Examples

### Discord

- **35 million concurrent WebSocket connections**
- Uses Erlang for WebSocket gateway servers
- Redis pub/sub for message distribution
- Custom protocol over WebSocket (not plain JSON)

**Architecture**:
\`\`\`
Discord App ← WebSocket → Gateway Servers → Redis → Backend Services
\`\`\`

### Slack

- **10+ million concurrent WebSocket connections**
- WebSocket for real-time messages
- HTTP REST API for message history
- Falls back to long polling if WebSocket unavailable

### Trading Platforms

- **Stock price updates**: WebSocket
- **Order submission**: WebSocket with acknowledgment
- Sub-100ms latency requirement

---

## Best Practices

### 1. Graceful Degradation

Provide fallback if WebSocket unavailable:
\`\`\`javascript
if ('WebSocket' in window) {
  // Use WebSocket
  useWebSocket();
} else {
  // Fall back to long polling
  useLongPolling();
}
\`\`\`

### 2. Message Format

Use structured messages:
\`\`\`javascript
{
  "type": "chat",
  "payload": {
    "message": "Hello",
    "timestamp": 1640000000
  }
}
\`\`\`

### 3. Implement Heartbeat

Both client and server should send periodic pings.

### 4. Automatic Reconnection

Client should auto-reconnect with exponential backoff.

### 5. Connection Limits

Limit connections per user (prevent abuse):
\`\`\`javascript
if (connectionsPerUser.get (userId) >= 5) {
  ws.close(1008, 'Too many connections');
}
\`\`\`

### 6. Monitor Performance

Track:
- Active connections
- Messages per second
- Connection duration
- Reconnection rate
- Error rate

---

## Common Mistakes

### ❌ Not Handling Reconnection

\`\`\`javascript
// Bad: No reconnection logic
const ws = new WebSocket (url);
\`\`\`

Always implement auto-reconnect with exponential backoff.

### ❌ Storing Too Much State Per Connection

\`\`\`javascript
// Bad: Storing entire user profile per connection
ws.userProfile = { /* 100 KB of data */ };
\`\`\`

Keep connection state minimal; fetch data as needed.

### ❌ Not Implementing Heartbeat

Without heartbeat, dead connections accumulate.

### ❌ Assuming Messages Arrive

WebSocket delivers messages, but network issues happen:
\`\`\`javascript
// Add message IDs and acknowledgments
ws.send(JSON.stringify({
  id: uuid(),
  type: 'chat',
  message: 'Hello'
}));
\`\`\`

---

## Interview Tips

### Question: "When would you use WebSockets vs HTTP?"

**Good answer**:
- WebSocket for real-time, bidirectional, high-frequency communication
- Examples: Chat, live notifications, collaborative editing
- HTTP for occasional requests, caching needed, one-time operations
- Trade-off: WebSocket more complex (stateful connections, harder to scale)

### Question: "How do you scale WebSockets?"

**Hit these points**:
1. WebSocket connections are stateful (can't easily load balance)
2. Use sticky sessions OR message broker (Redis pub/sub)
3. Message broker preferred: publish messages, all servers receive
4. Separate WebSocket layer from application logic
5. Monitor connection count, implement rate limiting

### Question: "How do you handle WebSocket connection failures?"

1. Client: Auto-reconnect with exponential backoff
2. Client: Queue messages if disconnected
3. Server: Heartbeat to detect dead connections
4. Server: Clean up resources when connection closes
5. Consider message acknowledgments for critical messages

---

## Key Takeaways

1. **WebSocket** enables full-duplex, real-time communication over single TCP connection
2. **Use cases**: Chat, live notifications, collaborative editing, trading platforms
3. **Scaling**: Use message broker (Redis pub/sub) to coordinate between servers
4. **Connection management**: Implement heartbeat, auto-reconnect, message queuing
5. **Security**: Validate messages, rate limiting, use WSS (encrypted)
6. **Alternatives**: SSE for one-way, gRPC for microservices
7. **Trade-off**: Real-time capability vs complexity of stateful connections
8. **Best practice**: Graceful degradation, structured messages, monitoring`,
};
