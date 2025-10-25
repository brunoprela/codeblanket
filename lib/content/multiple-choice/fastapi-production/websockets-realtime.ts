import { MultipleChoiceQuestion } from '@/lib/types';

export const websocketsRealtimeMultipleChoice = [
  {
    id: 1,
    question:
      'What is the primary advantage of WebSockets over HTTP polling for real-time applications?',
    options: [
      'WebSockets maintain a persistent bi-directional connection, eliminating request overhead and enabling instant server-to-client push',
      'WebSockets are more secure than HTTP because they use encryption',
      'WebSockets can transfer larger amounts of data than HTTP',
      'WebSockets work on all browsers while HTTP polling does not',
    ],
    correctAnswer: 0,
    explanation:
      'WebSockets provide a persistent, full-duplex connection that stays open, allowing the server to push data to clients instantly without the client making repeated requests. HTTP polling requires the client to repeatedly send requests (every 1-5 seconds) asking "any updates?", which creates massive overhead: new TCP connection, HTTP headers (~500 bytes), server processing for each poll. With WebSockets: 1 connection for the entire session, < 10 bytes per message, server pushes immediately when data available, sub-millisecond latency. For 10,000 clients polling every 5 seconds: 2,000 requests/second with polling vs. 10,000 active connections with WebSockets. WebSockets use the same TLS encryption as HTTPS (option 2), have similar data transfer capabilities (option 3), and HTTP polling works on all browsers too (option 4). The key advantage is efficiency and true real-time push capability.',
  },
  {
    id: 2,
    question:
      'In a WebSocket application with multiple server instances behind a load balancer, what is required to broadcast messages to clients connected to different servers?',
    options: [
      'A message broker like Redis Pub/Sub that distributes messages to all server instances, which then broadcast to their local connections',
      'The load balancer must proxy all messages between servers',
      'WebSockets automatically synchronize across servers using the WebSocket protocol',
      'Each server must maintain a direct connection to all other servers for broadcasting',
    ],
    correctAnswer: 0,
    explanation:
      "When scaling WebSockets horizontally (multiple servers), you need a message broker to coordinate broadcasting. Scenario: Client A connects to Server 1, Client B connects to Server 2. When Client A sends a message that should reach Client B, Server 1 must somehow notify Server 2. Solution: Redis Pub/Sub acts as a central message bus. Server 1 publishes message to Redis channel, Redis distributes to all subscribed servers (1, 2, 3, N), each server broadcasts to its local connections only. Pattern: manager.publish(channel, message) → Redis → all servers → local websockets. The load balancer (option 2) only routes initial connections, it doesn't proxy WebSocket messages. WebSockets (option 3) don't have built-in cross-server synchronization—it's an application concern. Direct server-to-server connections (option 4) don't scale (N² connections) and add complexity. Redis Pub/Sub is the production standard, used by Socket.io, FastAPI scaling, and most real-time systems.",
  },
  {
    id: 3,
    question:
      'What is the purpose of heartbeat (ping/pong) messages in WebSocket connections?',
    options: [
      'To detect dead connections and prevent intermediaries like proxies from closing idle connections due to timeout',
      'To encrypt messages for security',
      'To compress data for faster transmission',
      'To authenticate users periodically',
    ],
    correctAnswer: 0,
    explanation:
      'Heartbeat messages keep connections alive and detect failures. Problems solved: 1) Network intermediaries (load balancers, proxies, NATs) often close idle connections after 30-60 seconds of inactivity. Without heartbeat, client appears connected but is actually disconnected. 2) Detecting dead connections: if client crashes or loses network, server won\'t know until it tries to send and fails. Heartbeat implementation: Server sends ping every 30 seconds, client responds with pong. If no pong after 3 attempts, connection is dead. Or vice versa: client pings server. This prevents: zombie connections (server thinks client connected but isn\'t), resource leaks (memory for dead connections), delayed error detection. Heartbeat is NOT for encryption (option 2—use TLS), compression (option 3—use permessage-deflate), or periodic auth (option 4—use token expiry). It\'s purely for connection health monitoring and keeping intermediaries happy. Production pattern: await websocket.send_json({"type": "ping"}) every 30 seconds.',
  },
  {
    id: 4,
    question:
      'What is backpressure in WebSocket applications and why is it important?',
    options: [
      'Backpressure handles situations where the server generates data faster than the client can consume it, preventing memory exhaustion by using bounded queues',
      'Backpressure is the delay caused by network latency between client and server',
      'Backpressure is authentication pressure applied to verify user identity',
      'Backpressure compresses messages to reduce bandwidth usage',
    ],
    correctAnswer: 0,
    explanation:
      "Backpressure is when a producer (server) generates data faster than a consumer (client) can handle it. Scenario: Server broadcasts 1000 messages/second, but client can only process 100/second due to slow network or CPU. Without backpressure handling: messages queue up in memory, RAM exhausts, server crashes. Solution: bounded queues. Implementation: queue = deque(maxlen=100); when full, drop oldest messages or stop accepting new data. The client receives what it can handle, and slow clients don't crash the server. Indicators: monitor queue length, if consistently full → client is slow. Options: 1) Drop old messages (acceptable for real-time tickers), 2) Slow down producer (pause sending), 3) Disconnect slow clients (with warning). This is NOT network latency (option 2—that's just delay), NOT authentication (option 3—unrelated), NOT compression (option 4—different technique). Backpressure is about flow control: preventing fast producers from overwhelming slow consumers. Critical for production: without it, one slow client can crash your entire system.",
  },
  {
    id: 5,
    question: 'How should WebSocket authentication be implemented in FastAPI?',
    options: [
      'Pass JWT token as a query parameter in the WebSocket URL and validate before calling websocket.accept()',
      'Send authentication credentials after the WebSocket connection is established',
      'WebSockets inherit authentication from the HTTP session automatically',
      'Use HTTP Basic Auth in the WebSocket handshake headers',
    ],
    correctAnswer: 0,
    explanation:
      "WebSocket authentication must happen BEFORE accepting the connection because once accepted, the client has an open channel. Best practice: pass JWT in query parameter: ws://example.com/ws/chat?token=<JWT>. Implementation: 1) Extract token from query params, 2) Validate JWT, 3) If valid, call websocket.accept(), 4) If invalid, call websocket.close(code=1008) and return. Why NOT option 2 (auth after connect): Once connection accepted, client can start sending data before auth completes—security risk. Why NOT option 3 (inherit from HTTP session): WebSockets upgrade from HTTP, but don't maintain session state automatically—you must explicitly pass credentials. Why NOT option 4 (Basic Auth): While possible, JWT is preferred in modern apps (stateless, contains user info, expires). The pattern: async def websocket_endpoint(websocket: WebSocket, token: str): user = await verify_jwt(token); if not user: await websocket.close(1008); return; await websocket.accept(); # Now authenticated. Alternative: pass token in Sec-WebSocket-Protocol header, but query param is simpler and more widely supported.",
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
