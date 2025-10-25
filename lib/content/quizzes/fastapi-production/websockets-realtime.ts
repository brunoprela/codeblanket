export const websocketsRealtimeQuiz = [
    {
      id: 1,
      question:
        "Design a real-time collaborative document editing system (like Google Docs) using FastAPI WebSockets. The system must handle: (1) multiple users editing simultaneously, (2) operational transformation to resolve conflicts, (3) presence indicators showing who's online, (4) cursor position tracking, and (5) chat functionality. Design the WebSocket message protocol, connection management strategy, and conflict resolution approach. How would you ensure users don't overwrite each other's changes? What happens when a user's connection drops and reconnects? Implement the core WebSocket handler with all these features.",
      answer: `**Collaborative Document Editing System Design**:

**1. Message Protocol**:

\`\`\`python
"""
WebSocket message protocol for collaborative editing
"""

from typing import Literal, Optional
from pydantic import BaseModel

# Message types
class EditMessage(BaseModel):
    type: Literal["edit"]
    doc_id: str
    user_id: str
    position: int  # Character position
    operation: Literal["insert", "delete"]
    content: str
    version: int  # Document version for conflict resolution

class CursorMessage(BaseModel):
    type: Literal["cursor"]
    doc_id: str
    user_id: str
    username: str
    position: int
    color: str  # User's cursor color

class PresenceMessage(BaseModel):
    type: Literal["presence"]
    doc_id: str
    user_id: str
    username: str
    status: Literal["online", "offline"]

class ChatMessage(BaseModel):
    type: Literal["chat"]
    doc_id: str
    user_id: str
    username: str
    message: str
    timestamp: str

class SyncRequest(BaseModel):
    type: Literal["sync_request"]
    doc_id: str
    last_known_version: int

class SyncResponse(BaseModel):
    type: Literal["sync_response"]
    doc_id: str
    content: str
    version: int
    active_users: list
\`\`\`

**2. Document State Management**:

\`\`\`python
"""
Document state with version control
"""

from dataclasses import dataclass
from typing import Dict, List
import asyncio

@dataclass
class DocumentState:
    """
    Server-side document state
    """
    doc_id: str
    content: str
    version: int
    active_users: Dict[str, dict]  # user_id -> {username, websocket, cursor_pos, color}
    edit_history: List[dict]  # For undo/redo and conflict resolution
    lock: asyncio.Lock  # Prevent race conditions

class DocumentManager:
    """
    Manage document states and connections
    """
    def __init__(self):
        self.documents: Dict[str, DocumentState] = {}
    
    def get_or_create_document(self, doc_id: str) -> DocumentState:
        """Get existing document or create new one"""
        if doc_id not in self.documents:
            self.documents[doc_id] = DocumentState(
                doc_id=doc_id,
                content="",
                version=0,
                active_users={},
                edit_history=[],
                lock=asyncio.Lock()
            )
        return self.documents[doc_id]
    
    async def apply_edit(
        self,
        doc_id: str,
        user_id: str,
        operation: str,
        position: int,
        content: str,
        client_version: int
    ) -> dict:
        """
        Apply edit with operational transformation
        Returns: {success, new_version, transformed_operation}
        """
        doc = self.get_or_create_document(doc_id)
        
        async with doc.lock:
            # Check version conflict
            if client_version < doc.version:
                # Client is behind, transform operation
                transformed_op = self._transform_operation(
                    doc,
                    operation,
                    position,
                    content,
                    client_version
                )
                operation = transformed_op["operation"]
                position = transformed_op["position"]
                content = transformed_op["content"]
            
            # Apply operation
            if operation == "insert":
                doc.content = (
                    doc.content[:position] +
                    content +
                    doc.content[position:]
                )
            elif operation == "delete":
                doc.content = (
                    doc.content[:position] +
                    doc.content[position + len(content):]
                )
            
            # Increment version
            doc.version += 1
            
            # Record in history
            doc.edit_history.append({
                "version": doc.version,
                "user_id": user_id,
                "operation": operation,
                "position": position,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "success": True,
                "new_version": doc.version,
                "operation": operation,
                "position": position,
                "content": content
            }
    
    def _transform_operation(
        self,
        doc: DocumentState,
        operation: str,
        position: int,
        content: str,
        client_version: int
    ) -> dict:
        """
        Operational Transformation (OT)
        
        Transform client operation based on missed server operations
        """
        # Get operations that happened after client's version
        missed_ops = [
            op for op in doc.edit_history
            if op["version"] > client_version
        ]
        
        # Transform position based on missed operations
        new_position = position
        
        for op in missed_ops:
            if op["operation"] == "insert":
                if op["position"] <= new_position:
                    # Insert happened before our position, shift right
                    new_position += len(op["content"])
            
            elif op["operation"] == "delete":
                if op["position"] < new_position:
                    # Delete happened before our position, shift left
                    new_position -= len(op["content"])
                    new_position = max(0, new_position)
        
        return {
            "operation": operation,
            "position": new_position,
            "content": content
        }

# Global document manager
doc_manager = DocumentManager()
\`\`\`

**3. WebSocket Handler**:

\`\`\`python
"""
Collaborative editing WebSocket handler
"""

from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/document/{doc_id}")
async def document_websocket(
    websocket: WebSocket,
    doc_id: str,
    token: str  # JWT token in query params
):
    """
    Collaborative document editing endpoint
    """
    # Authenticate
    try:
        user = await get_current_user_ws(websocket, token)
    except HTTPException:
        return
    
    # Accept connection
    await websocket.accept()
    
    # Get document
    doc = doc_manager.get_or_create_document(doc_id)
    
    # Add user to document
    user_color = generate_user_color(user.id)
    async with doc.lock:
        doc.active_users[user.id] = {
            "username": user.username,
            "websocket": websocket,
            "cursor_pos": 0,
            "color": user_color
        }
    
    # Send initial sync
    await websocket.send_json({
        "type": "sync_response",
        "doc_id": doc_id,
        "content": doc.content,
        "version": doc.version,
        "active_users": [
            {
                "user_id": uid,
                "username": info["username"],
                "cursor_pos": info["cursor_pos"],
                "color": info["color"]
            }
            for uid, info in doc.active_users.items()
        ]
    })
    
    # Broadcast presence
    await broadcast_to_document(doc_id, {
        "type": "presence",
        "doc_id": doc_id,
        "user_id": user.id,
        "username": user.username,
        "status": "online"
    }, exclude=websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "edit":
                # Apply edit
                result = await doc_manager.apply_edit(
                    doc_id=doc_id,
                    user_id=user.id,
                    operation=data["operation"],
                    position=data["position"],
                    content=data["content"],
                    client_version=data["version"]
                )
                
                # Broadcast edit to all users
                await broadcast_to_document(doc_id, {
                    "type": "edit",
                    "doc_id": doc_id,
                    "user_id": user.id,
                    "username": user.username,
                    "operation": result["operation"],
                    "position": result["position"],
                    "content": result["content"],
                    "version": result["new_version"]
                })
            
            elif message_type == "cursor":
                # Update cursor position
                async with doc.lock:
                    if user.id in doc.active_users:
                        doc.active_users[user.id]["cursor_pos"] = data["position"]
                
                # Broadcast cursor position
                await broadcast_to_document(doc_id, {
                    "type": "cursor",
                    "doc_id": doc_id,
                    "user_id": user.id,
                    "username": user.username,
                    "position": data["position"],
                    "color": user_color
                }, exclude=websocket)
            
            elif message_type == "chat":
                # Broadcast chat message
                await broadcast_to_document(doc_id, {
                    "type": "chat",
                    "doc_id": doc_id,
                    "user_id": user.id,
                    "username": user.username,
                    "message": data["message"],
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "sync_request":
                # Client reconnected, send current state
                await websocket.send_json({
                    "type": "sync_response",
                    "doc_id": doc_id,
                    "content": doc.content,
                    "version": doc.version,
                    "active_users": [
                        {
                            "user_id": uid,
                            "username": info["username"],
                            "cursor_pos": info["cursor_pos"],
                            "color": info["color"]
                        }
                        for uid, info in doc.active_users.items()
                    ]
                })
            
    except WebSocketDisconnect:
        # Remove user from document
        async with doc.lock:
            if user.id in doc.active_users:
                del doc.active_users[user.id]
        
        # Broadcast user left
        await broadcast_to_document(doc_id, {
            "type": "presence",
            "doc_id": doc_id,
            "user_id": user.id,
            "username": user.username,
            "status": "offline"
        })

async def broadcast_to_document(
    doc_id: str,
    message: dict,
    exclude: Optional[WebSocket] = None
):
    """
    Broadcast message to all users editing document
    """
    doc = doc_manager.documents.get(doc_id)
    if not doc:
        return
    
    async with doc.lock:
        for user_id, user_info in list(doc.active_users.items()):
            ws = user_info["websocket"]
            if ws != exclude:
                try:
                    await ws.send_json(message)
                except:
                    # Connection lost, remove user
                    del doc.active_users[user_id]
\`\`\`

**4. Client-Side Handling**:

\`\`\`javascript
// Client-side collaborative editing
class CollaborativeEditor {
    constructor(docId, token) {
        this.docId = docId;
        this.ws = new WebSocket(\`ws://localhost:8000/ws/document/\${docId}?token=\${token}\`);
        this.localVersion = 0;
        this.pendingOperations = [];
        this.otherCursors = new Map();
        
        this.setupWebSocket();
    }
    
    setupWebSocket() {
        this.ws.onopen = () => {
            console.log("Connected to document");
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log("Disconnected, attempting reconnect...");
            setTimeout(() => this.reconnect(), 1000);
        };
    }
    
    handleMessage(data) {
        switch(data.type) {
            case "sync_response":
                // Initial sync or reconnect
                this.editor.setText(data.content);
                this.localVersion = data.version;
                this.updateActiveUsers(data.active_users);
                break;
            
            case "edit":
                // Apply remote edit
                if (data.user_id !== this.userId) {
                    this.applyRemoteEdit(data);
                }
                this.localVersion = data.version;
                break;
            
            case "cursor":
                // Update other user's cursor
                this.updateCursor(data.user_id, data.position, data.color);
                break;
            
            case "presence":
                if (data.status === "online") {
                    this.addUser(data.user_id, data.username);
                } else {
                    this.removeUser(data.user_id);
                }
                break;
            
            case "chat":
                this.addChatMessage(data.username, data.message);
                break;
        }
    }
    
    onEdit(operation, position, content) {
        // Send edit to server
        this.ws.send(JSON.stringify({
            type: "edit",
            doc_id: this.docId,
            operation: operation,
            position: position,
            content: content,
            version: this.localVersion
        }));
        
        // Store pending operation
        this.pendingOperations.push({operation, position, content});
    }
    
    onCursorMove(position) {
        // Debounced cursor position broadcast
        clearTimeout(this.cursorTimeout);
        this.cursorTimeout = setTimeout(() => {
            this.ws.send(JSON.stringify({
                type: "cursor",
                position: position
            }));
        }, 100);
    }
    
    reconnect() {
        // Reconnect and request sync
        this.ws = new WebSocket(\`ws://localhost:8000/ws/document/\${this.docId}?token=\${this.token}\`);
        this.setupWebSocket();
        
        this.ws.onopen = () => {
            // Request full sync
            this.ws.send(JSON.stringify({
                type: "sync_request",
                doc_id: this.docId,
                last_known_version: this.localVersion
            }));
        };
    }
}
\`\`\`

**Key Design Decisions**:

1. **Operational Transformation**: Resolves conflicts when clients edit concurrently
2. **Version numbers**: Track document state, detect when client is behind
3. **Lock-based concurrency**: Prevent race conditions on document state
4. **Presence tracking**: Show who's online with active_users dictionary
5. **Cursor broadcasting**: Real-time cursor position updates (debounced)
6. **Reconnection handling**: Client requests sync with last known version
7. **Edit history**: Enables undo/redo and conflict resolution

**What happens on disconnect/reconnect**:
1. Client detects disconnect (onclose event)
2. Attempts reconnect with exponential backoff
3. On reconnect, sends sync_request with last_known_version
4. Server sends full document state
5. Client reconciles any pending operations`,
    },
    {
      id: 2,
      question:
        'Design a scalable WebSocket architecture for a live sports/stock ticker application serving 100,000+ concurrent connections. The system must broadcast real-time price updates to all connected clients with sub-second latency. How would you scale WebSocket connections across multiple server instances? How do you handle broadcasting to clients connected to different servers? Design the architecture including Redis Pub/Sub for message distribution, load balancing strategies, and connection recovery. What are the bottlenecks and how do you monitor performance at scale?',
      answer: `**Scalable WebSocket Architecture for Real-Time Ticker**:

**1. System Architecture**:

\`\`\`
┌─────────────┐
│   Clients   │ (100K+ WebSocket connections)
└──────┬──────┘
       │
┌──────▼──────┐
│ Load        │ (WebSocket-aware LB)
│ Balancer    │ (Sticky sessions)
└──────┬──────┘
       │
   ┌───┴───┬───────┬───────┐
   │       │       │       │
┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
│WS-1 │ │WS-2 │ │WS-3 │ │WS-N │  FastAPI servers
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   │       │       │       │
   └───┬───┴───┬───┴───┬───┘
       │       │       │
    ┌──▼───────▼───────▼──┐
    │   Redis Pub/Sub     │ (Message broker)
    └──────────┬──────────┘
               │
        ┌──────▼──────┐
        │  Price Feed │ (External API / Database)
        └─────────────┘
\`\`\`

**2. Redis Pub/Sub for Broadcasting**:

\`\`\`python
"""
Redis Pub/Sub for multi-server WebSocket broadcasting
"""

import redis.asyncio as redis
import json
from typing import Dict, Set

class DistributedConnectionManager:
    """
    Manage WebSocket connections across multiple servers
    """
    def __init__(self):
        # Local connections on THIS server
        self.local_connections: Dict[str, Set[WebSocket]] = {}
        
        # Redis for pub/sub
        self.redis = redis.from_url("redis://localhost:6379")
        self.pubsub = self.redis.pubsub()
        
        # Start listening to Redis
        asyncio.create_task(self._listen_redis())
    
    async def connect(self, websocket: WebSocket, channel: str):
        """
        Connect client to ticker channel
        """
        await websocket.accept()
        
        # Add to local connections
        if channel not in self.local_connections:
            self.local_connections[channel] = set()
        self.local_connections[channel].add(websocket)
        
        # Subscribe to Redis channel (if first connection)
        if len(self.local_connections[channel]) == 1:
            await self.pubsub.subscribe(channel)
        
        print(f"Connected to {channel}. Local: {len(self.local_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """
        Disconnect client
        """
        if channel in self.local_connections:
            self.local_connections[channel].discard(websocket)
            
            # Unsubscribe if no more local connections
            if not self.local_connections[channel]:
                asyncio.create_task(self.pubsub.unsubscribe(channel))
                del self.local_connections[channel]
    
    async def publish(self, channel: str, message: dict):
        """
        Publish message to Redis (broadcasts to ALL servers)
        """
        await self.redis.publish(
            channel,
            json.dumps(message)
        )
    
    async def _listen_redis(self):
        """
        Listen for messages from Redis and broadcast to local connections
        """
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"].decode()
                data = json.loads(message["data"])
                
                # Broadcast to local connections only
                await self._broadcast_local(channel, data)
    
    async def _broadcast_local(self, channel: str, message: dict):
        """
        Broadcast to connections on THIS server
        """
        if channel in self.local_connections:
            disconnected = []
            
            for websocket in self.local_connections[channel]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected
            for ws in disconnected:
                self.disconnect(ws, channel)

# Global manager
manager = DistributedConnectionManager()
\`\`\`

**3. WebSocket Endpoints**:

\`\`\`python
"""
FastAPI WebSocket endpoints for tickers
"""

@app.websocket("/ws/ticker/{symbol}")
async def ticker_websocket(
    websocket: WebSocket,
    symbol: str  # Stock symbol: AAPL, TSLA, etc.
):
    """
    Real-time ticker updates for specific symbol
    """
    channel = f"ticker:{symbol.upper()}"
    
    await manager.connect(websocket, channel)
    
    try:
        # Send initial price
        current_price = await get_current_price(symbol)
        await websocket.send_json({
            "type": "price",
            "symbol": symbol,
            "price": current_price,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive (client can send pings)
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)

@app.websocket("/ws/ticker/watchlist")
async def watchlist_websocket(
    websocket: WebSocket,
    symbols: str  # Comma-separated: AAPL,TSLA,GOOGL
):
    """
    Subscribe to multiple tickers
    """
    await websocket.accept()
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    channels = [f"ticker:{symbol}" for symbol in symbol_list]
    
    # Subscribe to all channels
    for channel in channels:
        await manager.connect(websocket, channel)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                # Add new symbol to watchlist
                new_symbol = data["symbol"].upper()
                new_channel = f"ticker:{new_symbol}"
                await manager.connect(websocket, new_channel)
            
            elif data.get("type") == "unsubscribe":
                # Remove symbol from watchlist
                symbol = data["symbol"].upper()
                channel = f"ticker:{symbol}"
                manager.disconnect(websocket, channel)
    
    except WebSocketDisconnect:
        # Disconnect from all channels
        for channel in channels:
            manager.disconnect(websocket, channel)
\`\`\`

**4. Price Feed Publisher**:

\`\`\`python
"""
Background task that publishes price updates
"""

import asyncio

class PriceFeedPublisher:
    """
    Fetch prices and publish to Redis
    """
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379")
        self.running = False
    
    async def start(self):
        """Start publishing price updates"""
        self.running = True
        asyncio.create_task(self._publish_loop())
    
    async def _publish_loop(self):
        """
        Continuously fetch and publish prices
        """
        while self.running:
            # Fetch latest prices (from external API or database)
            prices = await self._fetch_prices()
            
            # Publish each price update
            for symbol, data in prices.items():
                channel = f"ticker:{symbol}"
                
                await self.redis.publish(
                    channel,
                    json.dumps({
                        "type": "price",
                        "symbol": symbol,
                        "price": data["price"],
                        "change": data["change"],
                        "change_percent": data["change_percent"],
                        "volume": data["volume"],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                )
            
            # Wait before next update (e.g., 100ms for high-frequency)
            await asyncio.sleep(0.1)
    
    async def _fetch_prices(self) -> dict:
        """
        Fetch prices from external API or database
        
        In production: batch fetch from price feed API
        """
        # Example: fetch from cache or API
        symbols = ["AAPL", "TSLA", "GOOGL", "MSFT"]  # Hot symbols
        
        prices = {}
        for symbol in symbols:
            # Fetch from Redis cache or API
            price_data = await self.redis.get(f"price:{symbol}")
            if price_data:
                prices[symbol] = json.loads(price_data)
        
        return prices

# Start price feed on startup
price_feed = PriceFeedPublisher()

@app.on_event("startup")
async def startup_event():
    await price_feed.start()
\`\`\`

**5. Load Balancing Strategy**:

\`\`\`nginx
# Nginx load balancer configuration for WebSockets

upstream websocket_servers {
    # IP hash for sticky sessions (keep client on same server)
    ip_hash;
    
    server ws-server-1:8000;
    server ws-server-2:8000;
    server ws-server-3:8000;
    server ws-server-4:8000;
}

server {
    listen 80;
    
    location /ws/ {
        # WebSocket proxying
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_read_timeout 3600s;  # 1 hour
        proxy_send_timeout 3600s;
    }
}
\`\`\`

**6. Monitoring & Metrics**:

\`\`\`python
"""
Prometheus metrics for WebSocket performance
"""

from prometheus_client import Counter, Gauge, Histogram

# Connection metrics
WEBSOCKET_CONNECTIONS = Gauge(
    'websocket_connections_total',
    'Total active WebSocket connections',
    ['channel']
)

WEBSOCKET_CONNECTIONS_OPENED = Counter(
    'websocket_connections_opened_total',
    'Total WebSocket connections opened',
    ['channel']
)

WEBSOCKET_CONNECTIONS_CLOSED = Counter(
    'websocket_connections_closed_total',
    'Total WebSocket connections closed',
    ['channel', 'reason']
)

# Message metrics
WEBSOCKET_MESSAGES_SENT = Counter(
    'websocket_messages_sent_total',
    'Total messages sent to clients',
    ['channel']
)

WEBSOCKET_MESSAGE_SIZE = Histogram(
    'websocket_message_size_bytes',
    'Size of WebSocket messages',
    ['channel']
)

WEBSOCKET_LATENCY = Histogram(
    'websocket_broadcast_latency_seconds',
    'Time from publish to client delivery',
    ['channel'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Update metrics in connection manager
class MonitoredConnectionManager(DistributedConnectionManager):
    async def connect(self, websocket: WebSocket, channel: str):
        await super().connect(websocket, channel)
        
        WEBSOCKET_CONNECTIONS.labels(channel=channel).inc()
        WEBSOCKET_CONNECTIONS_OPENED.labels(channel=channel).inc()
    
    def disconnect(self, websocket: WebSocket, channel: str):
        super().disconnect(websocket, channel)
        
        WEBSOCKET_CONNECTIONS.labels(channel=channel).dec()
        WEBSOCKET_CONNECTIONS_CLOSED.labels(channel=channel, reason="normal").inc()
    
    async def _broadcast_local(self, channel: str, message: dict):
        start_time = time.time()
        
        await super()._broadcast_local(channel, message)
        
        # Track latency
        latency = time.time() - start_time
        WEBSOCKET_LATENCY.labels(channel=channel).observe(latency)
        
        # Track messages sent
        connection_count = len(self.local_connections.get(channel, []))
        WEBSOCKET_MESSAGES_SENT.labels(channel=channel).inc(connection_count)
\`\`\`

**7. Performance Bottlenecks & Solutions**:

| Bottleneck | Solution |
|------------|----------|
| CPU (JSON serialization) | Use msgpack or protobuf for binary protocols |
| Memory (100K connections) | Use gevent/asyncio (low memory per connection) |
| Redis pub/sub latency | Use Redis Cluster, multiple Redis instances |
| Network bandwidth | Compress messages, batch updates |
| Single server limit | Horizontal scaling with Redis pub/sub |
| Connection drops | Client reconnection with exponential backoff |

**Scaling Checklist**:

✅ Use Redis Pub/Sub for multi-server broadcasting  
✅ Sticky sessions (ip_hash) in load balancer  
✅ Monitor connections per server (max 10K-50K each)  
✅ Horizontal scaling: add more WebSocket servers  
✅ Client-side: reconnection logic with backoff  
✅ Compression: gzip for large payloads  
✅ Metrics: track connections, latency, throughput  
✅ Alerting: connection spikes, high latency, Redis issues`,
    },
    {
      id: 3,
      question:
        "You are building a real-time monitoring dashboard with WebSockets that displays live metrics from thousands of IoT devices. Each device sends sensor data every second. Design a system that: (1) handles device authentication, (2) aggregates metrics in real-time, (3) broadcasts only changed values to reduce bandwidth, (4) handles device disconnections gracefully, and (5) stores historical data. How would you implement backpressure when clients can't keep up with the data rate? What happens when a client reconnects after being offline? Implement the complete solution including device ingestion, aggregation, and client WebSocket handler.",
      answer: `**IoT Real-Time Monitoring Dashboard**:

**1. System Architecture**:

\`\`\`
┌─────────────┐                    ┌──────────────┐
│ IoT Devices │ ─── MQTT/HTTP ───> │   Ingestion  │
│  (1000s)    │                    │   Service    │
└─────────────┘                    └──────┬───────┘
                                          │
                                    ┌─────▼──────┐
                                    │  Real-Time  │
                                    │ Aggregation │
                                    │  (Stream)   │
                                    └─────┬───────┘
                                          │
                         ┌────────────────┼────────────────┐
                         │                │                │
                    ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
                    │ TimeSeries│    │  Redis  │     │WebSocket│
                    │    DB     │    │ Pub/Sub │     │ Clients │
                    │(Historical)│   │(Live    │     │(Dashboard)│
                    └──────────┘    │ Data)   │     └─────────┘
                                    └─────────┘
\`\`\`

**2. Device Authentication & Ingestion**:

\`\`\`python
"""
IoT device authentication and data ingestion
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List

router = APIRouter()

class SensorReading(BaseModel):
    device_id: str
    sensor_type: str  # temperature, humidity, pressure, etc.
    value: float
    unit: str
    timestamp: str

class DeviceCredentials(BaseModel):
    device_id: str
    api_key: str

async def authenticate_device(
    device_id: str,
    api_key: str = Header(None)
) -> dict:
    """
    Authenticate IoT device
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Validate device and API key
    device = await get_device(device_id)
    
    if not device or device.api_key != api_key:
        raise HTTPException(status_code=403, detail="Invalid device credentials")
    
    return device

@router.post("/devices/{device_id}/readings")
async def ingest_readings(
    device_id: str,
    readings: List[SensorReading],
    device: dict = Depends(authenticate_device)
):
    """
    Ingest sensor readings from IoT device
    """
    # Process readings asynchronously
    for reading in readings:
        await process_reading(device_id, reading)
    
    return {"status": "accepted", "count": len(readings)}

async def process_reading(device_id: str, reading: SensorReading):
    """
    Process sensor reading: aggregate, store, broadcast
    """
    # 1. Store in time-series database (InfluxDB, TimescaleDB)
    await store_time_series(device_id, reading)
    
    # 2. Update real-time aggregation
    await update_aggregation(device_id, reading)
    
    # 3. Check if value changed significantly (delta compression)
    if await has_significant_change(device_id, reading):
        # Broadcast only if changed
        await broadcast_reading(device_id, reading)
\`\`\`

**3. Real-Time Aggregation**:

\`\`\`python
"""
Real-time metric aggregation
"""

import redis.asyncio as redis
from datetime import datetime, timedelta

class MetricAggregator:
    """
    Aggregate metrics in real-time with windowing
    """
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379")
    
    async def update(self, device_id: str, reading: SensorReading):
        """
        Update real-time aggregations
        """
        key_prefix = f"metrics:{device_id}:{reading.sensor_type}"
        
        # Current value
        await self.redis.set(
            f"{key_prefix}:current",
            reading.value,
            ex=300  # Expire after 5 minutes
        )
        
        # 1-minute average (rolling window)
        await self._update_window(
            f"{key_prefix}:1m",
            reading.value,
            window_seconds=60
        )
        
        # 5-minute average
        await self._update_window(
            f"{key_prefix}:5m",
            reading.value,
            window_seconds=300
        )
        
        # Min/Max tracking
        await self._update_minmax(key_prefix, reading.value)
    
    async def _update_window(
        self,
        key: str,
        value: float,
        window_seconds: int
    ):
        """
        Update rolling window average
        """
        now = datetime.utcnow().timestamp()
        
        # Add value with timestamp as score
        await self.redis.zadd(key, {str(value): now})
        
        # Remove values outside window
        cutoff = now - window_seconds
        await self.redis.zremrangebyscore(key, '-inf', cutoff)
        
        # Set expiry
        await self.redis.expire(key, window_seconds * 2)
    
    async def _update_minmax(self, key_prefix: str, value: float):
        """
        Track min and max values
        """
        # Update min
        current_min = await self.redis.get(f"{key_prefix}:min")
        if not current_min or value < float(current_min):
            await self.redis.set(f"{key_prefix}:min", value, ex=3600)
        
        # Update max
        current_max = await self.redis.get(f"{key_prefix}:max")
        if not current_max or value > float(current_max):
            await self.redis.set(f"{key_prefix}:max", value, ex=3600)
    
    async def get_aggregated_metrics(self, device_id: str) -> dict:
        """
        Get aggregated metrics for device
        """
        metrics = {}
        
        # Get all sensor types for device
        pattern = f"metrics:{device_id}:*:current"
        keys = await self.redis.keys(pattern)
        
        for key in keys:
            # Extract sensor type
            parts = key.decode().split(':')
            sensor_type = parts[2]
            
            # Get current value
            current = await self.redis.get(f"metrics:{device_id}:{sensor_type}:current")
            
            # Get averages
            avg_1m = await self._get_window_average(
                f"metrics:{device_id}:{sensor_type}:1m"
            )
            avg_5m = await self._get_window_average(
                f"metrics:{device_id}:{sensor_type}:5m"
            )
            
            # Get min/max
            min_val = await self.redis.get(f"metrics:{device_id}:{sensor_type}:min")
            max_val = await self.redis.get(f"metrics:{device_id}:{sensor_type}:max")
            
            metrics[sensor_type] = {
                "current": float(current) if current else None,
                "avg_1m": avg_1m,
                "avg_5m": avg_5m,
                "min": float(min_val) if min_val else None,
                "max": float(max_val) if max_val else None
            }
        
        return metrics
    
    async def _get_window_average(self, key: str) -> float:
        """
        Calculate average from window
        """
        values = await self.redis.zrange(key, 0, -1)
        
        if not values:
            return None
        
        values = [float(v) for v in values]
        return sum(values) / len(values)

aggregator = MetricAggregator()
\`\`\`

**4. Delta Compression (Only Broadcast Changes)**:

\`\`\`python
"""
Broadcast only significant changes to reduce bandwidth
"""

class DeltaCompression:
    """
    Only broadcast values that changed significantly
    """
    def __init__(self, threshold_percent: float = 1.0):
        self.threshold_percent = threshold_percent
        self.last_values = {}  # Cache last broadcast values
    
    async def has_significant_change(
        self,
        device_id: str,
        sensor_type: str,
        new_value: float
    ) -> bool:
        """
        Check if value changed significantly
        """
        key = f"{device_id}:{sensor_type}"
        last_value = self.last_values.get(key)
        
        if last_value is None:
            # First value, always broadcast
            self.last_values[key] = new_value
            return True
        
        # Calculate percent change
        if last_value == 0:
            change_percent = 100 if new_value != 0 else 0
        else:
            change_percent = abs((new_value - last_value) / last_value * 100)
        
        if change_percent >= self.threshold_percent:
            # Significant change, update cache and broadcast
            self.last_values[key] = new_value
            return True
        
        return False

delta_compression = DeltaCompression(threshold_percent=1.0)  # 1% change threshold
\`\`\`

**5. WebSocket with Backpressure**:

\`\`\`python
"""
WebSocket client handler with backpressure
"""

from collections import deque
import asyncio

class DashboardConnectionManager:
    """
    Manage dashboard WebSocket connections with backpressure
    """
    def __init__(self, max_queue_size: int = 100):
        self.connections: Dict[str, dict] = {}
        self.max_queue_size = max_queue_size
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        subscriptions: List[str]
    ):
        """
        Connect dashboard client
        """
        await websocket.accept()
        
        self.connections[client_id] = {
            "websocket": websocket,
            "subscriptions": set(subscriptions),
            "queue": deque(maxlen=self.max_queue_size),  # Bounded queue
            "slow": False
        }
        
        # Start send loop
        asyncio.create_task(self._send_loop(client_id))
    
    def disconnect(self, client_id: str):
        """
        Disconnect client
        """
        if client_id in self.connections:
            del self.connections[client_id]
    
    async def broadcast_reading(self, device_id: str, reading: dict):
        """
        Broadcast reading to subscribed clients
        """
        for client_id, conn in self.connections.items():
            # Check if client subscribed to this device
            if device_id in conn["subscriptions"]:
                # Add to queue (bounded, drops old if full)
                conn["queue"].append(reading)
                
                # Mark as slow if queue is full
                if len(conn["queue"]) >= self.max_queue_size:
                    conn["slow"] = True
    
    async def _send_loop(self, client_id: str):
        """
        Continuously send queued messages to client
        Implements backpressure
        """
        conn = self.connections.get(client_id)
        if not conn:
            return
        
        websocket = conn["websocket"]
        queue = conn["queue"]
        
        try:
            while client_id in self.connections:
                if queue:
                    # Batch send (reduce WebSocket overhead)
                    batch = []
                    while queue and len(batch) < 10:  # Max 10 messages per batch
                        batch.append(queue.popleft())
                    
                    await websocket.send_json({
                        "type": "batch",
                        "readings": batch,
                        "slow_client_warning": conn["slow"]
                    })
                    
                    # Reset slow flag if queue cleared
                    if not queue:
                        conn["slow"] = False
                else:
                    # No messages, wait
                    await asyncio.sleep(0.1)
        
        except:
            self.disconnect(client_id)

dashboard_manager = DashboardConnectionManager()

@app.websocket("/ws/dashboard")
async def dashboard_websocket(
    websocket: WebSocket,
    token: str,
    devices: str = None  # Comma-separated device IDs
):
    """
    Dashboard WebSocket with backpressure handling
    """
    # Authenticate
    user = await get_current_user_ws(websocket, token)
    
    # Parse subscriptions
    subscriptions = devices.split(",") if devices else []
    
    await dashboard_manager.connect(websocket, user.id, subscriptions)
    
    # Send initial aggregated data
    for device_id in subscriptions:
        metrics = await aggregator.get_aggregated_metrics(device_id)
        await websocket.send_json({
            "type": "initial",
            "device_id": device_id,
            "metrics": metrics
        })
    
    try:
        while True:
            # Handle client commands
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe":
                # Add device to subscription
                dashboard_manager.connections[user.id]["subscriptions"].add(
                    data["device_id"]
                )
            
            elif data["type"] == "unsubscribe":
                # Remove device from subscription
                dashboard_manager.connections[user.id]["subscriptions"].discard(
                    data["device_id"]
                )
            
            elif data["type"] == "get_history":
                # Fetch historical data
                history = await get_device_history(
                    data["device_id"],
                    start=data.get("start"),
                    end=data.get("end")
                )
                await websocket.send_json({
                    "type": "history",
                    "device_id": data["device_id"],
                    "data": history
                })
    
    except WebSocketDisconnect:
        dashboard_manager.disconnect(user.id)
\`\`\`

**Key Features**:

1. **Device Authentication**: API keys for IoT devices
2. **Real-time Aggregation**: Rolling windows (1m, 5m averages), min/max
3. **Delta Compression**: Only broadcast changes > 1%
4. **Backpressure**: Bounded queues, batch sending, slow client warnings
5. **Reconnection**: Send aggregated state + historical data on reconnect
6. **Scalability**: Redis for coordination, time-series DB for history

**Reconnection Handling**:
- Client sends last timestamp received
- Server sends aggregated state + missing historical data
- Resume real-time updates`,
    },

].map(({ id, ...q }, idx) => ({ id: `fastapi-websockets-realtime-q-${idx + 1}`, question: q.question, sampleAnswer: String(q.answer), keyPoints: [] }));
