export const websocketsRealtime = {
  title: 'WebSockets & Real-Time Communication',
  id: 'websockets-realtime',
  content: `
# WebSockets & Real-Time Communication

## Introduction

HTTP is request-response: client asks, server answers, connection closes. But modern apps need real-time updates: chat messages, live notifications, collaborative editing, stock tickers, gaming. WebSockets provide bi-directional, persistent connections between client and server.

**Why WebSockets matter:**
- **Real-time updates**: Server pushes data instantly to clients
- **Bi-directional**: Both client and server can initiate communication
- **Efficient**: Single connection vs polling (many HTTP requests)
- **Low latency**: No request overhead, instant message delivery

**Use cases:**
- Chat applications
- Live notifications
- Collaborative tools (Google Docs-like)
- Real-time dashboards
- Live sports/stock updates
- Multiplayer games
- IoT device communication

In this section, you'll master:
- WebSocket connections in FastAPI
- Broadcasting to multiple clients
- Authentication and authorization
- Connection management
- Production patterns
- Testing WebSockets

---

## WebSocket Basics

### Simple WebSocket Endpoint

\`\`\`python
"""
Basic WebSocket endpoint
"""

from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Basic WebSocket echo server
    """
    # Accept connection
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Send message back to client
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        # Clean up (optional, happens automatically)
        pass

# Client-side JavaScript
"""
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
    console.log("Connected");
    ws.send("Hello Server!");
};

ws.onmessage = (event) => {
    console.log("Received:", event.data);
};

ws.onclose = () => {
    console.log("Disconnected");
};
"""
\`\`\`

### WebSocket with JSON

\`\`\`python
"""
WebSocket with JSON messages
"""

import json

@app.websocket("/ws/chat/{room_id}")
async def chat_room(websocket: WebSocket, room_id: str):
    """
    Chat room WebSocket
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive JSON
            data = await websocket.receive_json()
            
            # Process message
            message_type = data.get("type")
            
            if message_type == "message":
                # Send JSON response
                await websocket.send_json({
                    "type": "message",
                    "room": room_id,
                    "content": data.get("content"),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"Client disconnected from room {room_id}")
\`\`\`

---

## Connection Management

### Connection Manager

\`\`\`python
"""
Manage multiple WebSocket connections
"""

from typing import Dict, List
from fastapi import WebSocketDisconnect

class ConnectionManager:
    """
    Manage WebSocket connections for broadcasting
    """
    def __init__(self):
        # Store active connections by room
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room_id: str):
        """
        Accept and store connection
        """
        await websocket.accept()
        
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        
        self.active_connections[room_id].append(websocket)
        
        print(f"Client connected to room {room_id}. Total: {len(self.active_connections[room_id])}")
    
    def disconnect(self, websocket: WebSocket, room_id: str):
        """
        Remove connection
        """
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            
            # Clean up empty rooms
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]
        
        print(f"Client disconnected from room {room_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific client
        """
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict, room_id: str):
        """
        Send message to all clients in room
        """
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_json(message)
    
    async def broadcast_except(self, message: dict, room_id: str, exclude: WebSocket):
        """
        Broadcast to all except one client
        """
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                if connection != exclude:
                    await connection.send_json(message)

# Global connection manager
manager = ConnectionManager()

@app.websocket("/ws/chat/{room_id}")
async def chat_room(websocket: WebSocket, room_id: str):
    """
    Chat room with broadcasting
    """
    await manager.connect(websocket, room_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Broadcast to all in room
            await manager.broadcast({
                "type": "message",
                "content": data["content"],
                "timestamp": datetime.utcnow().isoformat()
            }, room_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        
        # Notify others
        await manager.broadcast({
            "type": "user_left",
            "timestamp": datetime.utcnow().isoformat()
        }, room_id)
\`\`\`

---

## Authentication & Authorization

### Token-Based WebSocket Auth

\`\`\`python
"""
WebSocket authentication with JWT
"""

from fastapi import WebSocket, HTTPException, status
from jose import JWTError, jwt

async def get_current_user_ws(websocket: WebSocket, token: str) -> User:
    """
    Authenticate WebSocket connection with JWT
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=403, detail="Invalid token")
        
        user = get_user(username)
        if user is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=403, detail="User not found")
        
        return user
        
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail="Invalid token")

@app.websocket("/ws/chat/{room_id}")
async def authenticated_chat(
    websocket: WebSocket,
    room_id: str,
    token: str  # Query parameter: ws://localhost/ws/chat/123?token=xxx
):
    """
    Authenticated chat room
    """
    # Authenticate before accepting connection
    try:
        user = await get_current_user_ws(websocket, token)
    except HTTPException:
        return  # Connection already closed
    
    # Now accept connection
    await manager.connect(websocket, room_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Broadcast with username
            await manager.broadcast({
                "type": "message",
                "user": user.username,
                "content": data["content"],
                "timestamp": datetime.utcnow().isoformat()
            }, room_id)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
\`\`\`

---

## Production Patterns

### Heartbeat/Ping-Pong

\`\`\`python
"""
Keep connections alive with heartbeat
"""

import asyncio

class ConnectionManagerWithHeartbeat(ConnectionManager):
    """
    Connection manager with automatic heartbeat
    """
    async def connect(self, websocket: WebSocket, room_id: str):
        """Connect and start heartbeat"""
        await super().connect(websocket, room_id)
        
        # Start heartbeat task
        asyncio.create_task(self._heartbeat(websocket, room_id))
    
    async def _heartbeat(self, websocket: WebSocket, room_id: str):
        """
        Send periodic ping to keep connection alive
        """
        try:
            while True:
                await asyncio.sleep(30)  # Every 30 seconds
                
                await websocket.send_json({"type": "ping"})
                
        except:
            # Connection lost, clean up
            self.disconnect(websocket, room_id)
\`\`\`

### Graceful Shutdown

\`\`\`python
"""
Handle server shutdown gracefully
"""

import signal

@app.on_event("shutdown")
async def shutdown_event():
    """
    Close all WebSocket connections on shutdown
    """
    for room_id, connections in manager.active_connections.items():
        for websocket in connections:
            await websocket.close(code=1001, reason="Server shutting down")
\`\`\`

---

## Summary

✅ **WebSocket basics**: Bi-directional, persistent connections  
✅ **Connection management**: ConnectionManager for broadcasting  
✅ **Authentication**: JWT token-based WebSocket auth  
✅ **Broadcasting**: Send messages to all connected clients  
✅ **Production patterns**: Heartbeat, graceful shutdown  

### Next Steps

In the next section, we'll explore **File Uploads & Streaming Responses**: handling file uploads efficiently with validation, and implementing streaming for large files and real-time data.
`,
};
