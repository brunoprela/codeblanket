export const asyncHttpAiohttp = {
  title: 'Async HTTP with aiohttp',
  id: 'async-http-aiohttp',
  content: `
# Async HTTP with aiohttp

## Introduction

**aiohttp** is the most popular async HTTP library for Python, providing both client and server capabilities. It\'s essential for building high-performance APIs, web scrapers, and microservices.

### Why aiohttp Over requests

\`\`\`python
"""
requests (Blocking) vs aiohttp (Async)
"""

import time
import asyncio
import requests  # Blocking
import aiohttp   # Async

# Blocking: Sequential execution
def fetch_sync (urls):
    start = time.time()
    results = []
    for url in urls:
        response = requests.get (url)
        results.append (response.json())
    elapsed = time.time() - start
    print(f"Sync: {elapsed:.2f}s")
    return results

# Async: Concurrent execution
async def fetch_async (urls):
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append (session.get (url))
        responses = await asyncio.gather(*tasks)
        results = [await r.json() for r in responses]
    elapsed = time.time() - start
    print(f"Async: {elapsed:.2f}s")
    return results

urls = ['https://httpbin.org/delay/1'] * 10

# fetch_sync (urls)    # ~10 seconds (sequential)
# asyncio.run (fetch_async (urls))  # ~1 second (concurrent!)

# 10× speedup with aiohttp!
\`\`\`

By the end of this section, you'll master:
- ClientSession for HTTP requests
- GET, POST, and other HTTP methods
- Headers, cookies, and authentication
- Timeouts and error handling
- Connection pooling and performance
- Streaming responses
- WebSocket clients
- Building production-ready HTTP clients

---

## ClientSession: The Foundation

ClientSession manages connection pooling and should be reused across requests.

### Basic Session Usage

\`\`\`python
"""
ClientSession: Proper Usage Pattern
"""

import asyncio
import aiohttp

async def fetch_example():
    # ❌ Bad: Creates new session for each request (slow)
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/users/1') as response:
            user1 = await response.json()
    
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/users/2') as response:
            user2 = await response.json()
    
    # ✅ Good: Reuse session (connection pooling)
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/users/1') as response:
            user1 = await response.json()
        
        async with session.get('https://api.example.com/users/2') as response:
            user2 = await response.json()
    
    # Even better: Concurrent requests with same session
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.get('https://api.example.com/users/1'),
            session.get('https://api.example.com/users/2'),
        ]
        responses = await asyncio.gather(*tasks)
        users = [await r.json() for r in responses]

asyncio.run (fetch_example())

# Key insight: One session, many concurrent requests
# Session manages connection pool automatically
\`\`\`

### Session Configuration

\`\`\`python
"""
Configuring ClientSession for Production
"""

import aiohttp
import asyncio
from aiohttp import TCPConnector, ClientTimeout

async def production_session_example():
    # Configure connection pooling
    connector = TCPConnector(
        limit=100,              # Max total connections
        limit_per_host=30,      # Max connections per host
        ttl_dns_cache=300,      # DNS cache TTL (seconds)
        use_dns_cache=True,
    )
    
    # Configure timeouts
    timeout = ClientTimeout(
        total=30,       # Total timeout for request
        connect=10,     # Connection timeout
        sock_read=10,   # Socket read timeout
    )
    
    # Create session with configuration
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            'User-Agent': 'MyApp/1.0',
            'Accept': 'application/json',
        }
    ) as session:
        # Make requests
        async with session.get('https://api.example.com/data') as response:
            data = await response.json()
            return data

# Session configuration is critical for production performance!
\`\`\`

---

## Making HTTP Requests

### GET Requests

\`\`\`python
"""
GET Requests with aiohttp
"""

import aiohttp
import asyncio

async def get_examples():
    async with aiohttp.ClientSession() as session:
        # Simple GET
        async with session.get('https://httpbin.org/get') as response:
            print(f"Status: {response.status}")
            data = await response.json()
            print(f"Data: {data}")
        
        # GET with query parameters
        params = {'user_id': 123, 'limit': 10, 'offset': 0}
        async with session.get(
            'https://api.example.com/users',
            params=params
        ) as response:
            users = await response.json()
        
        # GET with headers
        headers = {'Authorization': 'Bearer token123'}
        async with session.get(
            'https://api.example.com/protected',
            headers=headers
        ) as response:
            data = await response.json()
        
        # GET with timeout
        timeout = aiohttp.ClientTimeout (total=5)
        try:
            async with session.get(
                'https://slow-api.example.com',
                timeout=timeout
            ) as response:
                data = await response.json()
        except asyncio.TimeoutError:
            print("Request timed out!")

asyncio.run (get_examples())
\`\`\`

### POST Requests

\`\`\`python
"""
POST Requests: JSON, Forms, Files
"""

import aiohttp
import asyncio

async def post_examples():
    async with aiohttp.ClientSession() as session:
        # POST JSON data
        user_data = {'name': 'Alice', 'email': 'alice@example.com'}
        async with session.post(
            'https://api.example.com/users',
            json=user_data  # Automatically sets Content-Type: application/json
        ) as response:
            created_user = await response.json()
            print(f"Created user: {created_user}")
        
        # POST form data
        form_data = {'username': 'alice', 'password': 'secret'}
        async with session.post(
            'https://api.example.com/login',
            data=form_data  # Content-Type: application/x-www-form-urlencoded
        ) as response:
            result = await response.json()
        
        # POST multipart (file upload)
        data = aiohttp.FormData()
        data.add_field('file',
                      open('document.pdf', 'rb'),
                      filename='document.pdf',
                      content_type='application/pdf')
        data.add_field('description', 'My document')
        
        async with session.post(
            'https://api.example.com/upload',
            data=data
        ) as response:
            result = await response.json()

asyncio.run (post_examples())
\`\`\`

### Other HTTP Methods

\`\`\`python
"""
PUT, PATCH, DELETE Requests
"""

import aiohttp
import asyncio

async def other_methods():
    async with aiohttp.ClientSession() as session:
        # PUT (full update)
        user_update = {'name': 'Alice Smith', 'email': 'alice.smith@example.com'}
        async with session.put(
            'https://api.example.com/users/123',
            json=user_update
        ) as response:
            updated_user = await response.json()
        
        # PATCH (partial update)
        partial_update = {'email': 'newemail@example.com'}
        async with session.patch(
            'https://api.example.com/users/123',
            json=partial_update
        ) as response:
            updated_user = await response.json()
        
        # DELETE
        async with session.delete('https://api.example.com/users/123') as response:
            print(f"Deleted: {response.status == 204}")
        
        # HEAD (get headers only, no body)
        async with session.head('https://api.example.com/large-file') as response:
            content_length = response.headers.get('Content-Length')
            print(f"File size: {content_length} bytes")

asyncio.run (other_methods())
\`\`\`

---

## Response Handling

### Reading Response Data

\`\`\`python
"""
Different Ways to Read Response Data
"""

import aiohttp
import asyncio

async def response_handling():
    async with aiohttp.ClientSession() as session:
        # Read as JSON
        async with session.get('https://httpbin.org/json') as response:
            data = await response.json()
            print(f"JSON: {type (data)}")
        
        # Read as text
        async with session.get('https://httpbin.org/html') as response:
            text = await response.text()
            print(f"Text: {len (text)} characters")
        
        # Read as bytes
        async with session.get('https://httpbin.org/bytes/100') as response:
            content = await response.read()
            print(f"Bytes: {len (content)} bytes")
        
        # Stream large response
        async with session.get('https://httpbin.org/stream/100') as response:
            # Read line by line (memory efficient)
            async for line in response.content:
                data = line.decode('utf-8')
                print(f"Line: {data[:50]}...")
        
        # Check status
        async with session.get('https://httpbin.org/status/404') as response:
            print(f"Status: {response.status}")
            print(f"OK: {response.ok}")  # True if 200 <= status < 400
            
            # Raise exception for error statuses
            response.raise_for_status()  # Raises ClientResponseError

asyncio.run (response_handling())
\`\`\`

### Headers and Cookies

\`\`\`python
"""
Working with Headers and Cookies
"""

import aiohttp
import asyncio

async def headers_cookies():
    async with aiohttp.ClientSession() as session:
        # Request headers
        headers = {
            'Authorization': 'Bearer token123',
            'Accept': 'application/json',
            'User-Agent': 'MyApp/1.0'
        }
        async with session.get(
            'https://api.example.com/data',
            headers=headers
        ) as response:
            data = await response.json()
        
        # Response headers
        async with session.get('https://httpbin.org/response-headers') as response:
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            print(f"All headers: {dict (response.headers)}")
        
        # Cookies (automatic handling)
        async with session.get('https://httpbin.org/cookies/set?name=value') as response:
            pass  # Cookie set automatically
        
        async with session.get('https://httpbin.org/cookies') as response:
            # Cookies sent automatically with subsequent requests
            cookies_data = await response.json()
            print(f"Cookies: {cookies_data}")
        
        # Manual cookie setting
        cookies = {'session_id': 'abc123', 'user_id': '456'}
        async with session.get(
            'https://api.example.com/profile',
            cookies=cookies
        ) as response:
            profile = await response.json()

asyncio.run (headers_cookies())
\`\`\`

---

## Error Handling

### Common Exceptions

\`\`\`python
"""
Handling aiohttp Exceptions
"""

import aiohttp
import asyncio

async def error_handling():
    async with aiohttp.ClientSession() as session:
        # Timeout
        try:
            timeout = aiohttp.ClientTimeout (total=1)
            async with session.get(
                'https://httpbin.org/delay/5',
                timeout=timeout
            ) as response:
                data = await response.json()
        except asyncio.TimeoutError:
            print("Request timed out!")
        
        # Connection error
        try:
            async with session.get('https://nonexistent-domain-12345.com') as response:
                data = await response.json()
        except aiohttp.ClientConnectorError as e:
            print(f"Connection error: {e}")
        
        # HTTP error status
        try:
            async with session.get('https://httpbin.org/status/500') as response:
                response.raise_for_status()  # Raises for 4xx/5xx
        except aiohttp.ClientResponseError as e:
            print(f"HTTP error: {e.status} - {e.message}")
        
        # SSL error
        try:
            async with session.get('https://expired.badssl.com/') as response:
                data = await response.text()
        except aiohttp.ClientSSLError as e:
            print(f"SSL error: {e}")
        
        # General client error (catches all above)
        try:
            async with session.get('https://api.example.com/data') as response:
                data = await response.json()
        except aiohttp.ClientError as e:
            print(f"Client error: {e}")

asyncio.run (error_handling())
\`\`\`

### Retry Logic with Exponential Backoff

\`\`\`python
"""
Production-Ready Retry Logic
"""

import aiohttp
import asyncio

async def fetch_with_retry (session, url, retries=3):
    """Fetch with exponential backoff retry"""
    for attempt in range (retries):
        try:
            async with session.get (url, timeout=aiohttp.ClientTimeout (total=10)) as response:
                response.raise_for_status()
                return await response.json()
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == retries - 1:
                # Last attempt, give up
                print(f"Failed after {retries} attempts: {e}")
                raise
            
            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            await asyncio.sleep (wait_time)

async def main():
    async with aiohttp.ClientSession() as session:
        try:
            data = await fetch_with_retry (session, 'https://api.example.com/data')
            print(f"Success: {data}")
        except Exception as e:
            print(f"All retries exhausted: {e}")

asyncio.run (main())
\`\`\`

---

## Streaming Responses

### Streaming Large Files

\`\`\`python
"""
Efficiently Download Large Files
"""

import aiohttp
import asyncio

async def download_file (url, destination):
    """Download file in chunks (memory efficient)"""
    async with aiohttp.ClientSession() as session:
        async with session.get (url) as response:
            # Get file size
            total_size = int (response.headers.get('Content-Length', 0))
            print(f"Downloading {total_size:,} bytes...")
            
            downloaded = 0
            with open (destination, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write (chunk)
                    downloaded += len (chunk)
                    
                    # Progress indicator
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\\r')
            
            print(f"\\nDownloaded to {destination}")

# asyncio.run (download_file('https://example.com/large-file.zip', 'file.zip'))

# Memory usage: Only 8KB chunk in memory at a time
# vs loading entire file (could be GB!)
\`\`\`

### Server-Sent Events (SSE)

\`\`\`python
"""
Consuming Server-Sent Events
"""

import aiohttp
import asyncio

async def consume_sse (url):
    """Consume real-time server-sent events"""
    async with aiohttp.ClientSession() as session:
        async with session.get (url) as response:
            # Read events as they arrive
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data:'):
                    # Extract event data
                    event_data = line[5:].strip()
                    print(f"Event: {event_data}")
                    
                    # Process event
                    await handle_event (event_data)

async def handle_event (data):
    """Handle individual event"""
    # Parse and process
    print(f"Processing: {data}")

# Use case: Real-time dashboards, live updates, notifications
# asyncio.run (consume_sse('https://api.example.com/events'))
\`\`\`

---

## WebSocket Client

### Basic WebSocket

\`\`\`python
"""
WebSocket Client with aiohttp
"""

import aiohttp
import asyncio

async def websocket_example():
    """Connect to WebSocket server"""
    session = aiohttp.ClientSession()
    
    async with session.ws_connect('wss://echo.websocket.org') as ws:
        # Send message
        await ws.send_str('Hello, WebSocket!')
        
        # Receive message
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                print(f"Received: {msg.data}")
                
                # Send another message
                await ws.send_str (f"Echo: {msg.data}")
                
                # Close after one exchange
                await ws.close()
                break
            
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")
                break
    
    await session.close()

asyncio.run (websocket_example())
\`\`\`

### Production WebSocket Client

\`\`\`python
"""
Production WebSocket with Reconnection
"""

import aiohttp
import asyncio

class WebSocketClient:
    """Robust WebSocket client with reconnection"""
    
    def __init__(self, url):
        self.url = url
        self.session = None
        self.ws = None
        self.should_reconnect = True
    
    async def connect (self):
        """Connect with automatic reconnection"""
        while self.should_reconnect:
            try:
                self.session = aiohttp.ClientSession()
                self.ws = await self.session.ws_connect (self.url)
                print("WebSocket connected")
                
                # Process messages
                await self.receive_messages()
            
            except Exception as e:
                print(f"Connection error: {e}")
                
                if self.session:
                    await self.session.close()
                
                if self.should_reconnect:
                    print("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    break
    
    async def receive_messages (self):
        """Receive and process messages"""
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self.handle_message (msg.data)
            
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                print("WebSocket closed")
                break
            
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {self.ws.exception()}")
                break
    
    async def handle_message (self, data):
        """Handle incoming message"""
        print(f"Received: {data}")
        # Process message...
    
    async def send (self, data):
        """Send message"""
        if self.ws and not self.ws.closed:
            await self.ws.send_str (data)
    
    async def close (self):
        """Close connection"""
        self.should_reconnect = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()

# Usage
# client = WebSocketClient('wss://api.example.com/ws')
# await client.connect()
\`\`\`

---

## Production Patterns

### Connection Pooling Optimization

\`\`\`python
"""
Optimized Session Configuration
"""

import aiohttp
from aiohttp import TCPConnector, ClientTimeout

class HTTPClient:
    """Production HTTP client with optimized settings"""
    
    def __init__(self):
        # Connection pooling
        connector = TCPConnector(
            limit=100,              # Total connections
            limit_per_host=30,      # Per host limit
            ttl_dns_cache=300,      # DNS cache 5 minutes
            enable_cleanup_closed=True,  # Clean up closed connections
        )
        
        # Timeouts
        timeout = ClientTimeout(
            total=30,
            connect=10,
            sock_read=10,
        )
        
        # Session with best practices
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=True,  # Auto-raise on HTTP errors
            headers={
                'User-Agent': 'MyApp/1.0',
            }
        )
    
    async def get (self, url, **kwargs):
        """GET with retry"""
        return await self._request('GET', url, **kwargs)
    
    async def post (self, url, **kwargs):
        """POST with retry"""
        return await self._request('POST', url, **kwargs)
    
    async def _request (self, method, url, retries=3, **kwargs):
        """Request with retry logic"""
        for attempt in range (retries):
            try:
                async with self.session.request (method, url, **kwargs) as response:
                    return await response.json()
            
            except aiohttp.ClientError as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def close (self):
        """Close session"""
        await self.session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()

# Usage
# async with HTTPClient() as client:
#     data = await client.get('https://api.example.com/data')
\`\`\`

---

## Summary

### Key Concepts

1. **ClientSession**
   - Reuse across requests
   - Manages connection pool
   - Configure timeouts and limits

2. **HTTP Methods**
   - GET: Retrieve data
   - POST: Create data (JSON, forms, files)
   - PUT/PATCH: Update data
   - DELETE: Remove data

3. **Response Handling**
   - \`.json()\`: Parse JSON
   - \`.text()\`: Get as string
   - \`.read()\`: Get as bytes
   - Stream with \`iter_chunked()\`

4. **Error Handling**
   - TimeoutError: Request timeout
   - ClientConnectorError: Connection failed
   - ClientResponseError: HTTP error status
   - Retry with exponential backoff

5. **Performance**
   - Connection pooling (reuse connections)
   - Concurrent requests (gather())
   - Streaming (memory efficient)
   - Configure timeouts

### Best Practices

1. ✅ Reuse ClientSession
2. ✅ Set appropriate timeouts
3. ✅ Implement retry logic
4. ✅ Stream large responses
5. ✅ Use connection pooling
6. ✅ Handle errors gracefully
7. ✅ Close sessions properly

### Next Steps

Now that you master async HTTP, we'll explore:
- Async database operations with asyncpg
- Async file I/O with aiofiles
- Building complete async applications
- Production deployment patterns

**Remember**: aiohttp enables concurrent HTTP requests, making your applications 10-100× faster for I/O-bound operations!
`,
};
