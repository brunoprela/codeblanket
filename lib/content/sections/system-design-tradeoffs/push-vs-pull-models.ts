/**
 * Push vs Pull Models Section
 */

export const pushvspullmodelsSection = {
  id: 'push-vs-pull-models',
  title: 'Push vs Pull Models',
  content: `The choice between push and pull data delivery models affects resource utilization, latency, and scalability. Understanding when to use each pattern is crucial for system design.

## Definitions

**Push Model**:
- **Server sends data to clients** proactively
- Server initiates data transfer
- Real-time updates
- Examples: WebSockets, Server-Sent Events (SSE), push notifications, CDN origin push

**Pull Model**:
- **Clients request data from server**
- Client initiates data transfer
- On-demand data retrieval
- Examples: HTTP requests, polling, CDN origin pull

---

## Push Model in Detail

### How It Works

Server actively sends updates to clients when data changes.

\`\`\`
Server                  Client
  |                       |
  |-- Data update ------->|  (Server initiates)
  |-- New message ------->|
  |-- Price change ------>|
\`\`\`

### Use Cases

**1. Real-Time Notifications**

**Example**: Chat application (Slack, WhatsApp)
- Server pushes new messages to clients immediately
- No polling needed
- Low latency (<100ms from send to receive)

**2. Live Updates**

**Example**: Stock price ticker
- Server pushes price updates every second
- Clients display updates in real-time
- Efficient (server only sends changes)

**3. Push Notifications**

**Example**: Mobile app notifications
- Server pushes to Apple Push Notification Service (APNS) / Firebase Cloud Messaging (FCM)
- Wakes up app even when closed
- Battery efficient (single connection for all apps)

### Advantages

✅ **Low latency**: Updates delivered immediately when data changes
✅ **Real-time**: Users see changes instantly
✅ **Efficient**: No unnecessary requests (only send when data changes)
✅ **Reduced client complexity**: Client doesn't manage polling

### Disadvantages

❌ **Persistent connections**: Need to maintain open connections (WebSockets)
❌ **Server resource usage**: More memory/CPU (thousands of open connections)
❌ **Complexity**: Need bidirectional communication infrastructure
❌ **Scaling challenges**: Connection state makes horizontal scaling harder

---

## Pull Model in Detail

### How It Works

Client requests data from server when needed.

\`\`\`
Client                  Server
  |                       |
  |-- GET /data --------->|  (Client initiates)
  |<-- Response ----------|
  |                       |
  |-- GET /data --------->|  (Client polls again)
  |<-- Response ----------|
\`\`\`

### Use Cases

**1. Static Content Delivery**

**Example**: CDN for images, JavaScript
- Client requests image when needed
- CDN pulls from origin on first request (origin pull)
- Caches for future requests

**2. API Requests**

**Example**: REST API calls
- Client requests data when user performs action
- Server responds with data
- Stateless (no persistent connection)

**3. Periodic Updates**

**Example**: Email inbox (check every 5 minutes)
- Client polls server for new emails
- User doesn't need instant updates
- Acceptable latency (up to 5 minutes)

### Advantages

✅ **Stateless**: No persistent connections, easier to scale
✅ **Simple**: Standard HTTP requests, well-understood
✅ **Client control**: Client decides when to fetch data
✅ **Easier horizontal scaling**: Load balancer can route to any server

### Disadvantages

❌ **Higher latency**: Delay between data change and client seeing it
❌ **Inefficient polling**: Wastes resources if no new data
❌ **Not real-time**: Delay based on polling interval

---

## Real-World Examples

### Example 1: Facebook News Feed (Hybrid)

**Pull model** (default):
- User opens app → Pulls latest posts
- User scrolls → Pulls more posts
- Simple, works offline (cached), scales well

**Push model** (optimization):
- New comment on your post → Push notification
- Friend goes live → Push notification
- Only for high-priority updates

**Why hybrid**: Pull for bulk data (timeline), push for critical notifications

---

### Example 2: YouTube Video Delivery (Pull)

**Pull model**:
- User clicks video → Client requests video chunks
- Client pulls chunks progressively (adaptive bitrate)
- CDN pulls from origin on cache miss

**Why pull**:
- Videos too large to push
- User controls playback (pause, seek)
- CDN caching very effective

---

### Example 3: Trading Platform (Push)

**Push model**:
- Server pushes stock price updates every 100ms
- WebSocket connection
- Real-time critical for trading

**Why push**:
- Low latency required (<100ms)
- Continuous stream of updates
- User needs instant data

---

## CDN: Push vs Pull

### CDN Origin Pull (More Common)

**How it works**:
1. User requests file from CDN edge server
2. If cached → Return immediately
3. If not cached → CDN pulls from origin server
4. CDN caches file
5. Future requests served from cache

**Advantages**:
- Automatic caching (only cache what's requested)
- No wasted storage (unpopular content not cached)
- Simple to set up

**Disadvantages**:
- First request slow (cache miss)
- Origin server must handle cache misses

**Use case**: Most static assets (images, CSS, JS)

---

### CDN Origin Push

**How it works**:
1. Developer uploads files to CDN
2. CDN distributes to all edge servers
3. File immediately available everywhere

**Advantages**:
- No cache misses (pre-warmed)
- Faster first request
- Predictable behavior

**Disadvantages**:
- Must manually upload all files
- Wastes storage (all files on all edges)
- More complex deployment

**Use case**: Critical assets, product launches (ensure availability)

---

## Feed Generation: Push vs Pull

### Twitter Timeline (Hybrid)

**Push model** (Fan-out on write):
- User tweets → Server writes to all followers' timelines (pre-compute)
- Follower opens app → Reads from pre-computed timeline (fast)

**Problem**: Celebrities with 100M followers → 100M writes per tweet!

**Solution**: Hybrid approach
- Regular users (<10K followers): Push (fan-out)
- Celebrities (>10K followers): Pull (on-demand)
- Merge both on timeline load

---

### LinkedIn Feed (Pull)

**Pull model** (Fan-out on read):
- User opens feed → Server fetches posts from connections
- Computes feed on demand (aggregation, ranking)

**Why pull**:
- Professional network (connections change frequently)
- Content less time-sensitive than Twitter
- Easier to personalize (compute at read time)

---

## Polling Patterns (Pull)

### Short Polling

Client polls server repeatedly at fixed intervals.

\`\`\`javascript
setInterval(() => {
  fetch('/api/messages')
    .then (res => res.json())
    .then (messages => updateUI(messages));
}, 5000); // Poll every 5 seconds
\`\`\`

**Advantages**: Simple to implement

**Disadvantages**: 
- Wastes bandwidth if no new data
- Latency = polling interval / 2 (average)
- High server load (constant requests)

---

### Long Polling

Client polls, server holds request open until data available.

\`\`\`javascript
async function longPoll() {
  const response = await fetch('/api/messages/poll');
  const messages = await response.json();
  updateUI(messages);
  longPoll(); // Poll again
}
\`\`\`

**Advantages**: 
- Lower latency than short polling
- Less wasteful (server responds only when data available)

**Disadvantages**:
- Ties up server connections
- Still uses more resources than WebSockets

---

## Trade-off Analysis

### Latency

**Push**: Near real-time (<100ms)
**Pull**: Depends on polling interval (seconds to minutes)

**Example**: Chat app
- Push: Message delivered instantly
- Pull (30s polling): Up to 30s delay

---

### Resource Utilization

**Push**: 
- Server: High (persistent connections)
- Network: Low (only send changes)

**Pull**:
- Server: Low (stateless, scales easily)
- Network: High (regular polls, even if no data)

**Example**: 1M users, updates every 10 minutes

**Push**: 1M WebSocket connections (high server memory)

**Pull**: 1M requests/minute (wastes bandwidth, but stateless)

---

### Scalability

**Push**: Harder to scale (connection state)
- Sticky sessions needed (user connected to specific server)
- Complex load balancing

**Pull**: Easier to scale (stateless)
- Any server can handle any request
- Standard load balancing

---

## Best Practices

### ✅ 1. Use Push for Real-Time Applications

Chat, live sports scores, stock trading → Push (WebSockets/SSE)

### ✅ 2. Use Pull for Static/On-Demand Content

Images, videos, API calls → Pull (HTTP)

### ✅ 3. Hybrid for Feeds

Bulk content: Pull
Critical updates: Push notifications

### ✅ 4. CDN: Default to Origin Pull

Only use origin push for critical assets or product launches

### ✅ 5. Implement Exponential Backoff for Polling

Reduce polling frequency if no new data

---

## Interview Tips

### Strong Answer Pattern

"For this system, I'd recommend:

**Push for**:
- [Real-time features like chat, notifications]
- Implementation: WebSockets or Server-Sent Events
- Trade-off: Higher server load, but real-time UX

**Pull for**:
- [Static content, API requests]
- Implementation: Standard HTTP requests, CDN
- Trade-off: Higher latency, but easier to scale

**Hybrid approach**:
- Timeline: Pull (bulk content)
- Notifications: Push (critical updates)
- Reasoning: Best of both worlds"

---

## Summary Table

| Aspect | Push | Pull |
|--------|------|------|
| **Initiator** | Server | Client |
| **Latency** | Near real-time (<100ms) | Polling interval (seconds+) |
| **Use Cases** | Chat, notifications, live updates | Static content, APIs, on-demand |
| **Connections** | Persistent (WebSocket) | Stateless (HTTP) |
| **Server Load** | High (maintain connections) | Low (stateless) |
| **Network Usage** | Efficient (only changes) | Wasteful (polls even if no data) |
| **Scalability** | Harder (connection state) | Easier (stateless) |
| **Examples** | WebSocket, SSE, push notifications | REST API, CDN origin pull, polling |

---

## Key Takeaways

✅ Push: Real-time, low latency, persistent connections, higher server load
✅ Pull: On-demand, stateless, easier to scale, higher latency
✅ Use push for real-time applications (chat, live data, notifications)
✅ Use pull for static/on-demand content (images, videos, API calls)
✅ Hybrid approach common: Pull for bulk, push for critical updates
✅ CDN origin pull more common (automatic caching), origin push for critical assets
✅ Twitter/Facebook use hybrid feed generation (push for regular users, pull for celebrities)`,
};
