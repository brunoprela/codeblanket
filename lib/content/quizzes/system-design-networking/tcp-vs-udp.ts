/**
 * Quiz questions for TCP vs UDP section
 */

export const tcpvsudpQuiz = [
  {
    id: 'tcp-vs-udp-choice',
    question:
      "You're building a multiplayer online game where players' positions are updated 60 times per second, but there's also a chat system and inventory management. Which transport protocol(s) would you use for each feature and why? Discuss the trade-offs and how you would handle potential issues like packet loss or latency.",
    sampleAnswer: `I would use a hybrid approach with different protocols for different features based on their requirements:

**1. Player Position Updates: UDP**

*Rationale*:
- Updates sent 60 times/second (every ~16ms)
- If a position update is lost, the next one arrives in 16ms
- Retransmitting old position is pointless (player already moved)
- TCP retransmission would cause "rubber banding" (player jumps backward)
- Low latency is critical for smooth gameplay

*Implementation*:
\`\`\`
struct PositionUpdate {
  playerId: uint32
  sequence: uint32        // Detect out-of-order packets
  timestamp: uint64       // Client can interpolate
  position: Vector3
  velocity: Vector3       // For dead reckoning
}
\`\`\`

*Handling packet loss*:
- **Client-side prediction**: Client predicts own movement immediately
- **Dead reckoning**: Estimate other players' positions using last known velocity
- **Interpolation**: Smooth movement between received positions
- **Accept 1-2% packet loss**: Not worth retransmitting

*Latency handling*:
- Lag compensation: Server rewinds game state when processing shots
- Show high-latency players' ping
- Regional servers to minimize RTT

**2. Chat System: TCP**

*Rationale*:
- Messages must arrive reliably (can't lose chat messages)
- Order matters (conversation flow)
- Users can tolerate 100-200ms delivery delay
- Security: TCP easier to secure with TLS

*Implementation*:
- Persistent WebSocket connection (over TCP)
- Message acknowledgments at application level too
- Retry logic for failed sends

*Alternative consideration*:
Could use UDP with application-level reliability (like WhatsApp), but TCP is simpler and delay is acceptable for chat.

**3. Inventory Management: TCP**

*Rationale*:
- **Critical data**: Losing item pickup/drop is unacceptable
- **Rare operations**: Not sent frequently like position
- **Order matters**: Pick up item then use item (must be in order)
- **Transactional**: Item trades must be reliable

*Implementation*:
\`\`\`
POST /api/inventory/pickup
{
  "itemId": "sword_123",
  "requestId": "unique-uuid"  // Idempotency key
}
\`\`\`

*Handling duplicate requests*:
- Idempotency keys prevent duplicate pickups if client retries
- Server-side validation (does player have space? is item still available?)

**Trade-offs Discussion**:

*Why not TCP for everything?*
- Position updates over TCP would cause latency spikes
- One lost packet blocks entire TCP stream (head-of-line blocking)
- TCP retransmits useless for real-time data
- 60 updates/sec * 100 players = 6000 updates/sec, TCP overhead too high

*Why not UDP for everything?*
- Chat messages can't be lost
- Inventory operations are critical
- Would need to rebuild TCP reliability at application layer
- TCP is battle-tested for reliable delivery

**Additional Optimizations**:

1. **Hybrid Protocol (UDP + reliability for critical events)**:
   - Most games use UDP for everything
   - Add sequence numbers and ACKs at application layer
   - Only retransmit critical events (item pickup, damage)
   - Skip retransmitting position updates

2. **State Synchronization**:
   - Full state sync every 10 seconds over TCP
   - Catch any drift from UDP packet loss
   - Helps players recover from disconnects

3. **Bandwidth Management**:
   - UDP: Compress position updates (e.g., delta compression)
   - Send updates for nearby players only (spatial filtering)
   - Reduce update rate for far-away players (30Hz instead of 60Hz)

4. **Fallback Mechanism**:
   - Some corporate networks block UDP
   - Fall back to WebSocket (TCP) with higher latency
   - Show warning: "Suboptimal connection detected"

**Monitoring**:
- Track packet loss percentage (alert if >5%)
- Monitor average RTT (latency)
- Measure client-side prediction accuracy
- Track desync events (client/server state mismatch)

**Real-world examples**:
- **Counter-Strike**: UDP for gameplay, TCP for server browser
- **League of Legends**: UDP primary, TCP fallback
- **Valorant**: Custom UDP protocol with application-level reliability

This hybrid approach balances performance (UDP for real-time) with reliability (TCP for critical data), providing the best player experience.`,
    keyPoints: [
      'Use UDP for high-frequency position updates (60Hz) where old data is useless',
      'Use TCP for chat and inventory where reliability and order matter',
      'Implement client-side prediction and interpolation to handle UDP packet loss',
      'Add idempotency keys for critical operations to prevent duplicates',
      'Consider hybrid UDP protocol with selective retransmission for critical events',
      'Always provide TCP fallback for networks that block UDP',
    ],
  },
  {
    id: 'tcp-optimization',
    question:
      "Your company's API serves mobile clients globally. Users complain about slow response times, especially on mobile networks. Explain how TCP's behavior contributes to this problem and what optimizations you would implement to improve performance. Consider both server-side and protocol-level changes.",
    sampleAnswer: `**TCP Behavior Contributing to Slowness**:

1. **Three-Way Handshake Latency**:
   - Mobile networks: 100-300ms RTT typical, can be 500ms+
   - TCP handshake requires 1 RTT before data
   - TLS adds another 1-2 RTT
   - Total: 2-3 RTT (200-900ms) before first byte of data

2. **Slow Start**:
   - TCP starts with small congestion window (typically 10 packets ~14KB)
   - Increases exponentially, but takes time
   - Problem: Initial API response might be 50KB JSON
   - Must wait multiple RTT to transmit entire response

3. **Mobile Network Characteristics**:
   - **High latency**: 100-300ms RTT (vs 10-50ms on broadband)
   - **Variable latency**: Spikes during handoffs, congestion
   - **Packet loss**: 1-5% typical (vs <0.1% on wired)
   - **Bandwidth asymmetry**: Fast download, slow upload

4. **Connection Resets**:
   - Mobile devices switch between WiFi/4G/5G
   - IP address changes → TCP connection broken
   - Must establish new connection (2-3 RTT penalty again)

**Optimization Strategy**:

**1. Enable TCP Fast Open (TFO)**

*What it does*:
- Skip 3-way handshake on subsequent connections
- Client sends SYN + Cookie + HTTP Request in one packet
- Server validates cookie and responds immediately

*Configuration* (Linux server):
\`\`\`bash
# Enable TFO
echo 3 > /proc/sys/net/ipv4/tcp_fastopen

# nginx configuration
server {
    listen 443 ssl fastopen=256;
}
\`\`\`

*Impact*: Saves 1 RTT (100-300ms on mobile)

*Limitation*: Only works on repeat connections

**2. Implement HTTP/3 with QUIC**

*Why QUIC is better for mobile*:
- **0-RTT connection resumption**: No handshake on repeat connections
- **Connection migration**: Survives IP changes (WiFi ↔ 4G)
- **No head-of-line blocking**: Independent streams
- **Built-in encryption**: TLS 1.3 integrated

*Implementation*:
\`\`\`nginx
# nginx with QUIC module
server {
    listen 443 quic reuseport;
    listen 443 ssl;
    
    # Tell clients QUIC is available
    add_header Alt-Svc 'h3=":443"; ma=86400';
}
\`\`\`

*Impact*: 
- 0-RTT instead of 2-3 RTT (saves 200-600ms)
- Survives network switches (no reconnection penalty)

**3. Increase Initial Congestion Window**

*Problem*: Default window (10 packets = 14KB) is too small

*Solution*: Increase to 32 packets (~45KB)
\`\`\`bash
# Linux server
ip route change default via [gateway] initcwnd 32 initrwnd 32
\`\`\`

*Impact*: 
- Small responses (<45KB) sent in one RTT
- Typical API response: 20-50KB → fits in initial window

**4. Enable BBR Congestion Control**

*Problem*: Traditional TCP (Cubic) is too conservative on high-latency networks

*Solution*: Google's BBR (Bottleneck Bandwidth and RTT)
\`\`\`bash
# Enable BBR (Linux 4.9+)
echo "tcp_bbr" >> /etc/modules-load.d/modules.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p
\`\`\`

*Impact*: 
- 2-5x faster throughput on high-latency networks
- Better packet loss recovery

**5. Use Connection Pooling & Keep-Alive**

*Client-side optimization*:
\`\`\`javascript
// Keep connections alive
fetch(url, {
  keepalive: true,
  headers: {
    'Connection': 'keep-alive'
  }
});

// Connection pooling (HTTP client library)
const agent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  keepAliveMsecs: 60000
});
\`\`\`

*Server-side*:
\`\`\`nginx
keepalive_timeout 65;
keepalive_requests 100;
\`\`\`

*Impact*: Reuse connections, avoid repeated handshakes

**6. Implement Compression**

*Enable Brotli/Gzip*:
\`\`\`nginx
gzip on;
gzip_types application/json text/plain text/css application/javascript;
gzip_min_length 1000;
gzip_comp_level 6;

# Brotli (better compression)
brotli on;
brotli_comp_level 6;
brotli_types application/json text/plain;
\`\`\`

*Impact*:
- 70-80% size reduction for JSON responses
- Fewer packets → less time with slow start

**7. Use CDN with Edge Computing**

*Strategy*:
- Place API endpoints at edge locations (Cloudflare Workers, AWS Lambda@Edge)
- Reduce RTT from 300ms (across ocean) to 20ms (nearby edge)
- Cache GET responses at edge

*Implementation*:
\`\`\`javascript
// Cloudflare Worker
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const cache = caches.default
  let response = await cache.match(request)
  
  if (!response) {
    response = await fetch(request)
    // Cache for 60 seconds
    event.waitUntil(cache.put(request, response.clone()))
  }
  
  return response
}
\`\`\`

*Impact*: 
- 200-300ms RTT reduction
- Handshake latency also reduced

**8. Optimize API Response Size**

*Techniques*:
- **GraphQL**: Let clients request only needed fields
- **Pagination**: Return 20 items instead of 1000
- **Lazy loading**: Fetch details on demand
- **Protocol Buffers**: Binary format (smaller than JSON)

*Example*:
\`\`\`graphql
# Instead of returning full user object
query {
  user(id: 123) {
    id
    name
    avatar  # Only what's needed
  }
}
\`\`\`

**9. Mobile-Specific Optimizations**

*Detect mobile clients*:
\`\`\`javascript
if (request.headers['User-Agent'].includes('Mobile')) {
  // Return smaller images
  // Reduce API response size
  // Enable aggressive caching
}
\`\`\`

*Adaptive responses*:
- Serve smaller thumbnails on slow connections
- Reduce video quality
- Defer non-critical data loading

**10. Implement Request Coalescing**

*Client-side*:
\`\`\`javascript
// Bad: 10 separate API calls
for (let id of userIds) {
  fetch(\`/ api / users / \${ id }\`)
}

// Good: One batch request
fetch('/api/users/batch', {
  method: 'POST',
  body: JSON.stringify({ ids: userIds })
})
\`\`\`

*Impact*: One connection instead of 10

**Monitoring & Validation**:

1. **Measure Key Metrics**:
   - Time to First Byte (TTFB)
   - Total Page Load Time
   - TCP connection time
   - TLS handshake time

2. **Use Real User Monitoring (RUM)**:
   - Collect performance data from actual mobile clients
   - Segment by network type (WiFi, 4G, 5G)
   - Identify problem regions

3. **Synthetic Testing**:
   - Test with simulated mobile networks
   - Use Chrome DevTools network throttling
   - Test from different geographic regions

**Expected Improvements**:

| Optimization | Latency Reduction | Complexity |
|--------------|-------------------|------------|
| TCP Fast Open | -100ms (1 RTT) | Low |
| HTTP/3 (QUIC) | -200-400ms | Medium |
| Increased cwnd | -100ms (small responses) | Low |
| BBR | 2-5x throughput | Low |
| CDN/Edge | -200-300ms | Medium |
| Response compression | -200ms (transfer time) | Low |

**Total Expected Improvement**: 50-70% reduction in API response time for mobile users.

**Real-world Example**:
- **Google**: Switching to QUIC improved search latency by 8% (desktop), 3.6% (mobile)
- **Facebook**: HTTP/3 reduced median request time by 12.4%
- **Cloudflare**: BBR improved throughput 2-3x on high-latency connections`,
    keyPoints: [
      'TCP handshake (1 RTT) and TLS (1-2 RTT) cause 200-900ms delay on mobile networks',
      'TCP slow start limits initial throughput, especially problematic with high latency',
      'Enable TCP Fast Open to save 1 RTT on repeat connections',
      'Implement HTTP/3 (QUIC) for 0-RTT and connection migration across network switches',
      'Use BBR congestion control for better performance on high-latency mobile networks',
      'Deploy edge computing/CDN to reduce geographic latency',
      'Compress responses and increase initial congestion window',
      'Measure TTFB and total load time with real user monitoring',
    ],
  },
  {
    id: 'dns-udp-tcp',
    question:
      'DNS primarily uses UDP, but sometimes falls back to TCP. Explain why UDP is preferred for DNS, in what situations TCP is used, and how you would design a high-performance DNS resolver that handles millions of queries per second. What caching strategies would you implement?',
    sampleAnswer: `**Why DNS Uses UDP**:

**1. Query/Response Fits in One Packet**:
- Typical DNS query: ~50-100 bytes
- Typical DNS response: ~200-500 bytes
- UDP packet max: 512 bytes (traditional), 4096 bytes (EDNS0)
- Single packet = single round trip

**2. TCP Overhead Too High**:
\`\`\`
UDP DNS Query:
  Client → Server: Query (1 packet)
  Server → Client: Response (1 packet)
  Total: 1 RTT, 2 packets

TCP DNS Query:
  Client → Server: SYN
  Server → Client: SYN-ACK
  Client → Server: ACK + Query
  Server → Client: Response
  Client → Server: ACK
  Client/Server: FIN, FIN-ACK (connection close)
  Total: 2-3 RTT, 7+ packets
\`\`\`

**3. Performance Implications**:
- TCP: 3x more packets, 2x more latency
- For 1 billion DNS queries/day:
  - UDP: 2 billion packets
  - TCP: 7 billion packets (3.5x load)

**4. Stateless = Scalable**:
- UDP has no connection state
- Server doesn't track connections
- Easy to load balance (any server can handle any request)
- Better for DDoS resilience

**When DNS Uses TCP**:

**1. Response Too Large (>512 bytes)**:
- Initial query over UDP
- If response is truncated (TC bit set), retry over TCP
- Common for:
  - DNSSEC responses (cryptographic signatures are large)
  - Many MX records
  - Large TXT records

**2. Zone Transfers (AXFR/IXFR)**:
- Transferring entire DNS zone to secondary nameserver
- Can be megabytes of data
- Requires TCP for reliability

**3. DNS over TLS (DoT) / DNS over HTTPS (DoH)**:
- Privacy-focused DNS
- Always uses TCP (TLS requires TCP)
- Port 853 (DoT) or 443 (DoH)

**Designing High-Performance DNS Resolver**:

**Architecture Overview**:
\`\`\`
Client → [Load Balancer] → [DNS Resolver Cluster] → [Cache Layer] → [Authoritative Servers]
                              ↓
                          [Metrics & Logs]
\`\`\`

**Component 1: Load Balancer**

*Technology*: Anycast + DNS
- Same IP address advertised from multiple locations
- Network routes queries to nearest server
- Geographic distribution

*Implementation*:
\`\`\`
# BGP Anycast configuration
# Advertise 1.1.1.1 from 200+ locations globally
# Network automatically routes to closest
\`\`\`

*Alternative*: ECMP (Equal-Cost Multi-Path)
- Distribute across multiple servers in same datacenter
- Hash-based distribution (source IP + source port)

**Component 2: DNS Resolver**

*Technology Choice*: 
- **unbound** (C, high performance)
- **PowerDNS Recursor**
- **BIND** (traditional, but slower)
- **Custom**: Cloudflare uses Rust-based resolver

*Configuration* (unbound):
\`\`\`yaml
server:
    # Performance tuning
    num-threads: 8           # Match CPU cores
    msg-cache-size: 256m     # Cache responses
    rrset-cache-size: 512m   # Cache resource records
    cache-min-ttl: 60        # Minimum cache time
    cache-max-ttl: 86400     # Maximum cache time
    
    # Prefetching (predict queries)
    prefetch: yes
    prefetch-key: yes
    
    # UDP optimization
    so-rcvbuf: 4m            # Large receive buffer
    so-sndbuf: 4m            # Large send buffer
    
    # Rate limiting (DDoS protection)
    ratelimit: 1000          # 1000 queries/sec per IP
\`\`\`

**Component 3: Caching Strategy**

**L1 Cache: In-Memory (per server)**
- **Technology**: Hash table in RAM
- **Size**: 256MB-1GB per server
- **TTL**: Respect DNS TTL from authoritative server
- **Eviction**: TTL-based (automatic expiration)

*Structure*:
\`\`\`
Key: (domain, record_type)
  Example: ("example.com", A)

Value: {
  records: ["93.184.216.34"],
  ttl: 300,
  expires_at: 1640000000
}
\`\`\`

**L2 Cache: Distributed (Redis/Memcached)**
- Share cache across resolver cluster
- Backup when local cache cold (server restart)
- 10-20GB capacity

*Implementation*:
\`\`\`python
def resolve(domain, record_type):
    # L1: Check local memory
    key = (domain, record_type)
    if key in local_cache and not expired(local_cache[key]):
        return local_cache[key]
    
    # L2: Check Redis
    redis_key = f"dns:{domain}:{record_type}"
    cached = redis.get(redis_key)
    if cached:
        # Populate L1 cache
        local_cache[key] = cached
        return cached
    
    # L3: Query authoritative servers
    response = query_authoritative(domain, record_type)
    
    # Cache response
    ttl = response.ttl
    local_cache[key] = response
    redis.setex(redis_key, ttl, response)
    
    return response
\`\`\`

**Caching Optimizations**:

**1. Prefetching**:
- When TTL has 10% remaining, fetch fresh record in background
- Ensures cache always hot for popular domains
- Reduces query latency

*Example*:
\`\`\`python
if record.ttl_remaining < record.original_ttl * 0.1:
    background_task.submit(refresh_record, domain, record_type)
\`\`\`

**2. Negative Caching**:
- Cache NXDOMAIN (domain doesn't exist) responses
- Prevents repeated queries for non-existent domains
- Typical TTL: 5-15 minutes

*Example*:
\`\`\`
Query: doesntexist.example.com
Response: NXDOMAIN (cache for 300 seconds)
\`\`\`

**3. Aggressive DNSSEC Caching**:
- DNSSEC responses prove non-existence cryptographically
- Can cache broader range (entire zone)

**4. Minimum TTL**:
- Some domains set TTL=0 (no cache)
- Ignore this, use minimum TTL (30-60 seconds)
- Trade-off: Slight staleness vs performance

**5. Cache Partitioning**:
- Partition cache by popularity
- Hot cache (top 10k domains): Never evict
- Warm cache (top 1M domains): Normal LRU
- Cold cache: Aggressive eviction

**Performance Optimizations**:

**1. Kernel Tuning (Linux)**:
\`\`\`bash
# Increase UDP buffer sizes
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456

# Increase connection tracking table
sysctl -w net.netfilter.nf_conntrack_max=1048576

# Increase ephemeral ports
sysctl -w net.ipv4.ip_local_port_range="10000 65535"

# Enable TCP Fast Open
sysctl -w net.ipv4.tcp_fastopen=3
\`\`\`

**2. Use Multiple Threads/Processes**:
- Bind each thread to CPU core (avoid context switching)
- Separate threads for UDP and TCP
- Lock-free data structures for cache

**3. Batch Processing**:
- Process multiple DNS queries in batch
- Reduces context switches
- Use io_uring (Linux 5.1+) for async I/O

**4. Connection Pooling (for upstream)**:
- Keep connections to authoritative servers open
- Reuse TCP connections for large responses
- Connection pool per upstream server

**5. Smart Timeout Handling**:
\`\`\`python
# Query multiple authoritative servers in parallel
responses = await asyncio.gather(
    query_ns1(domain),
    query_ns2(domain),
    return_exceptions=True
)
# Return first successful response
return next(r for r in responses if r.success)
\`\`\`

**Handling Millions of QPS**:

**Capacity Planning**:
- 1 server: ~50,000 QPS (with caching)
- 1 million QPS → 20 servers per datacenter
- 5 datacenters globally → 100 servers total

**Horizontal Scaling**:
- Stateless resolvers (easy to scale)
- Add servers behind load balancer
- Anycast distributes load geographically

**Cache Hit Rate Optimization**:
- Target: 95%+ cache hit rate
- 95% hit rate → 50k requests/sec per server
- 5% miss rate → 2,500 authoritative queries/sec per server

**DDoS Protection**:

**1. Rate Limiting**:
\`\`\`python
# Per-IP rate limit
if query_count[client_ip] > 100:  # 100 queries/sec
    return REFUSED
\`\`\`

**2. Query Validation**:
- Check for malformed queries
- Drop queries with suspicious patterns
- Validate EDNS0 payload size

**3. Response Rate Limiting (RRL)**:
- Limit identical responses (prevents amplification)
- If same response sent >5 times/sec, drop subsequent

**4. Anycast Sink Holing**:
- Detect attack traffic
- Route to dedicated scrubbing servers
- Clean traffic forwarded to resolvers

**Monitoring & Metrics**:

**Key Metrics**:
1. **QPS** (queries per second)
2. **Cache hit rate** (target: 95%+)
3. **Latency** (p50, p99, p999)
4. **Error rate** (SERVFAIL, NXDOMAIN)
5. **Upstream query rate**

*Implementation*:
\`\`\`python
# Prometheus metrics
dns_queries_total.inc()
dns_cache_hit_total.inc()
dns_query_duration.observe(latency_ms)
\`\`\`

**Real-World Examples**:

**Cloudflare 1.1.1.1**:
- Handles 700+ billion DNS queries/day
- Uses Anycast from 200+ locations
- Custom Rust-based resolver
- <10ms average response time globally

**Google Public DNS (8.8.8.8)**:
- 400+ billion queries/day
- Anycast from Google data centers globally
- Heavy prefetching and caching
- DNSSEC validation

**AWS Route 53**:
- 100% SLA (no downtime)
- Uses multiple Anycast networks
- Health checks and failover
- Geographic load balancing

**Expected Performance**:

| Metric | Target | Reality |
|--------|--------|---------|
| QPS per server | 50,000 | 30,000-100,000 |
| Latency (p50) | <10ms | 5-20ms |
| Latency (p99) | <50ms | 20-100ms |
| Cache hit rate | 95%+ | 90-98% |
| Availability | 99.99% | 99.95-99.99% |

**Cost Considerations**:
- UDP: ~$0.10 per million queries
- TCP: ~$0.30 per million queries
- Prefer UDP for cost and performance

This architecture provides a scalable, high-performance DNS resolver capable of handling millions of QPS with low latency and high availability.`,
    keyPoints: [
      "DNS uses UDP because queries fit in single packet, avoiding TCP's 3-way handshake overhead",
      'TCP used when response >512 bytes, zone transfers, or DNS-over-TLS/HTTPS',
      'Multi-layer caching: L1 in-memory (per server), L2 distributed (Redis)',
      'Prefetching keeps cache hot by refreshing records before TTL expires',
      'Anycast distributes load geographically, routing queries to nearest server',
      'Target 95%+ cache hit rate to minimize upstream queries',
      'Rate limiting and response rate limiting (RRL) protect against DDoS',
      '50k QPS per server typical with good caching; horizontal scaling via stateless resolvers',
    ],
  },
];
