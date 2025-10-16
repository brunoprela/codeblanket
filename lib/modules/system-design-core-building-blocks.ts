import { Module } from '../types';

export const systemDesignCoreBuildingBlocksModule: Module = {
  id: 'system-design-core-building-blocks',
  title: 'Core Building Blocks',
  description:
    'Master the fundamental components that power all distributed systems including load balancing, caching, sharding, and message queues',
  icon: '🏗️',
  sections: [
    {
      id: 'load-balancing',
      title: 'Load Balancing',
      content: `Load balancing is a critical component in distributed systems that distributes incoming network traffic across multiple servers to ensure no single server bears too much demand.

## What is Load Balancing?

**Definition**: A load balancer acts as a traffic cop sitting in front of your servers and routing client requests across all servers capable of fulfilling those requests in a manner that maximizes speed and capacity utilization.

### **Why Load Balancing Matters**

**Without Load Balancer:**
- Single point of failure (server goes down = entire app down)
- Limited by single server capacity
- Can't scale horizontally
- No redundancy

**With Load Balancer:**
- High availability (if one server fails, traffic routes to others)
- Horizontal scaling (add more servers as needed)
- Better resource utilization
- Improved response times

---

## Load Balancing Algorithms

### **1. Round Robin**

**How it works**: Requests distributed sequentially to each server in rotation.

**Example:**
- Request 1 → Server A
- Request 2 → Server B
- Request 3 → Server C
- Request 4 → Server A (back to start)

**Pros:**
- Simple to implement
- Fair distribution (each server gets equal requests)
- No need to track server state

**Cons:**
- Doesn't account for server capacity (powerful server treats same as weak server)
- Doesn't consider current server load
- Can be inefficient if servers have different capabilities

**Use case:** When all servers have similar capacity and requests require similar processing time.

---

### **2. Weighted Round Robin**

**How it works**: Servers assigned weights based on capacity. Higher weight = more requests.

**Example:** Server A (weight 5), Server B (weight 3), Server C (weight 2)
- 5 requests → Server A
- 3 requests → Server B
- 2 requests → Server C
- Repeat

**Pros:**
- Accounts for different server capacities
- More powerful servers handle more load
- Still simple to implement

**Cons:**
- Weights need to be configured manually
- Doesn't adapt to real-time server load

**Use case:** When servers have different hardware capabilities.

---

### **3. Least Connections**

**How it works**: Routes requests to the server with the fewest active connections.

**Example:**
- Server A: 10 active connections
- Server B: 5 active connections  
- Server C: 8 active connections
- **Next request → Server B** (fewest connections)

**Pros:**
- Considers actual server load
- Better for long-lived connections
- Adapts to real-time traffic

**Cons:**
- Requires tracking connections
- More complex than round robin
- Connection count ≠ actual load (some connections idle)

**Use case:** Applications with long-lived connections (databases, WebSockets, persistent HTTP connections).

---

### **4. Least Response Time**

**How it works**: Routes to the server with the fastest response time and fewest active connections.

**Calculation:** Response Time + Active Connections

**Example:**
- Server A: 50ms response, 10 connections = 60
- Server B: 30ms response, 5 connections = 35 ✅ (chosen)
- Server C: 40ms response, 8 connections = 48

**Pros:**
- Considers both load and performance
- Routes to fastest servers
- Adapts to server health

**Cons:**
- Requires active health monitoring
- More computational overhead
- Response time can be noisy metric

**Use case:** When server performance varies or you need optimal response times.

---

### **5. IP Hash**

**How it works**: Hash of client's IP address determines which server receives request. Same IP always routes to same server.

**Calculation:** \`hash(client_ip) % server_count\`

**Example:**
- Client 1.2.3.4 → hash % 3 = 1 → Server B
- Client 5.6.7.8 → hash % 3 = 0 → Server A
- Client 1.2.3.4 → always Server B

**Pros:**
- Session persistence (sticky sessions)
- No need for shared session storage
- Simple to implement

**Cons:**
- Uneven distribution if client IPs not evenly distributed
- Adding/removing servers disrupts mappings
- NAT can cause many users to appear as same IP

**Use case:** When session state stored locally on servers (not recommended for stateless design).

---

### **6. Random**

**How it works**: Selects a random server for each request.

**Pros:**
- Simple
- No state to maintain
- Prevents thundering herd

**Cons:**
- Uneven distribution possible (statistically evens out over time)
- Doesn't consider server capacity or load

**Use case:** When servers are homogeneous and requests are lightweight.

---

## Layer 4 vs Layer 7 Load Balancing

### **Layer 4 (Transport Layer) Load Balancing**

**What it does**: Routes based on network information (IP address, TCP port).

**How it works:**
- Examines TCP/UDP packet
- Routes based on source/destination IP and port
- Doesn't inspect packet contents

**Characteristics:**
- **Fast**: Minimal processing overhead
- **Simple**: No need to parse HTTP
- **Protocol-agnostic**: Works for any TCP/UDP traffic

**Example:** AWS Network Load Balancer (NLB)

**Use case:**
- High throughput requirements
- Non-HTTP traffic (database connections, custom protocols)
- Extreme performance needs (millions of requests/sec)

---

### **Layer 7 (Application Layer) Load Balancing**

**What it does**: Routes based on application-level data (HTTP headers, cookies, URL path).

**How it works:**
- Parses HTTP request
- Routes based on URL, headers, cookies
- Can modify request/response

**Characteristics:**
- **Intelligent**: Content-based routing
- **Feature-rich**: SSL termination, URL rewriting, cookie injection
- **Slower**: More processing overhead

**Example:** AWS Application Load Balancer (ALB), NGINX

**Routing examples:**
- /api/users/* → User service
- /api/posts/* → Post service
- Header "X-Mobile: true" → Mobile-optimized backend

**Use case:**
- Microservices (route based on URL path)
- A/B testing (route based on cookies)
- Need SSL termination
- HTTP-specific features

---

## Health Checks

**Purpose**: Detect unhealthy servers and stop sending traffic to them.

### **Active Health Checks**

Load balancer periodically pings servers.

**Example:**
- HTTP: \`GET / health\` every 5 seconds
- Expect: 200 OK response within 2 seconds
- Fail threshold: 3 consecutive failures
- Success threshold: 2 consecutive successes

**Configuration:**
- **Interval**: How often to check (e.g., 5 seconds)
- **Timeout**: Max wait time for response (e.g., 2 seconds)
- **Unhealthy threshold**: Consecutive failures before marking unhealthy (e.g., 3)
- **Healthy threshold**: Consecutive successes to mark healthy again (e.g., 2)

---

### **Passive Health Checks**

Monitor actual traffic. If server returns errors, mark as unhealthy.

**Example:**
- Server returns 5 consecutive 500 errors → Mark unhealthy
- After cooldown period, retry

**Advantage:** No additional health check traffic

---

## Session Persistence (Sticky Sessions)

**Problem**: User's session data stored on specific server. If next request goes to different server, session lost.

### **Solution 1: Cookie-Based Stickiness**

Load balancer sets a cookie with server identifier.

**Flow:**
1. User's first request → LB routes to Server A
2. LB sets cookie: \`LB_COOKIE = server_a\`
3. Subsequent requests include cookie → Always route to Server A

**Pros:** Simple, works across LB restarts

**Cons:** Not secure (cookie can be tampered), breaks if server fails

---

### **Solution 2: IP Hash**

Hash client IP to determine server (discussed above).

**Pros:** No cookies needed

**Cons:** NAT issues, inflexible

---

### **Solution 3: Stateless Design (Recommended)**

Store session in shared storage (Redis, database).

**Flow:**
1. User logs in → Session stored in Redis with session ID
2. Server sets cookie: \`SESSION_ID = abc123\`
3. Any server can retrieve session from Redis using session ID

**Pros:**
- No stickiness needed
- Servers truly stateless
- Better fault tolerance

**This is the recommended approach for distributed systems!**

---

## Global vs Local Load Balancing

### **Local Load Balancing**

**Scope**: Within a single data center

**Example:**
- 10 application servers in US-East-1 datacenter
- One load balancer distributes traffic among them

---

### **Global Load Balancing (GSLB)**

**Scope**: Across multiple data centers/regions

**How it works**: DNS-based routing to nearest datacenter

**Example:**
- User in Europe → routed to EU datacenter
- User in Asia → routed to Asia datacenter

**Benefits:**
- Reduced latency (users hit nearest datacenter)
- Geographic redundancy
- Disaster recovery

**Implementation:**
- Route53 (AWS)
- Cloud DNS (GCP)
- Azure Traffic Manager

**Routing strategies:**
- **Geolocation**: Based on user's location
- **Latency-based**: Route to lowest-latency endpoint
- **Failover**: Primary datacenter down → route to secondary

---

## Load Balancer Failure & High Availability

**Problem**: If load balancer fails, entire system goes down. Load balancer itself is a single point of failure!

### **Solution: Multiple Load Balancers**

**Active-Active:**
- Multiple LBs running simultaneously
- DNS returns multiple LB IPs
- Client tries IPs in order

**Active-Passive:**
- Primary LB handles traffic
- Secondary LB on standby
- Heartbeat between them
- If primary fails, secondary takes over (failover)

**Cloud-managed LBs:**
- AWS ELB automatically highly available
- Spans multiple availability zones
- Managed by AWS

---

## Real-World Examples

### **NGINX**

Open-source HTTP server and reverse proxy.

**Configuration example:**
\`\`\`
upstream backend {
            least_conn;  # Algorithm
    server backend1.example.com weight=5;
            server backend2.example.com weight=3;
            server backend3.example.com;
        }

        ** NGINX Config:**
        - upstream backend with least_conn algorithm
- server backend1.example.com weight = 5
        - server backend2.example.com weight = 3
            - server backend3.example.com
- listen on port 80
    - proxy_pass to http://backend
\`\`\`
---

### ** AWS Elastic Load Balancer(ELB) **

** Types:**
- ** Application Load Balancer(ALB) **: Layer 7, HTTP / HTTPS
    - ** Network Load Balancer(NLB) **: Layer 4, TCP / UDP, extreme performance
        - ** Classic Load Balancer **: Legacy, both L4 and L7

            ** Features:**
                - Automatic scaling
                    - SSL / TLS termination
                        - Health checks
                            - Multi - AZ high availability

---

### ** HAProxy **

    High - performance TCP / HTTP load balancer.

** Use cases:**
    - High - traffic websites
        - Database load balancing
            - Replaces expensive hardware load balancers

---

## Interview Tips

### ** Common Questions:**

** Q: "How would you design a load balancer?" **

✅ Good answer:
1. Clarify requirements: Scale ? Layer 4 or 7 ? Health checks ?
    2. Discuss algorithm choice and justify
3. Explain health check mechanism
4. Address high availability of LB itself
5. Mention SSL termination, connection pooling

    ** Q: "What load balancing algorithm would you use for a video streaming service?" **

✅ Good answer: "Least Connections or Least Response Time because:
    - Video streams are long - lived connections
        - Round Robin wouldn't consider that Server A has 100 active streams and Server B has 10
            - We want to distribute based on actual load, not just request count"

                ** Q: "How do you handle load balancer failure?" **

✅ Good answer:
- Multiple LBs(active - active or active - passive)
    - Health checks between LBs
        - Automatic failover
            - Cloud - managed LBs handle this automatically

---

## Key Takeaways

1. ** Load balancers distribute traffic across multiple servers for availability and scalability **
    2. ** Algorithm choice matters **: Round Robin for simple cases, Least Connections for long - lived connections
3. ** Layer 4 vs Layer 7 **: Choose based on need for content - based routing
4. ** Health checks are critical **: Detect and remove unhealthy servers
5. ** Avoid sticky sessions **: Design stateless servers with shared session storage
6. ** LB itself needs high availability **: Active - passive or active - active configuration
7. ** Global load balancing **: Route users to nearest datacenter for reduced latency`,
      quiz: [
        {
          id: 'q1',
          question:
            "You're designing a system where user sessions contain large amounts of state stored in server memory. The load balancer is currently using Round Robin. Users complain they're randomly logged out. Explain what's happening and propose two solutions with trade-offs.",
          sampleAnswer:
            "PROBLEM DIAGNOSIS: Users are being logged out because Round Robin distributes each request randomly across servers. When User A logs in on Server 1, their session is stored in Server 1's memory. The next request from User A might go to Server 2, which doesn't have their session data, making it appear they're logged out. This is the classic session persistence problem. SOLUTION 1: STICKY SESSIONS (Quick Fix): Configure load balancer to use IP Hash or cookie-based stickiness. Each user consistently routed to same server. Implementation: Set cookie on first request, use that cookie to route all subsequent requests to same server. PROS: (1) Quick to implement, no architecture changes. (2) Session data stays in memory (fast). CONS: (1) If server crashes, all sessions on that server are lost (users logged out). (2) Uneven load distribution (popular users might all hash to same server). (3) Doesn't scale well (can't easily add/remove servers). (4) Prevents true stateless design. When to use: Temporary fix, legacy systems, low-traffic applications. SOLUTION 2: SHARED SESSION STORAGE (Recommended): Store sessions in Redis or database instead of server memory. Any server can retrieve any session. Implementation: (1) User logs in → Server generates session ID, stores session data in Redis. (2) Server sets cookie with session ID. (3) Next request → Any server retrieves session from Redis using session ID. PROS: (1) Servers truly stateless (can route to any server). (2) No data loss if server crashes. (3) Easy to add/remove servers. (4) Better load distribution. (5) Scalable horizontally. CONS: (1) Extra network hop to Redis (slight latency increase). (2) Redis becomes dependency (single point of failure - mitigate with Redis cluster). (3) Requires architecture change. When to use: Production systems, high-availability requirements, any system that needs to scale. PERFORMANCE COMPARISON: Sticky sessions: Session retrieval ~0ms (in memory). Shared storage: Session retrieval ~1-2ms (Redis). For most applications, 1-2ms is acceptable trade-off for much better reliability and scalability. RECOMMENDED APPROACH: Implement Solution 2 (shared session storage). This is standard practice in distributed systems. Make Redis highly available (Redis Sentinel or Redis Cluster). If latency critical, cache frequently-accessed session data in server memory but always verify with Redis (cache-aside pattern). WHY STATELESS IS BETTER: (1) Servers interchangeable (no special handling). (2) Auto-scaling works properly (can add/remove servers freely). (3) Deployment easier (can update servers one at a time). (4) Better fault tolerance. MIGRATION PATH: If currently using sticky sessions: (1) Deploy Redis cluster. (2) Write session data to BOTH memory and Redis (dual write). (3) Read from memory first (for performance), fall back to Redis if missing. (4) After verification period, remove memory storage. (5) Remove sticky session configuration.",
          keyPoints: [
            'Round Robin without session persistence causes random logouts',
            'Sticky sessions (quick fix): keeps user on same server, but not scalable',
            'Shared session storage (recommended): Redis/database for sessions, servers stateless',
            'Stateless design enables horizontal scaling, better fault tolerance',
            'Trade-off: slight latency increase (1-2ms) vs much better scalability and reliability',
          ],
        },
        {
          id: 'q2',
          question:
            'Your application has 3 servers: Server A (32 cores, 64GB RAM), Server B (16 cores, 32GB RAM), Server C (16 cores, 32GB RAM). Which load balancing algorithm would you choose and why? What configuration parameters would you set?',
          sampleAnswer:
            "ANALYSIS: Servers have different capacities. Server A is 2× as powerful as Servers B and C. This rules out simple Round Robin (would overload B and C, underutilize A). RECOMMENDED ALGORITHM: WEIGHTED ROUND ROBIN: Why: (1) Accounts for different server capacities. (2) More powerful Server A gets more requests. (3) Simple to implement and configure. (4) Predictable behavior. CONFIGURATION: Server A: Weight = 4 (32 cores). Server B: Weight = 2 (16 cores). Server C: Weight = 2 (16 cores). Ratio: 4:2:2 means for every 8 requests: 4 → Server A, 2 → Server B, 2 → Server C. Result: Server A handles 50% of traffic, B and C each handle 25%. This matches their relative capacities. ALTERNATIVE ALGORITHM: LEAST RESPONSE TIME: If servers have variable performance or different hardware beyond CPU (e.g., SSD vs HDD), Least Response Time adapts dynamically. Configuration: Monitor response times every 5 seconds, route to server with best (response time + active connections). Why this might be better: (1) Adapts to actual performance. (2) If Server A is overloaded, automatically routes less traffic to it. (3) Handles heterogeneous environments better. Trade-off: More complex, requires active monitoring. IMPLEMENTATION DETAILS: Health Checks: Interval: 5 seconds, Timeout: 2 seconds, Unhealthy threshold: 3 consecutive failures, Healthy threshold: 2 consecutive successes. Connection Limits: Set max connections per server based on capacity: Server A: max 1000 connections, Server B: max 500 connections, Server C: max 500 connections. Prevents any server from being overwhelmed. Timeout Configuration: Request timeout: 30 seconds (prevent slow requests from tying up resources), Idle timeout: 60 seconds (close inactive connections). NGINX CONFIGURATION EXAMPLE: upstream backend { least_conn; # Or use weighted round robin server server-a.example.com:8080 weight=4 max_conns=1000; server server-b.example.com:8080 weight=2 max_conns=500; server server-c.example.com:8080 weight=2 max_conns=500; } MONITORING: Track per-server metrics: (1) Request count (verify weights working correctly). (2) CPU utilization (should be similar across all servers ~80%). (3) Response times (identify bottlenecks). (4) Error rates (detect failing servers). Alert if: Any server consistently at 100% CPU, Response time > 1 second at p95, Error rate > 1%. SCALING STRATEGY: If all servers reaching capacity: Add more servers with appropriate weights. Scale horizontally (add more 16-core servers) rather than vertically (less cost-effective). If Server A is bottleneck: Could split into 2× 16-core servers (easier to scale horizontally). COST CONSIDERATION: Running one 32-core server might be more cost-effective than two 16-core servers (depends on cloud provider pricing). But multiple smaller servers give better fault tolerance. REAL-WORLD EXAMPLE: At scale (e.g., Netflix, AWS): Don't mix server sizes - use homogeneous server types for simplicity. Easier operations: Any server can handle any load, simpler configuration, easier capacity planning. Auto-scaling groups use identical instance types. But in your scenario with existing heterogeneous hardware: Weighted Round Robin is the pragmatic solution. FINAL RECOMMENDATION: Start with Weighted Round Robin (4:2:2 ratio). Monitor closely for 1 week. If seeing uneven load despite weights: Switch to Least Response Time (more dynamic). If working well: Keep it simple.",
          keyPoints: [
            'Heterogeneous servers require Weighted Round Robin or Least Response Time',
            'Weight based on server capacity: 32-core gets 2× weight of 16-core',
            'Set max connection limits per server to prevent overload',
            'Monitor CPU utilization to verify weights are correct (all ~80%)',
            'Ideal: Use homogeneous servers for simplicity (same hardware for all)',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the difference between Layer 4 and Layer 7 load balancing. Give a real-world scenario where you would choose each, and explain why.',
          sampleAnswer:
            "LAYER 4 (TRANSPORT LAYER) LOAD BALANCING: What it examines: IP addresses, TCP/UDP ports, Network layer information only. What it CANNOT see: HTTP headers, URL paths, cookies, Request body. How it routes: Based on IP:port only. All requests from same client/port go to same backend server. Speed: VERY FAST (minimal processing, just forwards packets). Example: AWS Network Load Balancer (NLB), HAProxy in TCP mode. LAYER 7 (APPLICATION LAYER) LOAD BALANCING: What it examines: Full HTTP request: URL path, headers, cookies, query parameters. What it CAN do: Content-based routing, URL rewriting, SSL termination, Request/response modification. How it routes: Based on application-level data: /api/users → User Service, /api/posts → Post Service. Speed: SLOWER (must parse HTTP, more processing). Example: AWS Application Load Balancer (ALB), NGINX, HAProxy in HTTP mode. KEY DIFFERENCES COMPARISON: Processing: L4: Fast (just IP routing), L7: Slower (HTTP parsing). Routing Intelligence: L4: Basic (IP/port), L7: Advanced (content-based). Protocol: L4: Any TCP/UDP, L7: HTTP/HTTPS only. SSL Termination: L4: No (passes through), L7: Yes (can decrypt/encrypt). Use Cases: L4: High throughput, L7: Microservices, intelligent routing. SCENARIO 1: CHOOSE LAYER 4: Real-world: Database load balancing. Requirements: (1) Need to load balance PostgreSQL connections (not HTTP). (2) High throughput (100K connections/sec). (3) Minimal latency (every millisecond matters). (4) No need for intelligent routing (all requests go to read replicas). Setup: Layer 4 load balancer distributes PostgreSQL connections (port 5432) across 10 read replicas using Least Connections algorithm. Why Layer 4: (1) PostgreSQL is not HTTP (Layer 7 wouldn't work). (2) Need maximum performance (Layer 4 faster). (3) Simple round-robin to replicas sufficient. (4) No need to inspect query content. Configuration: Protocol: TCP, Port: 5432, Algorithm: Least Connections (long-lived DB connections), Health Check: TCP connect to port 5432. SCENARIO 2: CHOOSE LAYER 7: Real-world: Microservices architecture. Requirements: (1) Route different API paths to different services: /api/users/* → User Service, /api/posts/* → Post Service, /api/comments/* → Comment Service. (2) Need SSL termination (decrypt HTTPS at LB, HTTP to backends). (3) A/B testing (route 10% traffic to new service version). (4) Header-based routing (mobile app gets different backend). Setup: Layer 7 load balancer examines URL path and routes accordingly. Why Layer 7: (1) Need content-based routing (path inspection). (2) SSL termination (Layer 4 can't decrypt). (3) A/B testing requires cookie/header inspection. (4) Can't do this with Layer 4 (doesn't see URL). Configuration: Route by path: /api/users/* → User Service (3 instances), /api/posts/* → Post Service (5 instances), /api/comments/* → Comment Service (2 instances). SSL termination: HTTPS from client → HTTP to backends. Health Check: HTTP GET /health per service. A/B Testing Rule: If cookie: experiment=new → Route to new version, Else → Route to stable version. SCENARIO 3: HYBRID APPROACH: Real-world: High-traffic e-commerce (Amazon-scale). Architecture: Layer 4 LB (entry point) → Layer 7 LBs (routing) → Services. Why hybrid: (1) Layer 4 handles massive initial traffic (millions of req/sec). (2) Layer 4 distributes load across multiple Layer 7 LBs. (3) Layer 7 LBs do intelligent routing to microservices. Benefits: Best of both: L4 speed + L7 intelligence. Scaling: Can add more L7 LBs behind L4 as traffic grows. Flow: Client → L4 LB (based on IP) → One of 10 L7 LBs → L7 routes by path to appropriate microservice. PERFORMANCE COMPARISON: Throughput: L4: Millions of req/sec, L7: Hundreds of thousands req/sec. Latency: L4: <1ms overhead, L7: 1-5ms overhead (HTTP parsing). For most applications: 1-5ms is acceptable for the intelligence gained. Only use L4 when: (1) Non-HTTP protocol (databases, custom TCP), (2) Extreme performance needs (> 500K req/sec), (3) No need for content-based routing. COST CONSIDERATION: AWS pricing: NLB (L4): $0.0225/hour + $0.006/GB processed. ALB (L7): $0.0225/hour + $0.008/GB processed + $0.008 per LCU (Load Balancer Capacity Unit). L7 slightly more expensive but worth it for microservices. RECOMMENDATION: Default to Layer 7 (ALB/NGINX) for: Web applications, APIs, Microservices, When you need SSL termination, Content-based routing. Use Layer 4 (NLB/HAProxy TCP) for: Non-HTTP protocols, Extreme performance requirements, Simple IP-based distribution, Database load balancing. REAL-WORLD EXAMPLE: Company: Netflix. Layer 4 (Zuul 1): Initially used for API gateway. Layer 7 (Zuul 2): Migrated to for: Path-based routing to microservices, Request authentication, Rate limiting, Much more flexibility despite slight performance cost.",
          keyPoints: [
            'Layer 4: Fast, TCP/UDP routing, protocol-agnostic, use for databases or extreme performance',
            'Layer 7: Intelligent, HTTP-based routing, SSL termination, use for microservices',
            'Layer 7 can route by URL path (/api/users vs /api/posts)',
            'Layer 4 throughput: millions req/sec, Layer 7: hundreds of thousands',
            'Default to Layer 7 for web apps/APIs, use Layer 4 for non-HTTP or extreme scale',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'Your application has 10 servers behind a Round Robin load balancer. 8 servers have 16GB RAM, 2 servers have 64GB RAM. What problem will you likely encounter?',
          options: [
            'All servers will be equally loaded',
            'The 2 powerful servers will be underutilized, 8 weaker servers may be overloaded',
            'Round Robin will automatically detect server capacity and adjust',
            'This configuration will work optimally',
          ],
          correctAnswer: 1,
          explanation:
            'Round Robin distributes requests equally (10% to each server) regardless of server capacity. The 2 servers with 64GB RAM will handle 10% of traffic each (underutilized), while the 8 servers with 16GB RAM also get 10% each (may be overloaded if requests are memory-intensive). Solution: Use Weighted Round Robin with higher weights for powerful servers, or Least Connections/Least Response Time to adapt to actual load.',
        },
        {
          id: 'mc2',
          question:
            'Which load balancing algorithm is BEST for WebSocket connections (long-lived, persistent connections)?',
          options: ['Round Robin', 'Random', 'Least Connections', 'IP Hash'],
          correctAnswer: 2,
          explanation:
            "Least Connections is best for WebSocket connections because: (1) WebSockets are long-lived (can last hours). (2) Round Robin doesn't account for existing connections - Server A might have 100 active WebSocket connections while Server B has 10, but Round Robin still sends requests equally. (3) Least Connections routes new connections to the server with fewest active connections, balancing load based on actual work. IP Hash could work but doesn't balance load well if users are unevenly distributed.",
        },
        {
          id: 'mc3',
          question:
            "Your users complain they're randomly logged out. You discover the load balancer uses Round Robin and sessions are stored in server memory. What is the recommended solution?",
          options: [
            'Switch to IP Hash for sticky sessions',
            'Use cookie-based sticky sessions',
            'Store sessions in Redis (shared storage) and keep any load balancing algorithm',
            'Increase server memory',
          ],
          correctAnswer: 2,
          explanation:
            "Storing sessions in Redis (shared storage) is the recommended solution because: (1) Any server can retrieve any session (truly stateless). (2) No data loss if server crashes. (3) Easy to scale (add/remove servers). (4) Doesn't rely on sticky sessions which have problems (server failure, uneven load). Options 1 and 2 (sticky sessions) are quick fixes but problematic: server crash loses all sessions, prevents true horizontal scaling, uneven load distribution. This is standard practice in distributed systems.",
        },
        {
          id: 'mc4',
          question:
            'What is the main advantage of Layer 7 load balancing over Layer 4?',
          options: [
            'Layer 7 is faster and has higher throughput',
            'Layer 7 can route based on HTTP content (URL path, headers, cookies)',
            'Layer 7 works with any protocol (TCP, UDP, HTTP)',
            'Layer 7 is simpler to configure',
          ],
          correctAnswer: 1,
          explanation:
            "Layer 7 can route based on HTTP content: URL paths (/api/users → User Service, /api/posts → Post Service), headers (X-Mobile: true → mobile backend), cookies (A/B testing). Layer 4 only sees IP addresses and ports, cannot inspect HTTP content. Trade-off: Layer 7 is SLOWER (must parse HTTP) but much more intelligent. Layer 4 is faster but can't do content-based routing. For microservices architecture, Layer 7's content-based routing is essential despite performance cost.",
        },
        {
          id: 'mc5',
          question:
            'Your load balancer health check is set to check every 5 seconds with unhealthy threshold of 3. A server crashes. How long until the load balancer stops sending traffic to it?',
          options: ['5 seconds', '10 seconds', '15 seconds', 'Immediately'],
          correctAnswer: 2,
          explanation:
            '15 seconds. Calculation: (1) Health check every 5 seconds. (2) Unhealthy threshold = 3 (need 3 consecutive failures). (3) Time = 5 seconds × 3 = 15 seconds. So 15 seconds pass before server marked unhealthy and removed from rotation. During these 15 seconds, some user requests will still go to the crashed server (failed requests). To reduce this window: Decrease health check interval (e.g., every 2 seconds → 6 second detection time). But: More frequent health checks = more network overhead. Trade-off between fast failure detection and overhead.',
        },
      ],
    },
    {
      id: 'caching',
      title: 'Caching',
      content: `Caching stores frequently accessed data in a fast storage layer to reduce latency, database load, and improve system performance.

## What is Caching ?

** Definition **: A cache is a high - speed data storage layer that stores a subset of data so future requests for that data are served faster.

### ** Why Caching Matters **

** Without Cache:**
    - Every request hits the database(slow)
        - Database becomes bottleneck at scale
            - High latency for users
                - Expensive(database costs scale with requests)

** With Cache:**
    - Most requests served from memory(fast)
        - Database load reduced by 80 - 95 %
            - Lower latency(milliseconds vs hundreds of milliseconds)
                - Cost savings(cache cheaper than database scaling)

                    ** Real - world impact:**
- ** Reddit **: 99 % cache hit rate on homepage
    - ** Twitter **: Caches timelines to serve billions of requests
        - ** Netflix **: Caches video metadata and recommendations

---

## Cache Hit vs Cache Miss

### ** Cache Hit **

    Request data exists in cache → return immediately from cache.

** Example:**
    1. User requests profile for user ID 123
2. Check cache: Found! → Return from cache(2ms)
3. No database query needed

---

### ** Cache Miss **

    Request data NOT in cache → fetch from database, store in cache, return to user.

** Example:**
    1. User requests profile for user ID 456
2. Check cache: Not found(miss)
3. Query database(50ms)
4. Store result in cache
5. Return to user

    ** Subsequent requests for user ID 456 will be cache hits! **

        ---

### ** Cache Hit Rate **

** Formula:** \`Cache Hit Rate = Hits / (Hits + Misses)\`

    ** Example:**
        - 90 requests served from cache(hits)
            - 10 requests went to database(misses)
                - Hit rate: 90 / (90 + 10) = 90 %

** Target hit rates:**
- ** Good **: 80 - 90 %
- ** Excellent **: 95 %+
- ** Needs tuning **: <80%

** Higher hit rate = better performance and lower cost **

    ---

## Where to Place Cache

### ** 1. Client - Side Cache(Browser) **

** Location **: User's browser

    ** What's cached**: HTML pages, CSS, JavaScript, images

        ** Example **: HTTP cache headers
            - \`Cache - Control: max - age=3600\`(cache for 1 hour)
    - \`ETag\` for validation

        ** Pros:**
            - Fastest(no network request)
            - Reduces server load
                - Works offline

                    ** Cons:**
                        - No control after deployment(can't invalidate)
                            - Limited storage
                        - User can clear cache

---

### ** 2. CDN Cache(Edge) **

** Location **: Geographically distributed edge servers(CloudFront, Akamai)

                        ** What's cached**: Static assets (images, videos, CSS, JS)

                        ** Example **: User in Japan requests image
                        - First request: Fetches from US origin server(200ms)
                        - CDN caches at Tokyo edge location
                        - Subsequent requests: Served from Tokyo(20ms)

                        ** Pros:**
                        - Reduced latency(geographically close to users)
                        - Offloads origin server
                        - Scales globally

                        ** Cons:**
                        - Only for static / semi - static content
                            - CDN costs
    - Invalidation complexity

---

### ** 3. Application Cache(In - Memory) **

** Location **: Application server's memory

    ** What's cached**: Application-level data (local to each server)

        ** Example **: Each API server caches configuration in memory

            ** Pros:**
                - Extremely fast(no network call)
                    - Simple to implement

                        ** Cons:**
                            - Cache inconsistency across servers
                                - Lost on server restart
                                    - Limited by server memory

                                        ** Use case:** Small, read - only data that rarely changes(config, feature flags)

---

### ** 4. Distributed Cache(Redis, Memcached) **

** Location **: Dedicated cache cluster, separate from app servers

    ** What's cached**: Frequently accessed data (user sessions, API responses, database queries)

        ** Example:** Redis cluster shared by all API servers

            ** Pros:**
                - Shared across all app servers(consistent)
                    - Survives app server restarts
                        - Scales independently
                            - Rich features(TTL, data structures)

                                ** Cons:**
                                    - Network latency(1 - 2ms)
                                        - Additional infrastructure to manage
                                            - Cache becomes dependency

                                                ** This is the most common caching strategy for distributed systems! **

                                                    ---

### ** 5. Database Cache(Query Result Cache) **

** Location **: Inside database(MySQL query cache, PostgreSQL shared buffers)

    ** What's cached**: Query results

        ** Example **: MySQL caches\`SELECT * FROM users WHERE id = 123\`

            ** Pros:**
                - Transparent(application doesn't need to manage)
                    - Works automatically

                ** Cons:**
                - Limited control
                - Invalidation can be tricky
                - Often disabled in modern databases(MySQL 8.0 removed query cache)

                ** Recommendation **: Use application - level cache(Redis) instead for better control.

---

## Cache Reading Patterns

### ** 1. Cache - Aside(Lazy Loading) **

** Most common pattern.**

** Flow:**
    1. Application checks cache for data
2. If ** cache hit **: Return data from cache
3. If ** cache miss **:
- Query database
    - Store result in cache
        - Return data

            ** Pseudocode:**
\`\`\`python
def get_user(user_id):
    # Check cache first
    user = cache.get(f"user:{user_id}")
    if user:
        return user  # Cache hit
    
    # Cache miss: query database
    user = database.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache for next time
    cache.set(f"user:{user_id}", user, ttl=3600)  # Cache for 1 hour
    
    return user
\`\`\`

**Pros:**
- Only caches data that's actually requested (efficient)
- Cache failures don't break system (degrades to database)
- Simple to implement

**Cons:**
- Cache miss penalty (3 operations: check cache, query DB, write cache)
- Initial requests always miss (cold start)
- Can have stale data if not invalidated properly

**When to use:** Most general-purpose caching scenarios.

---

### **2. Read-Through Cache**

**Flow:**
1. Application requests data from cache
2. Cache library handles database query if miss
3. Cache automatically loads and returns data

**Difference from Cache-Aside:** Cache library manages database interaction (not application).

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    # Cache library handles everything
    return cache.get_or_load(f"user:{user_id}")
    
# Cache library internally:
# - Checks cache
# - If miss, calls registered loader function
# - Stores result in cache
# - Returns data
\`\`\`

**Pros:**
- Cleaner application code
- Consistent caching logic

**Cons:**
- Tighter coupling (cache must know about database)
- Less flexibility

**When to use:** When cache library supports it (e.g., Rails caching, some Java frameworks).

---

## Cache Writing Patterns

### **1. Write-Through Cache**

**Flow:**
1. Application writes data to cache
2. Cache **synchronously** writes to database
3. Return success only after database write succeeds

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write to cache, which writes through to database
    cache.set(f"user:{user_id}", new_data)  # Internally writes to DB
    return success
\`\`\`

**Pros:**
- Data consistency (cache and DB always in sync)
- Cache is always up-to-date
- No stale data

**Cons:**
- Slower writes (every write hits database)
- Write latency = cache write + database write
- Wasted cache space (might cache data never read)

**When to use:** When consistency is critical and stale data is unacceptable.

---

### **2. Write-Back (Write-Behind) Cache**

**Flow:**
1. Application writes data to cache
2. Return success immediately
3. Cache **asynchronously** writes to database later (batched)

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write to cache only
    cache.set(f"user:{user_id}", new_data)  # Instant return
    # Cache will flush to database periodically
    return success
\`\`\`

**Pros:**
- Fast writes (only to memory)
- Can batch multiple writes to database (more efficient)
- Reduces database write load

**Cons:**
- Risk of data loss (if cache crashes before flushing to database)
- Complex to implement
- Eventual consistency

**When to use:** Write-heavy workloads where some data loss is acceptable (e.g., view counts, analytics).

---

### **3. Write-Around Cache**

**Flow:**
1. Application writes directly to database
2. Cache is bypassed on write
3. Data loaded into cache only when read (cache-aside pattern)

**Pseudocode:**
\`\`\`python
def update_user(user_id, new_data):
    # Write directly to database
    database.update("UPDATE users SET ... WHERE id = ?", user_id, new_data)
    
    # Optionally invalidate cache
    cache.delete(f"user:{user_id}")
    
    return success
\`\`\`

**Pros:**
- Avoids cache pollution (don't cache data that won't be read)
- Simpler than write-through

**Cons:**
- Cache miss on first read after write
- Stale data possible if not invalidated

**When to use:** Write-once, read-rarely data (e.g., logs, historical data).

---

## Cache Eviction Policies

**Problem**: Cache has limited memory. When full, which items should be removed?

### **1. Least Recently Used (LRU)**

**Policy**: Evict the item that hasn't been accessed for the longest time.

**Example:**
- Cache holds: A (accessed 10 min ago), B (accessed 2 min ago), C (accessed 5 min ago)
- Cache full, need to evict
- **Evict A** (least recently used)

**Pros:**
- Good for most workloads
- Keeps hot data in cache
- Simple to understand

**Cons:**
- Doesn't account for access frequency
- Can evict items that are frequently accessed but haven't been accessed recently

**Implementation:** Doubly linked list + hash map

**This is the most common eviction policy!**

---

### **2. Least Frequently Used (LFU)**

**Policy**: Evict the item accessed the fewest times.

**Example:**
- A: 100 accesses
- B: 5 accesses
- C: 50 accesses
- **Evict B** (least frequently used)

**Pros:**
- Keeps truly hot data (frequently accessed)

**Cons:**
- Old popular items stay forever (accessed 1000× last year, 0× this year)
- Complexity

**When to use:** When access frequency more important than recency (recommendation systems).

---

### **3. First In First Out (FIFO)**

**Policy**: Evict oldest item, regardless of access patterns.

**Example:**
- Items added: A, B, C
- **Evict A** (first in)

**Pros:**
- Simple

**Cons:**
- Ignores access patterns
- Poor performance

**Rarely used in practice.**

---

### **4. Time To Live (TTL)**

**Policy**: Evict items after a specified time, regardless of access.

**Example:**
- Cache user profile for 1 hour
- After 1 hour, automatically evicted (even if accessed frequently)

**Pros:**
- Guarantees data freshness
- Simple

**Cons:**
- Might evict hot data
- Cache misses when TTL expires

**Common practice:** Combine TTL with LRU (both limits).

---

### **5. Random Replacement**

**Policy**: Evict a random item.

**Pros:**
- Simple, fast

**Cons:**
- Unpredictable
- Might evict hot data

**When to use:** When simplicity > performance, or when access patterns are truly random.

---

## Cache Invalidation

**Two hard problems in computer science: cache invalidation, naming things, and off-by-one errors.**

**Challenge**: How to ensure cache stays consistent with database?

### **Strategy 1: Time-Based (TTL)**

Set expiration time on cached data.

**Example:**
\`\`\`python
cache.set("user:123", user_data, ttl=3600)  # Expire after 1 hour
\`\`\`

**Pros:**
- Simple
- Automatic

**Cons:**
- Stale data until TTL expires
- Too short TTL = frequent cache misses
- Too long TTL = stale data

**When to use:** Data that changes infrequently, stale data acceptable.

---

### **Strategy 2: Explicit Invalidation**

Delete from cache when data changes.

**Example:**
\`\`\`python
def update_user(user_id, new_data):
    # Update database
    database.update(user_id, new_data)
    
    # Invalidate cache
    cache.delete(f"user:{user_id}")
\`\`\`

**Pros:**
- No stale data (cache always fresh)

**Cons:**
- Must invalidate in every write path
- Easy to miss a write path (leads to stale data)
- Race conditions possible

**When to use:** When data consistency is critical.

---

### **Strategy 3: Write-Through**

Update cache and database together (discussed above).

**Pros:**
- Cache always consistent

**Cons:**
- Slower writes

---

### **Strategy 4: Event-Based Invalidation**

Publish events when data changes; cache listeners invalidate.

**Example:**
1. User service updates user → publishes event: \`user.updated: 123\`
2. Cache service subscribes to events → invalidates \`user: 123\`

**Pros:**
- Decoupled (services don't need to know about cache)
- Scalable

**Cons:**
- Complexity (need event system)
- Eventual consistency

**When to use:** Large systems with many services.

---

## Cache Stampede Problem

**Problem**: Cache expires, many requests simultaneously hit database (thundering herd).

**Scenario:**
1. Cache for popular item expires
2. 10,000 concurrent requests arrive
3. All 10,000 requests miss cache
4. All 10,000 query database simultaneously
5. Database overwhelmed!

**Solution 1: Locking**

First request acquires lock, queries database, updates cache. Other requests wait for cache to be populated.

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    user = cache.get(f"user:{user_id}")
    if user:
        return user
    
    # Acquire lock
    with cache.lock(f"lock:user:{user_id}", timeout=5):
        # Double-check cache (might have been populated while waiting for lock)
        user = cache.get(f"user:{user_id}")
        if user:
            return user
        
        # Query database
        user = database.query(user_id)
        cache.set(f"user:{user_id}", user, ttl=3600)
        return user
\`\`\`

---

**Solution 2: Probabilistic Early Expiration**

Randomly refresh cache before TTL expires.

**Pseudocode:**
\`\`\`python
def get_user(user_id):
    user, ttl_remaining = cache.get_with_ttl(f"user:{user_id}")
    
    if user and ttl_remaining > 0:
        # Probabilistically refresh early
        if random.random() < (1.0 / ttl_remaining):
            # Asynchronously refresh
            background_task.enqueue(refresh_user_cache, user_id)
        return user
    
    # Cache miss: load and cache
    user = database.query(user_id)
    cache.set(f"user:{user_id}", user, ttl=3600)
    return user
\`\`\`

---

## Cache Consistency Models

### **Strong Consistency**

Cache always reflects latest database state.

**Implementation**: Write-through cache or synchronous invalidation.

**Use case**: Financial data, inventory counts (stale data unacceptable).

---

### **Eventual Consistency**

Cache may be temporarily stale but eventually consistent.

**Implementation**: TTL or asynchronous invalidation.

**Use case**: Social media (OK if user sees profile updated 5 seconds later).

---

## Distributed Caching (Redis Example)

**Redis**: In-memory data structure store, commonly used as cache.

### **Key Features:**

**1. Data Structures:**
- Strings (most common)
- Hashes (store objects)
- Lists (queues)
- Sets (unique items)
- Sorted Sets (leaderboards)

**2. Persistence:**
- RDB snapshots (periodic saves to disk)
- AOF (append-only file, logs every write)

**3. Replication:**
- Master-slave replication
- Read replicas for scaling reads

**4. Clustering:**
- Sharding across multiple nodes
- Automatic failover

---

### **Example: Caching User Profile**

\`\`\`python
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_user(user_id):
    # Try cache first
    cached = r.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss: query database
    user = database.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Store in cache (expire after 1 hour)
    r.setex(f"user:{user_id}", 3600, json.dumps(user))
    
    return user

def update_user(user_id, new_data):
    # Update database
    database.update(user_id, new_data)
    
    # Invalidate cache
    r.delete(f"user:{user_id}")
\`\`\`

---

## Interview Tips

### **Common Questions:**

**Q: "How would you reduce database load in your system?"**

✅ Good answer: "Add caching layer (Redis):
1. Cache-aside pattern for reads
2. Invalidate on writes
3. TTL of 1 hour
4. Expected 90% cache hit rate
5. Reduces database load by 90%"

**Q: "What happens if the cache goes down?"**

✅ Good answer: "Two approaches:
1. **Graceful degradation**: Application falls back to database (slower but functional)
2. **High availability**: Redis Sentinel or Redis Cluster for automatic failover"

**Q: "How do you decide what to cache?"**

✅ Good answer: "Cache data that is:
1. **Read-heavy**: Read 100×more than written
2. **Expensive to compute**: Complex queries, API calls
3. **Frequently accessed**: Hot data (80/20 rule: 20% of data accounts for 80% of requests)
4. **Tolerate staleness**: Eventual consistency OK"

---

## Key Takeaways

1. **Caching dramatically improves performance**: 2ms cache vs 50ms database
2. **Cache-aside most common**: Check cache first, query DB on miss, populate cache
3. **Invalidation is hard**: Use TTL + explicit invalidation for consistency
4. **Distributed cache (Redis) standard**: Shared across app servers, survives restarts
5. **LRU most common eviction**: Evict least recently used items when cache full
6. **Target 80-90% cache hit rate**: Monitor and optimize
7. **Cache stampede prevention**: Use locking or probabilistic early expiration
8. **High availability**: Redis Sentinel/Cluster for cache reliability`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your application has 100K QPS and a 90% cache hit rate. You notice occasional database overload spikes. After investigation, you discover popular items expire simultaneously at midnight (TTL set at deployment time). Explain the problem and propose a solution.',
          sampleAnswer:
            'PROBLEM: CACHE STAMPEDE (THUNDERING HERD): What\'s happening: (1) All cached items deployed at same time with same TTL (e.g., 24 hours). (2) All items expire simultaneously at midnight. (3) At midnight: 100K QPS × 10% miss rate = 10K QPS normally hitting database. (4) When all cache expires: 100K QPS ALL hit database simultaneously. (5) Database overwhelmed (designed for 10K QPS, now receiving 100K QPS). (6) Queries slow down or timeout. (7) More cache misses (slow queries timeout), making problem worse. This is a cache stampede/thundering herd problem. SOLUTION 1: JITTERED TTL (Recommended): Instead of fixed TTL, add randomness. Implementation: BASE_TTL = 3600 seconds (1 hour). JITTER = random(0, 600) seconds (0-10 minutes). ACTUAL_TTL = BASE_TTL + JITTER. Code: cache.set("user:123", user_data, ttl=3600 + random.randint(0, 600)). Result: Items expire gradually over a 10-minute window, not all at once. Impact: Database load spreads out: Instead of 100K QPS spike at one moment, Spread over 10 minutes: Average ~16K QPS (manageable). Pros: Simple to implement (one line change). No coordination needed. Works for all cache keys. Cons: Items still expire (cache misses during the window). SOLUTION 2: PROBABILISTIC EARLY REFRESH: Refresh cache before it expires, with probability increasing as TTL approaches zero. Implementation: When serving from cache, calculate: time_until_expiry = ttl_remaining. probability_refresh = 1 / time_until_expiry. If random() < probability_refresh: Asynchronously refresh cache in background. Result: Cache refreshed before expiry, no mass expiration. Code example: def get_user(user_id): user, ttl_remaining = cache.get_with_ttl(f"user:{user_id}"). if user and ttl_remaining > 0: if random.random() < (1.0 / ttl_remaining): background_job.enqueue(refresh_user, user_id). return user. else: # Cache miss: load from database. Pros: Proactive (prevents stampede). Cache stays warm. Cons: More complex. Background job infrastructure needed. SOLUTION 3: CACHE LOCKING: When cache miss occurs, first request acquires lock and queries database. Other requests wait for cache to be populated. Implementation: Check cache. If miss: Try to acquire lock. If lock acquired: Query database, update cache, release lock. If lock not acquired: Wait briefly, retry reading cache. Result: Only ONE request hits database per expired key. Pros: Minimizes database load (only 1 query per key). Cons: Lock contention. Increased latency for waiting requests. Lock can become bottleneck. SOLUTION 4: BACKGROUND REFRESH: Separate background job refreshes cache before expiration. Implementation: When setting cache, also schedule background job to run before TTL expires. Job re-queries database and updates cache. Pros: Zero cache misses (proactive refresh). Predictable database load. Cons: Complexity (job scheduler needed). Might refresh data that\'s no longer accessed. RECOMMENDED APPROACH: Combine Solution 1 (Jittered TTL) + Solution 2 (Probabilistic Early Refresh): Jittered TTL: Spreads out expirations. Probabilistic refresh: Hot items stay cached. Implementation: cache.set(key, value, ttl=BASE_TTL + random.randint(0, JITTER)). In read path, probabilistically refresh. BEFORE vs AFTER: Before: Database load at midnight: 100K QPS spike (database crashes). After: Database load at midnight: Gradual increase from 10K to ~20K QPS over 10 minutes (smooth, manageable). MONITORING: Track: Cache expiration events per minute (should be smooth, not spiky). Database QPS over time (should be relatively flat). Cache hit rate (should stay high, ~90%). Alert if: Sudden drop in cache hit rate. Database QPS spike >2× normal. REAL-WORLD EXAMPLE: Twitter: Doesn\'t use fixed TTL, uses probabilistic early refresh. Ensures timeline cache always warm. Database load is predictable.',
          keyPoints: [
            'Simultaneous cache expiration causes database overload spike (cache stampede)',
            'Solution 1: Jittered TTL (add randomness to TTL to spread expirations)',
            'Solution 2: Probabilistic early refresh (refresh cache before expiry)',
            'Combine both: Jittered TTL + early refresh for hot items',
            'Result: Smooth database load instead of spikes',
          ],
        },
        {
          id: 'q2',
          question:
            'You\'re deciding whether to use write-through or write-around caching for a social media "like" counter. Discuss the trade-offs and recommend an approach.',
          sampleAnswer:
            'SCENARIO ANALYSIS: "Like" Counter Characteristics: (1) Write-heavy: Users frequently like posts (high write frequency). (2) Read-heavy: Like counts displayed everywhere (even more reads). (3) Eventual consistency OK: Seeing 1,234 vs 1,235 likes doesn\'t matter. (4) Stale data acceptable: Count can be few seconds old. (5) Aggregate data: Total count, not individual likes. OPTION 1: WRITE-THROUGH CACHE: Flow: User likes post → Update cache → Cache updates database. Pros: Cache always up-to-date. Reads always fast (always in cache). Consistency guaranteed. Cons: Every like hits database (slow, expensive). Database write bottleneck. Wasted effort (updating cache and DB immediately). Analysis for Like Counter: Not ideal because: (1) Likes are write-heavy: Every like = database write (expensive at scale). (2) Perfect consistency not needed: OK if count is slightly stale. (3) Database becomes bottleneck: 10K likes/sec = 10K DB writes/sec. OPTION 2: WRITE-AROUND CACHE: Flow: User likes post → Write directly to database, invalidate cache. Next read: Cache miss, fetch from database, populate cache. Pros: Simple. Avoids cache pollution (if data won\'t be read). Cons: Cache miss after every write (read penalty). For frequently read data (like likes), this is bad. Analysis for Like Counter: Not ideal because: (1) Likes are read-heavy: Invalidating cache causes cache miss on next read. (2) High read volume: 100K reads/sec → frequent cache misses → database overload. RECOMMENDED: WRITE-BACK (WRITE-BEHIND) CACHE: Flow: User likes post → Increment counter in cache → Return success immediately. Background job: Periodically flush cached counts to database (batched). Pros: (1) Instant writes: Increment in-memory counter (< 1ms). (2) Batched database writes: Instead of 10K individual writes, batch into 1 write per post every 5 seconds. (3) Reduces database load: 10K likes/sec → 200 DB writes/sec (50× reduction). (4) Scales well: Cache handles write volume. Cons: (1) Risk of data loss: If cache crashes before flushing, some likes lost. (2) Complexity: Need background job to flush. (3) Eventual consistency: Database lags behind cache. Mitigation for Data Loss: (1) Acceptable for likes: Losing few likes during crash is OK (not financial data). (2) Use persistent cache: Redis with AOF (append-only file) logs every increment. (3) Frequent flushes: Flush every 5-10 seconds (minimize data loss window). IMPLEMENTATION DETAILS: Redis: Use INCR command (atomic, fast). cache.incr(f"post:123:likes")  # Atomic increment. Background Job (Celery/Sidekiq): Every 5 seconds: (1) Fetch all dirty counters from cache. (2) Batch update database: UPDATE posts SET likes = likes + delta WHERE id IN (...). (3) Mark counters as flushed in cache. Read Path: Check cache first: likes = cache.get(f"post:123:likes"). If cache miss: Query database, populate cache. Result: Reads almost always cache hit. Writes instant (in-memory). Database handles 50× less load. ALTERNATIVE: HYBRID APPROACH (Write-Through + Batch): Flow: User likes post → Increment cache immediately → Return success. Every N likes (e.g., every 10 likes) OR every T seconds → Update database. Example: Like count in cache: 1, 2, 3, ..., 10 → Database write (batch 10 likes). Pros: Balances immediate writes with batching. Reduces database load. Limits data loss (at most N likes lost). Cons: Still complexity of batching logic. COMPARISON TABLE: Write-Through: Database Load: High (every write), Read Performance: Excellent, Consistency: Strong, Data Loss Risk: None, Complexity: Low. Write-Around: Database Load: High (every write), Read Performance: Poor (frequent misses), Consistency: Strong, Data Loss Risk: None, Complexity: Low. Write-Back: Database Load: Low (batched), Read Performance: Excellent, Consistency: Eventual, Data Loss Risk: Small (mitigated), Complexity: Medium. REAL-WORLD EXAMPLES: Facebook Likes: Use write-back with batching. Redis counters flushed periodically. Acceptable to lose few likes in crash. Twitter Favorites: Similar approach, batched writes. YouTube View Counts: Write-back, approximate counts (eventual consistency fine). FINAL RECOMMENDATION: Use Write-Back (Write-Behind) for like counters: (1) Increment cache immediately (fast user experience). (2) Batch writes to database every 5-10 seconds. (3) Use Redis with persistence (AOF). (4) Monitor cache health and flush frequency. Result: 50× reduction in database writes, instant user experience, scalable to billions of likes. Trade-off accepted: Small risk of data loss (mitigated) + eventual consistency (acceptable for likes).',
          keyPoints: [
            'Like counters: write-heavy, read-heavy, eventual consistency OK',
            "Write-through: every like hits database (doesn't scale)",
            'Write-around: invalidates cache on write (causes read misses)',
            'Recommended: Write-back (increment cache, batch flush to DB)',
            'Result: 50× less database load, instant writes, small data loss risk (acceptable)',
          ],
        },
        {
          id: 'q3',
          question:
            'Design a caching strategy for an e-commerce product catalog with 1M products. 80% of traffic goes to 1000 popular products (0.1% of catalog). How would you optimize cache size and hit rate?',
          sampleAnswer:
            'SCENARIO ANALYSIS: Key Facts: (1) 1M total products. (2) 80% traffic to 1000 products (hot items). (3) 20% traffic to 999,000 products (long tail). (4) Heavy skew (Pareto principle: 80/20 rule). Goal: Maximize cache hit rate while minimizing cache size. NAIVE APPROACH: Cache all 1M products. Assuming 10KB per product: 1M × 10KB = 10GB cache. Problem: Wastes memory. 999,000 products rarely accessed (cache pollution). Expensive (Redis memory costs). OPTIMIZED APPROACH: Cache based on access patterns. TARGET CACHE SIZE: Cache top 100K products (10% of catalog). 100K × 10KB = 1GB cache. Much more cost-effective! EXPECTED CACHE HIT RATE: Top 1000 products: 80% of traffic (always cached). Next 99,000 products: Some frequently accessed, some not. Estimate: 15% of remaining 20% traffic cached. Total hit rate: 80% + (20% × 0.75) = 95% hit rate. Result: 1GB cache achieves 95% hit rate (vs 10GB for 100% hit rate). CACHING STRATEGY: Combination of techniques: TECHNIQUE 1: LRU EVICTION: Use LRU (Least Recently Used) eviction policy. Hot products stay in cache. Cold products evicted automatically. Configuration: Max cache size: 1GB. Eviction policy: LRU. Result: Top 1000 products always cached. Frequently accessed long-tail products cached temporarily. TECHNIQUE 2: TIERED CACHING: Two-tier cache: (1) Tier 1 (Hot): 1000 products, never expire, always in memory. (2) Tier 2 (Warm): 99,000 products, LRU eviction, TTL 1 hour. Implementation: def get_product(product_id): # Tier 1: Hot products (no TTL). product = cache.get(f"product:hot:{product_id}"). if product: return product. # Tier 2: Warm products (1 hour TTL). product = cache.get(f"product:warm:{product_id}"). if product: return product. # Cache miss: query database. product = database.query(product_id). # Determine tier based on popularity. if product.popularity > THRESHOLD: cache.set(f"product:hot:{product_id}", product)  # No TTL. else: cache.set(f"product:warm:{product_id}", product, ttl=3600)  # 1 hour. return product. Pros: Hot products never evicted. Optimal memory usage. Cons: Need to track product popularity. TECHNIQUE 3: POPULARITY-BASED TTL: Hot products: Longer TTL (e.g., 24 hours). Cold products: Shorter TTL (e.g., 1 hour). Implementation: def get_cache_ttl(product): if product.views_last_week > 10000: return 86400  # 24 hours. elif product.views_last_week > 1000: return 3600  # 1 hour. else: return 600  # 10 minutes. Pros: Adaptive (hot products stay cached longer). Simple to implement. TECHNIQUE 4: PROACTIVE CACHE WARMING: On application startup or deployment: Pre-populate cache with top 1000 products. Background job: Every hour: (1) Query analytics for top products. (2) Refresh cache for these products. Pros: Top products always cached (zero cold starts). Predictable cache hit rate. Cons: Requires analytics infrastructure. TECHNIQUE 5: CACHE ASIDE WITH POPULARITY TRACKING: Track access count in Redis. Increment on each access: cache.incr(f"product:access_count:{product_id}"). Periodically analyze: Identify top N products. Ensure they\'re always cached. CODE EXAMPLE (CACHE ASIDE + LRU): def get_product(product_id): # Check cache. product = cache.get(f"product:{product_id}"). if product: # Track access (for popularity analysis). cache.incr(f"access:product:{product_id}"). return product. # Cache miss: query database. product = database.query(product_id). # Cache with TTL based on popularity. ttl = get_ttl_for_product(product). cache.set(f"product:{product_id}", product, ttl=ttl). return product. MONITORING & OPTIMIZATION: Metrics to track: (1) Cache hit rate (target: >95%). (2) Cache size (should stabilize around 1GB). (3) Eviction rate (how often items evicted). (4) Top 100 products by access count. Optimization: If hit rate < 95%: Increase cache size (e.g., 2GB). If cache consistently full: Analyze eviction patterns, adjust TTL. Weekly analysis: Refresh list of top 1000 products, proactively cache them. CACHE INVALIDATION: Product updates (price, description): Invalidate cache immediately: cache.delete(f"product:{product_id}"). Product deletion: Invalidate cache. Bulk updates: Flush cache keys matching pattern (carefully!). SCALABILITY: If catalog grows to 10M products: Same strategy scales: Cache top 100K (1% of catalog), Still achieve 95%+ hit rate, 1-2GB cache. Redis Cluster: Shard across multiple Redis nodes if needed. COST ANALYSIS: Naive approach: 10GB Redis cache: ~$200/month. Optimized approach: 1GB Redis cache: ~$20/month. Savings: $180/month (90% cost reduction). Performance: Similar (95% hit rate). REAL-WORLD EXAMPLES: Amazon: Doesn\'t cache all products. Caches bestsellers and recently viewed. Uses tiered caching (hot/warm/cold). Netflix: Doesn\'t cache all videos. Caches popular titles and personalized recommendations. Uses popularity-based TTL. FINAL RECOMMENDATION: Use LRU eviction with 1-2GB cache. Proactively warm cache with top 1000 products. Popularity-based TTL (hot = 24h, cold = 1h). Monitor hit rate and adjust size if needed. Result: 95% hit rate with 1GB cache (10× smaller than naive approach).',
          keyPoints: [
            'Pareto principle: 80% traffic to 0.1% of products (1000 of 1M)',
            "Don't cache entire catalog (wasteful): cache hot items only",
            'Use LRU eviction: hot products stay, cold products evicted',
            '1GB cache (100K products) achieves 95% hit rate vs 10GB (all products)',
            'Proactively warm cache with top products, use popularity-based TTL',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'Your application has 10K QPS with a 90% cache hit rate. The cache goes down. What happens to your database?',
          options: [
            'Database receives 1K QPS (same as before)',
            'Database receives 9K QPS (only the cache hits)',
            'Database receives 10K QPS (all traffic)',
            'Database receives 5K QPS (system automatically throttles)',
          ],
          correctAnswer: 2,
          explanation:
            'Database receives 10K QPS (all traffic). Before: 90% cache hit rate meant 9K QPS served by cache, 1K QPS hit database. After cache down: 100% cache miss rate, all 10K QPS hit database. Result: Database load increases 10× (from 1K to 10K QPS). This can overwhelm the database. Solution: (1) Make cache highly available (Redis Sentinel/Cluster). (2) Implement graceful degradation (rate limiting, fallback responses). (3) Ensure database can handle full load temporarily (over-provision).',
        },
        {
          id: 'mc2',
          question:
            'Which caching pattern should you use for a read-heavy workload where data rarely changes and stale data is acceptable for a few minutes?',
          options: [
            'Write-through cache',
            'Write-back cache',
            'Cache-aside with TTL',
            'Write-around cache',
          ],
          correctAnswer: 2,
          explanation:
            'Cache-aside with TTL is perfect for this scenario. Why: (1) Read-heavy: Cache-aside checks cache first (fast reads). (2) Rarely changes: Low cache miss rate. (3) Stale data OK: TTL of a few minutes is acceptable. Implementation: Check cache first, on miss query database and cache result with TTL (e.g., 300 seconds). Write-through: Overkill (slow writes, unnecessary consistency for this scenario). Write-back: For write-heavy workloads. Write-around: For write-once-read-rarely data.',
        },
        {
          id: 'mc3',
          question:
            'Your cache is 100% full. A new item needs to be cached. The eviction policy is LRU. Which item gets evicted?',
          options: [
            'The oldest item (first added to cache)',
            'The item that has been accessed the fewest times',
            "The item that hasn't been accessed for the longest time",
            'A random item',
          ],
          correctAnswer: 2,
          explanation:
            'LRU (Least Recently Used) evicts the item that hasn\'t been accessed for the longest time. Example: Item A last accessed 10 minutes ago, Item B last accessed 2 minutes ago → Evict A. Option 1 (FIFO): Evicts oldest by insertion time (ignores access patterns). Option 2 (LFU): Evicts least frequently used (access count, not recency). Option 4: Random eviction. LRU is most common because it keeps "hot" (recently accessed) data in cache.',
        },
        {
          id: 'mc4',
          question:
            "You update a user's profile in the database. What should you do with the cached version to avoid serving stale data?",
          options: [
            'Leave it in cache (TTL will eventually expire)',
            'Delete the cache entry immediately',
            'Update both database and cache (write-through)',
            "Do nothing, cache consistency doesn't matter",
          ],
          correctAnswer: 1,
          explanation:
            'Delete the cache entry immediately (cache invalidation). This ensures: (1) Next read will fetch updated data from database. (2) No stale data served. (3) Simple to implement. Option 1 (rely on TTL): Serves stale data until TTL expires (unacceptable for profile updates). Option 3 (write-through): Also valid but requires updating cache logic (delete is simpler). Option 4: Wrong - consistency matters for user-facing data. Best practice: On write, invalidate cache. On read, check cache → miss → query DB → populate cache.',
        },
        {
          id: 'mc5',
          question:
            'What is the main risk of write-back (write-behind) caching?',
          options: [
            'Slow writes (every write hits database)',
            'Stale data in cache',
            'Data loss if cache crashes before flushing to database',
            'High database load',
          ],
          correctAnswer: 2,
          explanation:
            'Data loss if cache crashes before flushing to database. Write-back flow: Write to cache → return success immediately → asynchronously flush to database later. Risk: If cache crashes before flush, data in cache (not yet in database) is lost. Mitigation: (1) Use persistent cache (Redis AOF). (2) Frequent flushes (every few seconds). (3) Accept risk for non-critical data (view counts, likes). Benefits: Fast writes, reduced database load. Trade-off: Small data loss risk vs performance.',
        },
      ],
    },
    {
      id: 'data-partitioning-sharding',
      title: 'Data Partitioning & Sharding',
      content: `Data partitioning (sharding) splits large databases into smaller, faster, more manageable pieces called shards to achieve horizontal scalability.

## What is Data Partitioning?

**Definition**: Splitting a large dataset across multiple databases or servers so that each machine holds only a portion of the data.

### **Why Partition Data?**

**Without Partitioning:**
- Single database has limits (storage, CPU, memory, connections)
- Query performance degrades as data grows
- Single point of failure
- Can't scale beyond one machine's capacity

**Example**: Twitter has billions of tweets. Can't fit all tweets in a single PostgreSQL database.

**With Partitioning:**
- Horizontal scalability (add more machines)
- Better query performance (each shard smaller, queries faster)
- Higher throughput (queries distributed across shards)
- Fault isolation (one shard fails, others still work)

**Real-world scale**: Instagram shards photos across thousands of database servers.

---

## Horizontal vs Vertical Partitioning

### **Vertical Partitioning**

**Definition**: Split table by columns.

**Example**: User table with 50 columns
- Shard 1: \`id, username, email\` (frequently accessed)
- Shard 2: \`profile_description, hobbies, bio\` (rarely accessed)

**Use case**: Separate hot columns (accessed often) from cold columns (accessed rarely).

**Pros:**
- Reduces I/O for queries accessing only hot columns
- Simpler queries (fewer joins)

**Cons:**
- Still limited by single machine (not true horizontal scaling)
- Joins across shards more complex

**This is less common than horizontal partitioning.**

---

### **Horizontal Partitioning (Sharding)**

**Definition**: Split table by rows.

**Example**: User table with 100M users
- Shard 1: Users 1-25M
- Shard 2: Users 25M-50M
- Shard 3: Users 50M-75M
- Shard 4: Users 75M-100M

**Use case**: Scale beyond single machine capacity (most common).

**Pros:**
- True horizontal scalability (add more shards)
- Each shard smaller and faster
- Parallelism (multiple queries across shards simultaneously)

**Cons:**
- Complex queries (cross-shard joins, aggregations)
- Rebalancing when adding shards
- Application logic more complex

**This is what most people mean by "sharding."**

---

## Sharding Strategies

### **1. Range-Based Sharding**

**How it works**: Partition data based on ranges of a key (e.g., user ID, date).

**Example: Shard by User ID**
- Shard 1: User IDs 1-1,000,000
- Shard 2: User IDs 1,000,001-2,000,000
- Shard 3: User IDs 2,000,001-3,000,000

**Example: Shard by Date**
- Shard 1: Orders from 2020
- Shard 2: Orders from 2021
- Shard 3: Orders from 2022
- Shard 4: Orders from 2023

**Pros:**
- Simple to implement
- Range queries efficient (all data in one or few shards)
- Easy to understand

**Cons:**
- **Hotspots**: Uneven distribution if data not uniform
  - Example: New users always go to latest shard (Shard 3 overloaded, Shard 1 idle)
- **Sequential IDs**: If user IDs sequential, recent users on one shard (hotspot)

**When to use**: When data has natural ranges and distribution is even.

---

### **2. Hash-Based Sharding**

**How it works**: Apply hash function to partition key, use result to determine shard.

**Formula**: \`shard_id = hash(key) % num_shards\`

**Example: Shard by User ID**
- User ID 123 → \`hash(123) % 4\` = 3 → Shard 3
- User ID 456 → \`hash(456) % 4\` = 0 → Shard 0
- User ID 789 → \`hash(789) % 4\` = 1 → Shard 1

**Pros:**
- **Uniform distribution**: Hash function distributes data evenly
- **No hotspots**: Random distribution prevents any shard from being overloaded
- Simple to implement

**Cons:**
- **Range queries hard**: Finding users 1-1000 requires querying ALL shards
- **Rebalancing**: Adding/removing shards requires rehashing all data
  - Example: 4 shards → 5 shards changes all \`hash(key) % 4\` to \`hash(key) % 5\`
  - Requires massive data migration

**When to use**: When uniform distribution more important than range queries.

---

### **3. Consistent Hashing**

**How it works**: Hash both data and shards onto a ring (0 to 2^32-1). Data assigned to next shard clockwise on ring.

**Visual**:

**Hash Ring Visualization:**
- Ring ranges from 0 to 2^32-1
- Shards: S1 at 25%, S2 at 50%, S3 at 75%
- User 123: hash(123) = 30% → Next shard clockwise = S2
- User 456: hash(456) = 60% → Next shard clockwise = S3

**Adding a shard**:
- New Shard S4 at 40%
- Only data between 25% and 40% moves (from S2 to S4)
- Other shards unaffected

**Virtual Nodes**: Place each physical shard at multiple positions on ring for better distribution.

**Pros:**
- **Minimal rebalancing**: Adding/removing shards only affects neighboring data
- **Scalable**: Easy to add shards
- **Uniform distribution** with virtual nodes

**Cons:**
- More complex to implement
- Still no efficient range queries

**Use case**: Distributed caches (Redis Cluster), distributed databases (Cassandra, DynamoDB).

**This is the standard for modern distributed systems!**

---

### **4. Directory-Based Sharding**

**How it works**: Maintain a lookup table (directory) mapping keys to shards.

**Example**:
\`\`\`
Directory Table:
User ID Range → Shard
    1 - 1M       → Shard 1
    1M - 2M      → Shard 2
    2M - 3M      → Shard 3
        \`\`\`

**Lookup**:
- User ID 500,000 → Check directory → Shard 1
- User ID 1,500,000 → Check directory → Shard 2

**Pros:**
- **Flexible**: Can rebalance by updating directory (no data migration immediately)
- **Custom logic**: Can shard based on complex rules
- **Dynamic**: Easy to add shards, update mappings

**Cons:**
- **Directory is bottleneck**: Every query must lookup directory first
- **Single point of failure**: If directory down, entire system down
- **Latency**: Extra hop for every query

**Mitigation**: Cache directory, replicate directory.

**When to use**: When need flexibility and custom sharding logic.

---

### **5. Geo-Based Sharding**

**How it works**: Partition data by geographic region.

**Example**:
- Shard 1 (US-East): Users in US-East coast
- Shard 2 (US-West): Users in US-West coast
- Shard 3 (EU): Users in Europe
- Shard 4 (Asia): Users in Asia

**Pros:**
- **Reduced latency**: Data stored close to users
- **Compliance**: Data sovereignty laws (EU data must stay in EU)
- **Natural partitioning**: Users mostly access local data

**Cons:**
- **Uneven distribution**: Some regions have more users
- **Cross-region queries**: Complex and slow

**When to use**: Global applications with geographic user distribution.

---

## Choosing a Partition Key

**Partition key** determines how data is distributed across shards.

### **Good Partition Key Characteristics:**

1. **High cardinality**: Many unique values (good: user ID, order ID; bad: country, gender)
2. **Uniform distribution**: Values evenly spread (good: hash of email; bad: sequential IDs)
3. **Predictable access patterns**: Key used in most queries (good: user_id for user queries)

### **Examples:**

**Good**: \`user_id\` for user data
- High cardinality (millions of users)
- Uniform distribution (with hash)
- Queries typically filter by user_id

**Bad**: \`country\` for user data
- Low cardinality (~200 countries)
- Uneven distribution (US has more users than Liechtenstein)
- Creates hotspots

**Bad**: \`created_at\` (timestamp) for logs
- Recent data all goes to one shard (hotspot)
- Old shards idle

**Better for logs**: \`hash(log_id)\` or combine \`created_at\` + \`hash(source_id)\`

---

## Cross-Shard Operations

### **Problem: Joins Across Shards**

**Example**: Users sharded by \`user_id\`, Posts sharded by \`post_id\`.

Query: "Get all posts by user 123 with >100 likes"

**Challenge**: 
- User 123 on Shard 2
- Posts scattered across all shards

**Solution approaches:**

**1. Denormalization**: Store user info with each post (duplicate data, avoid join)

**2. Application-level joins**: Query all shards, merge results in application

**3. Shard by same key**: Shard users AND posts by \`user_id\` (colocate related data)

**Recommendation**: Design schema to avoid cross-shard joins. Denormalize or shard by dominant access pattern.

---

### **Problem: Aggregations Across Shards**

**Example**: "Count total users"

**Challenge**: Users distributed across 10 shards.

**Solution**: Map-Reduce pattern
- **Map**: Each shard counts its users
- **Reduce**: Sum counts from all shards

**Implementation**:

**Map-Reduce Example:**
- Shard 1: 1M users
- Shard 2: 1.2M users
- Shard 3: 900K users
- ...
- Total: 1M + 1.2M + 900K + ... = 15M users

**Cost**: O(num_shards) queries.

---

### **Problem: Distributed Transactions**

**Example**: Transfer money from user A (Shard 1) to user B (Shard 2).

**Challenge**: Need atomicity across shards (both succeed or both fail).

**Solution approaches:**

**1. Avoid distributed transactions**: Design to avoid (e.g., use event sourcing)

**2. Two-Phase Commit (2PC)**:
- Phase 1: Ask all shards "Can you commit?"
- Phase 2: If all yes, commit; if any no, rollback

**Cons**: Slow, complex, blocks on failure

**3. Saga pattern**: Sequence of local transactions with compensating actions

**Recommendation**: Design to avoid distributed transactions. They're slow and complex.

---

## Rebalancing Shards

**Scenario**: Add a new shard (4 shards → 5 shards).

### **Problem with Simple Hash**

Old: \`shard_id = hash(key) % 4\`  
New: \`shard_id = hash(key) % 5\`

**Result**: Almost ALL data needs to move (hash values change).

**Example**:
- User 123: \`hash(123) % 4\` = 3 → Shard 3
- After adding Shard 5: \`hash(123) % 5\` = 2 → Shard 2 (moves!)

**This is why consistent hashing is better!**

---

### **Rebalancing Strategies**

**1. Stop-the-world**: Stop application, migrate data, restart
- **Pro**: Simple
- **Con**: Downtime

**2. Dual-write**: Write to both old and new shards during migration
- **Pro**: No downtime
- **Con**: Complex

**3. Consistent hashing**: Only move data from neighboring shards
- **Pro**: Minimal data movement
- **Con**: Requires consistent hashing implementation

---

## Handling Hotspots

**Problem**: One shard receives disproportionate traffic.

**Example**: Celebrity user with millions of followers on Shard 3. All requests for that user hit Shard 3.

### **Solutions:**

**1. Further partition hot shard**: Split Shard 3 into Shard 3a and 3b

**2. Replicate hot data**: Cache celebrity user's data in Redis (read-heavy workload)

**3. Composite partition key**: Instead of just \`user_id\`, use \`hash(user_id + timestamp)\` to distribute

**4. Dedicated shard for hot entities**: Move celebrity users to separate, more powerful shard

---

## Real-World Sharding Examples

### **Instagram**

**Challenge**: Billions of photos, can't fit in single database.

**Solution**: Shard by \`photo_id\`
- Generate unique photo ID with embedded shard ID
- Each photo stored on specific shard based on ID
- Consistent hashing for even distribution

**ID format**: \`[shard_id][timestamp][sequence]\`

---

### **Twitter**

**Challenge**: Billions of tweets, high write throughput.

**Solution**: Shard by \`user_id\`
- User's tweets stored together on same shard (good for "get user timeline" query)
- Snowflake ID generation (distributed, k-sorted IDs)

**Trade-off**: Cross-user queries (follower timelines) require fan-out to multiple shards.

---

### **Uber**

**Challenge**: Millions of trips, geographic distribution.

**Solution**: Geo-based sharding
- Shard by city/region
- Trips in San Francisco → Shard SF
- Trips in New York → Shard NY

**Benefit**: Reduced latency (data close to users), compliance with local laws.

---

## Interview Tips

### **Common Questions:**

**Q: "Your database has 100M users and can't handle the load. How would you scale it?"**

✅ Good answer:
"I'd implement sharding:
1. **Partition key**: \`user_id\` (high cardinality, queries filter by user_id)
2. **Strategy**: Consistent hashing (minimal rebalancing when adding shards)
3. **Initial shards**: 10 shards (10M users each)
4. **Rebalancing**: As data grows, add shards incrementally
5. **Trade-off**: Cross-user queries (e.g., 'find all users in city X') become harder, would need to query all shards or maintain separate index"

**Q: "What's the difference between sharding and replication?"**

✅ Good answer:
"**Sharding (horizontal partitioning)**: Splits data across machines. Each machine holds DIFFERENT data. Goal: Scale capacity and throughput.

**Replication**: Copies SAME data to multiple machines. Each machine holds same data. Goal: Availability and read scalability.

Often used together: Shard for writes, replicate each shard for reads."

**Q: "How do you handle a celebrity user with millions of followers (hotspot)?"**

✅ Good answer:
"Several approaches:
1. **Cache**: Cache celebrity's data in Redis (read-heavy workload)
2. **Read replicas**: Add more read replicas for shard containing celebrity
3. **Separate shard**: Move hot entities to dedicated, more powerful shard
4. **CDN**: Cache celebrity's public data at edge
5. **Rate limiting**: Limit requests per user to prevent abuse"

---

## Key Takeaways

1. **Sharding = horizontal partitioning**: Splits data across machines by rows
2. **Partition key critical**: Choose high-cardinality, uniformly distributed key
3. **Hash-based sharding**: Uniform distribution, but rebalancing hard
4. **Consistent hashing**: Industry standard, minimal rebalancing when adding shards
5. **Avoid cross-shard joins**: Denormalize or colocate related data
6. **Hotspots**: Monitor and handle (caching, replication, repartitioning)
7. **Rebalancing**: Plan ahead, use consistent hashing to minimize data movement`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your application shards user data by user_id using simple hash: shard_id = hash(user_id) % 4. You need to add a 5th shard. Walk through what happens and explain why this is problematic. How would you design it differently?',
          sampleAnswer:
            "PROBLEM WITH SIMPLE HASH SHARDING: Current setup: 4 shards, formula shard_id = hash(user_id) % 4. User 123: hash(123) = 1000 → 1000 % 4 = 0 → Shard 0. User 456: hash(456) = 2001 → 2001 % 4 = 1 → Shard 1. User 789: hash(789) = 3002 → 3002 % 4 = 2 → Shard 2. WHAT HAPPENS WHEN ADDING SHARD 5: New formula: shard_id = hash(user_id) % 5. User 123: hash(123) = 1000 → 1000 % 5 = 0 → Still Shard 0 (lucky!). User 456: hash(456) = 2001 → 2001 % 5 = 1 → Still Shard 1 (lucky!). User 789: hash(789) = 3002 → 3002 % 5 = 2 → Still Shard 2 (lucky!). But: User 111: hash(111) = 998 → OLD: 998 % 4 = 2 (Shard 2) → NEW: 998 % 5 = 3 (Shard 3) → MOVES!. SCALE OF PROBLEM: On average, ~80% of data needs to move! Why: Changing modulo from 4 to 5 completely changes hash distribution. Almost every key gets reassigned. IMPACT: (1) Massive data migration: Copy 80% of database across network. (2) Downtime: Application needs to pause while migration happens. (3) Time: Migrating 100GB database might take hours. (4) Risk: If migration fails midway, data inconsistent. (5) Cost: Network bandwidth, storage I/O, potential lost revenue. REAL EXAMPLE: Instagram had 100TB of data. Adding a shard would require moving 80TB. At 1Gbps network: 80TB = 640,000 Gb → 640,000 seconds = 7.4 days (!). BETTER DESIGN: CONSISTENT HASHING. How Consistent Hashing Works: (1) Hash ring: Circle from 0 to 2^32-1. (2) Place shards on ring: Shard 0 at 25%, Shard 1 at 50%, Shard 2 at 75%, Shard 3 at 100%. (3) Place data on ring: hash(user_id) gives position on ring. (4) Assign to next shard clockwise. Example: User 123: hash(123) = 30% → Next shard clockwise = Shard 1 (at 50%). User 456: hash(456) = 60% → Next shard clockwise = Shard 2 (at 75%). User 789: hash(789) = 85% → Next shard clockwise = Shard 3 (at 100%). ADDING SHARD 5 WITH CONSISTENT HASHING: Place Shard 4 at 40% on ring. Before: Data from 25% to 50% → Shard 1. After: Data from 25% to 40% → Shard 4 (NEW). Data from 40% to 50% → Shard 1 (unchanged). Result: Only data in range 25%-40% moves (1/4 of Shard 1's data). COMPARISON: Simple hash: ~80% of ALL data moves. Consistent hash: ~20% of ONE shard's data moves (1/20 = 5% of total). Reduction: 16× less data movement! VIRTUAL NODES OPTIMIZATION: Problem: 4 physical shards on ring might not distribute evenly. Solution: Each physical shard appears at multiple positions (virtual nodes). Example: Shard 0: appears at 10%, 35%, 60%, 85% (4 virtual nodes). Shard 1: appears at 15%, 40%, 65%, 90% (4 virtual nodes). Result: More uniform distribution, even better rebalancing. IMPLEMENTATION: Use consistent hashing library: Python: hash_ring package, Java: Ketama, Go: stathat/consistent. Configure virtual nodes: typically 150-500 per physical shard. MIGRATION PROCESS WITH CONSISTENT HASHING: (1) Add Shard 4 to ring. (2) Identify affected data range (e.g., 25%-40%). (3) Dual-write: New writes go to both old and new shard. (4) Backfill: Copy existing data from Shard 1 to Shard 4. (5) Switch reads: Start reading from Shard 4 for that range. (6) Stop dual-write: Delete data from Shard 1. Result: No downtime, gradual migration. FINAL RECOMMENDATION: Never use simple hash (hash % N) for sharding in production. Always use consistent hashing for systems that will scale. Accept upfront complexity for long-term operability. REAL-WORLD USAGE: Cassandra: Consistent hashing with virtual nodes. DynamoDB: Consistent hashing. Redis Cluster: Hash slots (variant of consistent hashing). Chord DHT: Consistent hashing for P2P systems.",
          keyPoints: [
            'Simple hash (hash % N): adding shard requires ~80% of data to move',
            'Consistent hashing: only ~5-10% of data moves when adding shard',
            'Consistent hashing uses ring structure: data assigned to next shard clockwise',
            'Virtual nodes: each physical shard appears multiple times on ring for uniform distribution',
            'Production systems (Cassandra, DynamoDB, Redis) use consistent hashing',
          ],
        },
        {
          id: 'q2',
          question:
            "You're designing a social media platform. Users create posts, and other users like/comment on posts. You need to shard the data. Walk through your sharding strategy: what partition keys would you use for users, posts, likes, and comments? Explain trade-offs.",
          sampleAnswer:
            "SHARDING STRATEGY FOR SOCIAL MEDIA PLATFORM: ENTITIES TO SHARD: (1) Users (profiles, settings). (2) Posts (content, metadata). (3) Likes (user_id, post_id pairs). (4) Comments (user_id, post_id, comment text). KEY QUERIES TO OPTIMIZE: (1) Get user profile: by user_id. (2) Get user's posts: by user_id. (3) Get post with likes and comments: by post_id. (4) Get user's feed: posts from followed users. OPTION 1: SHARD EVERYTHING BY USER_ID. Strategy: All user data, their posts, their likes, their comments on same shard. Partition key: user_id. Shards: Shard 1: Users 1-25M + their posts/likes/comments. Shard 2: Users 25M-50M + their posts/likes/comments. Shard 3: Users 50M-75M + their posts/likes/comments. PROS: (1) User profile query: Single shard (fast). (2) Get user's posts: Single shard (fast). (3) Data locality: All user data together. CONS: (1) Get post with likes/comments: Post on Shard 1, but likes from users on all shards → Need to query ALL shards (slow!). (2) Hotspot: Celebrity user with millions of posts overwhelms one shard. (3) Uneven distribution: Some users very active (many posts), others inactive. OPTION 2: SHARD EVERYTHING BY POST_ID. Strategy: Posts, likes, and comments for each post on same shard. Partition key: post_id. Shards: Shard 1: Posts 1-25M + their likes/comments. Shard 2: Posts 25M-50M + their likes/comments. Shard 3: Posts 50M-75M + their likes/comments. Users: Separate table, sharded by user_id. PROS: (1) Get post with likes/comments: Single shard (fast). (2) Even distribution: Posts relatively uniform size. CONS: (1) Get user's posts: User's posts scattered across all shards → Query ALL shards (slow). (2) User profile + posts: Need 2 separate queries (user shard + multiple post shards). RECOMMENDED: HYBRID SHARDING (SEPARATE PARTITION KEYS). Strategy: Shard different entities by their dominant access pattern. USERS TABLE: Partition key: user_id. Sharding: Hash-based (uniform distribution). Query: Get user by user_id → Single shard. POSTS TABLE: Partition key: user_id (NOT post_id!). Why: Most common query: \"Get user's posts\". Sharding: Posts by same user colocated on same shard as user. Example: User 123 on Shard 2 → All posts by User 123 also on Shard 2. Query: Get User 123's posts → Single shard. LIKES TABLE: Partition key: post_id. Why: Most common query: \"Get all likes for post X\". Sharding: All likes for a post on same shard. Schema: (post_id, user_id, timestamp). Example: Post 456 on Shard 3 → All likes for Post 456 on Shard 3. Query: Get likes for Post 456 → Single shard. COMMENTS TABLE: Partition key: post_id. Why: Most common query: \"Get all comments for post X\". Sharding: All comments for a post on same shard as its likes. Schema: (post_id, user_id, comment_text, timestamp). Query: Get comments for Post 456 → Single shard. FEED GENERATION (COMPLEX QUERY): Query: \"Get feed for User 123\" (posts from followed users). Challenge: Followed users scattered across all shards. SOLUTION 1: FAN-OUT ON WRITE (PUSH MODEL): When user posts: (1) User 123 posts → Identify User 123's followers. (2) Write post_id to each follower's feed table. (3) Feed table sharded by user_id (colocated with user). Result: Get feed for User 789 → Single shard (fast read). Trade-off: Slow write (if user has 1M followers, write 1M times). Use case: Read-heavy, users have moderate followers. SOLUTION 2: FAN-OUT ON READ (PULL MODEL): When user requests feed: (1) Get User 789's followed users (e.g., 200 users). (2) Query posts from each followed user (scatter to multiple shards). (3) Merge and sort by timestamp. Result: Fast write (single write), slower read (query multiple shards). Use case: Write-heavy, users follow many people. HYBRID (TWITTER APPROACH): Regular users: Fan-out on write. Celebrities: Fan-out on read (don't push to millions of timelines). Feed query: Merge both sources. DENORMALIZATION TO AVOID CROSS-SHARD JOINS: Problem: Post on Shard 1, likes on Shard 2 (if sharded differently). Solution: Denormalize: Store like count WITH post (no need to query likes table). Update: When user likes post: (1) Write to Likes table. (2) Increment like_count in Posts table (eventually consistent OK). Result: Get post with like count → Single query. HANDLING HOTSPOTS: Problem: Viral post gets millions of likes → Likes table shard overloaded. Solutions: (1) Cache hot posts in Redis. (2) Rate limit likes (e.g., max 1000 likes/sec per post). (3) Write-back caching: Batch writes to likes table. (4) Replicate hot post's shard (more read replicas). SUMMARY TABLE: Entity / Partition Key / Reason: Users / user_id / Get user profile, Hash-based for uniform distribution. Posts / user_id / Get user's posts (colocate with user). Likes / post_id / Get all likes for a post. Comments / post_id / Get all comments for a post. Feeds / user_id / Get user's feed (fan-out on write). TRADE-OFFS ACCEPTED: (1) Cross-shard queries: Getting post + likes requires 2 shards (posts shard + likes shard) → Mitigate with denormalization (store like count with post). (2) Feed generation: Either slow writes (fan-out on write) or slow reads (fan-out on read) → Hybrid approach for celebrities. (3) Eventual consistency: Like count might lag slightly → Acceptable for social media. REAL-WORLD EXAMPLES: Instagram: Shards photos by user_id, likes by post_id. Twitter: Shards tweets by user_id, fans out to followers' feeds. Facebook: Complex sharding with TAO (social graph cache). FINAL RECOMMENDATION: Shard by dominant access pattern. Users/Posts by user_id, Likes/Comments by post_id. Denormalize to avoid cross-shard joins (store counts). Use hybrid fan-out for feed generation.",
          keyPoints: [
            'Shard by dominant access pattern: Users/Posts by user_id, Likes/Comments by post_id',
            'Denormalize to avoid cross-shard joins: store like_count with post',
            'Feed generation: hybrid approach (fan-out on write for regular users, on read for celebrities)',
            'Hotspots: cache viral posts, rate limit, batch writes',
            'Accept trade-off: some cross-shard queries for better performance on common queries',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain the difference between sharding and replication. When would you use each? Can you use both together? Give a real-world example.',
          sampleAnswer:
            "SHARDING VS REPLICATION: SHARDING (HORIZONTAL PARTITIONING): Definition: Splitting data across multiple machines. Each machine holds DIFFERENT subset of data. Goal: Increase capacity (storage, throughput, parallelism). Example: 100M users sharded across 10 servers: Shard 1: Users 1-10M, Shard 2: Users 10M-20M, ..., Shard 10: Users 90M-100M. Each shard has unique data (no overlap). Query: Get User ID 5M → Query Shard 1 only. Benefit: Storage: Single DB max 10M users → Sharding allows 100M users. Throughput: 10 shards × 10K writes/sec each = 100K writes/sec total. Trade-offs: Complexity: Cross-shard queries harder. No redundancy: If Shard 1 fails, Users 1-10M unavailable. REPLICATION: Definition: Copying SAME data to multiple machines. Each machine holds IDENTICAL data. Goal: Increase availability, fault tolerance, and read scalability. Example: Primary database + 2 replicas: Primary: Users 1-100M (handles writes). Replica 1: Copy of Users 1-100M (handles reads). Replica 2: Copy of Users 1-100M (handles reads). Query: Read User ID 5M → Can query Primary, Replica 1, OR Replica 2. Benefit: Availability: If Primary fails, promote Replica 1 to Primary (no data loss). Read scaling: Distribute read queries across 3 servers (3× read throughput). Fault tolerance: Data redundancy protects against hardware failure. Trade-offs: Storage cost: 3× storage (same data copied 3 times). Write complexity: Writes go to Primary, must replicate to Replicas (replication lag). COMPARISON TABLE: Aspect / Sharding / Replication: Data distribution: Different data per machine / Same data per machine. Goal: Capacity & throughput / Availability & read scaling. Storage: Total capacity = sum of shards / Total capacity = single machine (redundant). Writes: Distributed across shards / All writes to primary, then replicated. Reads: Query specific shard / Distribute across replicas. Fault tolerance: No (shard failure = data loss) / Yes (replicas have backup). Complexity: High (cross-shard queries) / Low (replicas interchangeable). WHEN TO USE SHARDING: (1) Data doesn't fit on single machine (>1TB). (2) Write throughput exceeds single machine capacity (>10K writes/sec). (3) Need horizontal scalability (add more machines as data grows). (4) Read patterns can be isolated (e.g., users don't need to query all shards). Examples: Social media (billions of users/posts). E-commerce (millions of products/orders). Analytics (petabytes of logs). WHEN TO USE REPLICATION: (1) Need high availability (system must stay up even if server fails). (2) Read-heavy workload (10× more reads than writes). (3) Geographic distribution (replicas in different regions for low latency). (4) Disaster recovery (backup in case of data loss). Examples: E-commerce product catalog (read-heavy). Banking (availability critical). Content websites (read-heavy, low writes). USING BOTH TOGETHER (COMMON!): Strategy: Shard for capacity, replicate each shard for availability. Example: 100M users, 10 shards, 3 replicas per shard: Shard 1: Users 1-10M, Shard 1 Primary + Replica1a + Replica1b. Shard 2: Users 10M-20M, Shard 2 Primary + Replica2a + Replica2b. ..., Shard 10: Users 90M-100M, Shard 10 Primary + Replica10a + Replica10b. Total: 30 database instances (10 primaries + 20 replicas). Benefits: Capacity: 10 shards → 100M users (single DB can't hold this). Availability: If Shard 1 Primary fails, promote Replica1a to Primary (no downtime). Read scaling: Each shard has 3 instances → 3× read throughput per shard. Fault tolerance: Data redundancy within each shard. Queries: Write User 5M: Route to Shard 1 Primary. Read User 5M: Route to Shard 1 (any of Primary, Replica1a, Replica1b). REAL-WORLD EXAMPLE: INSTAGRAM. Scale: Billions of photos, petabytes of data. Architecture: Photos sharded by photo_id across thousands of shards. Each shard replicated 3× (primary + 2 replicas) for availability. Photo storage on Amazon S3 (itself sharded and replicated). Implementation: Cassandra database (built-in sharding + replication). Shard key: photo_id (consistent hashing). Replication factor: 3 (RF=3). Queries: Post photo: Shard by photo_id, write to 3 replicas. Get photo: Read from nearest replica for low latency. User feed: Fan-out to multiple shards (user's followed users). Benefits: Capacity: Petabytes of photos (single DB can't handle). Availability: 99.99% uptime (replica failover). Read scaling: Billions of photo views/day. Fault tolerance: Hardware failures don't cause data loss. Cost: Thousands of database instances, but necessary at scale. ANOTHER EXAMPLE: MONGODB SHARDED CLUSTER. Setup: Data sharded across multiple shards (e.g., 5 shards). Each shard is a replica set (primary + 2 replicas). Config servers (3 instances, replicated) store metadata. Mongos routers (query routers) direct queries to appropriate shards. Architecture: Shard 1: Replica set (Primary + 2 Replicas). Shard 2: Replica set (Primary + 2 Replicas). ..., Shard 5: Replica set (Primary + 2 Replicas). Total: 15 data instances + 3 config servers + N routers. Benefits: Sharding: Horizontal scaling (add more shards as data grows). Replication: High availability (automatic failover within replica sets). Operations: Add shard: MongoDB rebalances data automatically (consistent hashing). Shard failure: Replica promoted to Primary, no downtime. WHEN NOT TO USE SHARDING: (1) Data fits comfortably on single machine (<100GB). (2) Low traffic (<1000 QPS). (3) Team lacks expertise to manage sharding complexity. Alternative: Scale vertically (bigger machine), add replication for availability. SUMMARY: Sharding: For capacity and write throughput. Splits data across machines (different data). Replication: For availability and read scaling. Copies data to machines (same data). Together: Shard for capacity, replicate each shard for availability. Real-world: Most large-scale systems use both (Instagram, Facebook, Twitter). Rule of thumb: Start with replication, add sharding when single machine can't handle load.",
          keyPoints: [
            'Sharding: different data per machine, for capacity and write throughput',
            'Replication: same data per machine, for availability and read scaling',
            'Use both together: shard for capacity, replicate each shard for availability',
            'Real-world: Instagram shards photos, replicates each shard 3× for availability',
            "Start with replication, add sharding when single machine can't handle load",
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'You have a users table with 100M rows. You shard by user_id using hash(user_id) % 10. Which query is MOST efficient?',
          options: [
            'SELECT * FROM users WHERE user_id = 12345',
            'SELECT * FROM users WHERE country = "USA"',
            'SELECT * FROM users WHERE created_at > "2023-01-01"',
            'SELECT COUNT(*) FROM users',
          ],
          correctAnswer: 0,
          explanation:
            'Query by user_id is most efficient: hash(12345) % 10 = X determines exact shard, query hits only 1 shard. Other queries: (1) country = "USA": Must query ALL 10 shards (country not in partition key). (2) created_at > "2023-01-01": Must query ALL 10 shards (date not in partition key). (3) COUNT(*): Must query ALL 10 shards and sum results. General rule: Queries filtering by partition key are efficient (single shard). Queries not filtering by partition key require querying all shards (scatter-gather).',
        },
        {
          id: 'mc2',
          question:
            'What is the main advantage of consistent hashing over simple hash (hash % N) for sharding?',
          options: [
            'Consistent hashing provides better data distribution',
            'Consistent hashing is simpler to implement',
            'Adding/removing shards requires minimal data movement',
            'Consistent hashing enables efficient range queries',
          ],
          correctAnswer: 2,
          explanation:
            'Minimal data movement when adding/removing shards is the main advantage. Simple hash (hash % N): Changing N requires rehashing almost all data (~80% moves). Consistent hashing: Only data from neighboring shards moves (~5-10% of total). This is critical for production systems that need to scale without downtime. Option 1: Both provide similar distribution (with virtual nodes). Option 2: Consistent hashing is actually more complex. Option 4: Neither enables efficient range queries (both use hashing).',
        },
        {
          id: 'mc3',
          question:
            "You're sharding a logs table by created_at (timestamp). What problem will you likely encounter?",
          options: [
            'Uneven distribution of data across shards',
            'Recent logs go to one shard (hotspot), old shards are idle',
            'Difficult to query logs by timestamp',
            'Too much data movement when adding shards',
          ],
          correctAnswer: 1,
          explanation:
            "Sharding by timestamp creates hotspot: all recent logs go to one shard (the current time range). Example: 10 shards by date ranges. Today's logs: ALL writes hit Shard 10 (today's shard). Old shards (1-9): Idle (no writes, only rare historical reads). Result: Uneven load, Shard 10 overloaded. Better solution: Shard by hash(log_id) or composite key hash(source_id + timestamp) for uniform distribution. Lesson: Avoid sharding by sequential/time-based keys unless data is truly immutable and access patterns are historical.",
        },
        {
          id: 'mc4',
          question:
            'What is the recommended approach for handling cross-shard joins in a sharded database?',
          options: [
            'Use distributed transactions (2-phase commit)',
            'Query all shards and join results in application',
            'Avoid cross-shard joins by denormalizing or colocating related data',
            'Use a global secondary index spanning all shards',
          ],
          correctAnswer: 2,
          explanation:
            'Avoid cross-shard joins by denormalizing or colocating related data. Example: If Users and Posts often joined, shard both by user_id (colocate). Or: Denormalize user info into Posts table (duplicate data, avoid join). Option 1 (2PC): Slow, complex, blocks on failure. Option 2 (app-level join): Works but slow (query all shards). Option 4 (global index): Possible but adds complexity and latency. Best practice: Design schema to avoid cross-shard operations. Denormalization is often acceptable trade-off in distributed systems.',
        },
        {
          id: 'mc5',
          question:
            'Your sharded database uses range-based sharding by user_id: Shard 1 (1-1M), Shard 2 (1M-2M), Shard 3 (2M-3M). User IDs are sequential (auto-increment). What problem will you encounter?',
          options: [
            'Uneven read distribution across shards',
            'All new users go to the latest shard (hotspot for writes)',
            'Difficult to query users by ID range',
            'Cannot add more shards without rehashing',
          ],
          correctAnswer: 1,
          explanation:
            'Sequential IDs with range sharding creates write hotspot: all new users go to the latest shard. Example: User IDs 1-2M exist. Shard 1: Users 1-1M (idle for writes). Shard 2: Users 1M-2M (idle for writes). Shard 3: Users 2M-3M (ALL writes go here). Result: Shard 3 overloaded, Shards 1-2 underutilized. Solution: Use hash-based sharding (uniform distribution) instead of range-based for sequential IDs. Or: Use UUIDs/random IDs instead of sequential IDs with range sharding.',
        },
      ],
    },
    {
      id: 'database-replication',
      title: 'Database Replication',
      content: `Database replication copies data from one database to another to improve availability, fault tolerance, and read scalability.

## What is Database Replication?

**Definition**: The process of copying and maintaining database data in multiple database instances to ensure data availability, reliability, and performance.

### **Why Replicate Databases?**

**Without Replication:**
- Single point of failure (database crashes = entire app down)
- Limited read capacity (single database can't handle high read traffic)
- No disaster recovery
- Maintenance requires downtime

**With Replication:**
- High availability (if primary fails, replica takes over)
- Read scalability (distribute reads across replicas)
- Disaster recovery (data backup in different locations)
- Zero-downtime maintenance (update replicas one at a time)

**Real-world**: Facebook has thousands of MySQL replicas to handle billions of reads per day.

---

## Primary-Replica (Master-Slave) Architecture

**Most common replication pattern.**

### **Architecture**

**Components:**
- **Primary (Master)**: Accepts writes, source of truth
- **Replicas (Slaves)**: Receive data from primary, handle reads

**Data flow:**
1. Application writes to Primary
2. Primary logs changes
3. Changes replicated to Replicas
4. Application reads from Replicas

**Example:**
- Primary: Handles 10K writes/sec
- Replica 1: Handles 50K reads/sec
- Replica 2: Handles 50K reads/sec
- Replica 3: Handles 50K reads/sec
- Total read capacity: 150K reads/sec

---

## Synchronous vs Asynchronous Replication

### **Synchronous Replication**

**How it works**: Write confirmed only after data written to BOTH primary and replica(s).

**Flow:**
1. App writes to Primary
2. Primary writes to disk
3. Primary sends data to Replica
4. Replica writes to disk
5. Replica acknowledges to Primary
6. Primary confirms write to App

**Pros:**
- **Strong consistency**: Replica always has latest data
- **No data loss**: If primary fails immediately after write, data exists on replica
- **Guaranteed durability**

**Cons:**
- **Slower writes**: Must wait for replica acknowledgment (network latency)
- **Availability risk**: If replica down, writes fail or block
- **Geographic limitations**: High latency if replica in different region

**When to use**: Banking, financial transactions (data loss unacceptable).

---

### **Asynchronous Replication**

**How it works**: Write confirmed after data written to primary, replica updated later.

**Flow:**
1. App writes to Primary
2. Primary writes to disk
3. Primary confirms write to App immediately
4. Primary asynchronously sends data to Replica (in background)
5. Replica writes to disk eventually

**Pros:**
- **Fast writes**: No wait for replica
- **High availability**: Primary can write even if replica down
- **Works across regions**: No latency penalty

**Cons:**
- **Eventual consistency**: Replica may lag behind primary (replication lag)
- **Data loss risk**: If primary fails before replicating, recent writes lost
- **Stale reads**: Reading from replica may return old data

**When to use**: Most web applications (social media, e-commerce), where slight delay acceptable.

**Replication lag**: Time between write to primary and appearance on replica.
- Typical: 0-5 seconds
- High load: Up to 60 seconds or more

---

### **Semi-Synchronous Replication**

**Compromise between sync and async.**

**How it works**: Write confirmed after at least ONE replica acknowledges, others updated asynchronously.

**Flow:**
1. App writes to Primary
2. Primary writes to disk
3. Primary sends to all replicas
4. Wait for ONE replica to acknowledge
5. Confirm write to App
6. Other replicas update asynchronously

**Pros:**
- Faster than fully synchronous
- Better data durability than async (at least 1 replica has data)
- Balance of performance and safety

**When to use**: Production systems needing durability without full sync penalty.

---

## Multi-Master Replication

**Multiple databases accept writes simultaneously.**

### **Architecture**

**Setup:**
- Multiple primary nodes (no single primary)
- Each node can accept writes
- Changes replicate to other nodes

**Example:**
- Primary 1 (US): Handles US writes
- Primary 2 (EU): Handles EU writes
- Both replicate to each other

**Pros:**
- **Write scalability**: Distribute writes across multiple nodes
- **No single point of failure**: Any node can accept writes
- **Geographic distribution**: Users write to nearest node (low latency)

**Cons:**
- **Conflict resolution**: Two users update same data on different nodes → conflict
- **Complexity**: Much harder to implement and maintain
- **Consistency challenges**: Eventual consistency, not immediate

---

### **Conflict Resolution**

**Problem**: User A updates record on Primary 1, User B updates same record on Primary 2 simultaneously.

**Strategies:**

**1. Last Write Wins (LWW)**
- Use timestamp to determine which write is newer
- Discard older write
- **Problem**: Clocks may not be synchronized, data loss possible

**2. Version Vectors**
- Track version history of each record
- Merge conflicting updates
- **Example**: CouchDB uses this approach

**3. Application-Level Resolution**
- Store both versions, let application decide
- **Example**: Shopping cart (merge both carts)

**4. Conflict-Free Replicated Data Types (CRDTs)**
- Data structures designed to merge automatically
- **Example**: Counters, sets, maps

---

## Read Replicas and Read Scaling

**Purpose**: Offload read traffic from primary to replicas.

### **Configuration**

**Application routing:**
- Writes → Primary
- Reads → Replicas (round-robin or load balanced)

**Example:**
\`\`\`
Write Path:
App → Primary DB

Read Path:
App → Load Balancer → Replica 1, 2, 3, ...
\`\`\`

**Scaling reads:**
- 1 primary + 0 replicas: 10K reads/sec
- 1 primary + 5 replicas: 50K reads/sec
- 1 primary + 10 replicas: 100K reads/sec

**Linear read scaling** (add replicas → more read capacity).

---

### **Handling Replication Lag**

**Problem**: User writes data, immediately reads from replica, data not yet replicated → appears data is lost.

**Solutions:**

**1. Read Your Own Writes**
- After user writes, route their reads to primary for short time (e.g., 5 seconds)
- Then route to replicas
- Ensures user sees their own changes immediately

**2. Sticky Sessions**
- Route user's requests to same replica consistently
- Replica eventually catches up
- User experiences consistency (even if stale)

**3. Monitor Replication Lag**
- Track lag: \`primary_log_position - replica_log_position\`
- If lag > threshold, don't route reads to that replica
- Alert if lag consistently high

**4. Causal Consistency**
- Use version numbers or timestamps
- Read from replica only if version >= version of last write

---

## Failover and Promotion

**Scenario**: Primary database crashes. How to maintain availability?

### **Automatic Failover**

**Process:**
1. **Detection**: Monitor detects primary is down (heartbeat timeout)
2. **Election**: Choose which replica to promote (typically most up-to-date)
3. **Promotion**: Promote replica to new primary
4. **Reconfiguration**: Update application to write to new primary
5. **Recovery**: When old primary recovers, it becomes a replica

**Challenges:**

**1. Split-Brain Problem**
- Network partition isolates primary
- System thinks primary is down, promotes replica
- Now two primaries (both accepting writes)
- **Solution**: Use consensus algorithm (Raft, Paxos) or fencing

**2. Data Loss**
- If using async replication, promoted replica may not have latest writes
- **Solution**: Accept data loss or use semi-sync replication

**3. Failover Time**
- Detection: 10-30 seconds
- Promotion: 10-60 seconds
- Total downtime: 30-90 seconds
- **Solution**: Use automated tools (Orchestrator, ProxySQL)

---

### **Manual Failover**

**Process:**
1. Administrator manually promotes replica
2. Update DNS or load balancer configuration
3. Restart application with new primary connection

**When to use:**
- Planned maintenance
- Upgrading database version
- When automatic failover risky

**Downtime**: Minutes to hours (depending on planning).

---

## Real-World Replication Examples

### **MySQL Replication**

**Configuration:**
\`\`\`
Primary → Replica1, Replica2, Replica3
    \`\`\`

**Replication methods:**
- **Statement-based**: Replicates SQL statements
- **Row-based**: Replicates actual data changes (more reliable)
- **Mixed**: Hybrid approach

**Lag monitoring:**
- \`SHOW SLAVE STATUS\` → check \`Seconds_Behind_Master\`

---

### **PostgreSQL Streaming Replication**

**Configuration:**
- Primary streams WAL (Write-Ahead Log) to replicas
- Replicas replay WAL to stay in sync

**Modes:**
- **Asynchronous**: Default, fast
- **Synchronous**: Wait for replica acknowledgment
- **Quorum-based**: Wait for N of M replicas

**Hot Standby**: Replicas accept read queries while replicating.

---

### **MongoDB Replica Sets**

**Configuration:**
- 3+ nodes: Primary + Secondaries
- Automatic failover via election (Raft-like consensus)

**Example:**
- 3-node replica set
- Primary accepts writes
- If primary fails, secondaries elect new primary (takes ~10 seconds)

**Read preferences:**
- Primary: Always read from primary (strong consistency)
- PrimaryPreferred: Primary if available, secondary otherwise
- Secondary: Always read from secondary (stale reads possible)
- Nearest: Lowest latency node

---

## Replication Topologies

### **1. Simple Primary-Replica**

\`\`\`
Primary → Replica1
       → Replica2
       → Replica3
    \`\`\`

**Pros**: Simple, easy to understand
**Cons**: Primary is bottleneck for replication

---

### **2. Cascading Replication**

\`\`\`
Primary → Replica1 → Replica2
                   → Replica3
    \`\`\`

**Pros**: Reduces replication load on primary
**Cons**: Increased replication lag for downstream replicas

---

### **3. Circular Replication (Multi-Master)**

\`\`\`
Primary1 ↔ Primary2 ↔ Primary3
    \`\`\`

**Pros**: Multi-region writes
**Cons**: Conflict resolution complexity

---

## Interview Tips

### **Common Questions:**

**Q: "How would you scale reads for a database receiving 100K reads/sec?"**

✅ Good answer: "Add read replicas:
1. Current: 1 primary handling all 100K reads/sec (overloaded)
2. Add 10 read replicas: Each handles 10K reads/sec
3. Load balancer distributes reads across replicas
4. Writes still go to primary
5. Monitor replication lag to ensure data not too stale"

**Q: "What happens if the primary database fails?"**

✅ Good answer: "Failover process:
1. Monitor detects primary down (heartbeat failure)
2. Promote most up-to-date replica to new primary
3. Reconfigure app to write to new primary
4. Total downtime: 30-90 seconds with auto-failover
5. Risk: If async replication, may lose recent writes (last few seconds)
6. Mitigation: Use semi-sync replication for critical data"

**Q: "How do you handle replication lag?"**

✅ Good answer: "Several strategies:
1. Monitor lag metrics (alert if >5 seconds)
2. Read-your-own-writes: Route user's reads to primary after they write
3. Remove lagging replicas from load balancer
4. Use semi-sync replication to reduce lag
5. Accept eventual consistency for non-critical reads"

---

## Key Takeaways

1. **Replication = copying data to multiple databases** for availability and read scaling
2. **Primary-replica** most common: primary handles writes, replicas handle reads
3. **Async replication**: Fast but eventual consistency (most common in practice)
4. **Sync replication**: Slow but strong consistency (use for critical data)
5. **Read replicas** enable horizontal read scaling (add replicas → more read capacity)
6. **Replication lag**: Monitor and handle (read-your-own-writes pattern)
7. **Failover**: Automatic preferred, 30-90 second downtime typical`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your application has high read traffic (100K reads/sec) but low write traffic (1K writes/sec). Users complain about slow response times. You have a single PostgreSQL database. Propose a solution and explain the trade-offs.',
          sampleAnswer:
            'SOLUTION: ADD READ REPLICAS. DIAGNOSIS: High read traffic overwhelming single database: 100K reads/sec is too much for one PostgreSQL instance (typically handles 10-20K reads/sec efficiently). Writes are low (1K writes/sec), so primary can handle write load easily. Bottleneck is reads, not writes. PROPOSED ARCHITECTURE: Primary-Replica Replication: 1 Primary database: Handles ALL writes (1K writes/sec). 10 Read Replicas: Each handles 10K reads/sec. Load balancer: Distributes reads across 10 replicas. Total read capacity: 10 replicas × 10K = 100K reads/sec. IMPLEMENTATION DETAILS: Replication Setup: Configure async replication from Primary to Replicas. Replication lag: Acceptable 1-5 seconds for most apps. Load Balancer: Round-robin or least-connections algorithm. Health checks: Remove lagging replicas if lag >10 seconds. Application Changes: Write queries: Route to Primary. Read queries: Route to Load Balancer → Replicas. BENEFITS: Read scalability: Handles 100K reads/sec (vs 10K before). Response time: Reduced from seconds to milliseconds. High availability: If one replica fails, other 9 still handle traffic. Fault tolerance: If Primary fails, promote replica to Primary. TRADE-OFFS ACCEPTED: TRADE-OFF 1: EVENTUAL CONSISTENCY. Problem: User writes data, immediately reads from replica, data not yet replicated → sees stale data. Impact: User posts comment, refreshes page, comment missing (appears for 1-5 seconds later). Severity: Minor for social media, critical for banking. Mitigation: (1) Read-your-own-writes: After user writes, route their reads to Primary for 5 seconds. (2) Sticky sessions: Route user to same replica (lag affects all users equally). (3) Show "saving..." indicator to set expectations. Decision: Acceptable for most web apps. TRADE-OFF 2: REPLICATION LAG. Problem: Under high load, replicas may lag 10-60 seconds behind Primary. Impact: Users see outdated data (e.g., old inventory counts, stale comments). Monitoring: Track replication lag: lag = Primary_log_position - Replica_log_position. Alert if lag >5 seconds consistently. Mitigation: (1) Remove lagging replicas from load balancer temporarily. (2) Add more replicas (reduce per-replica load). (3) Upgrade replica hardware (faster disk I/O). Decision: Monitor actively, accept occasional lag. TRADE-OFF 3: COMPLEXITY. Added complexity: Multiple databases to maintain. Monitoring: Need to monitor Primary + 10 Replicas. Failover: Need automated failover process if Primary fails. Costs: 11 database instances (Primary + 10 Replicas) vs 1 → 11× cost. Decision: Worth it for 10× read performance improvement. TRADE-OFF 4: WRITE SCALABILITY NOT IMPROVED. Replication helps reads, NOT writes. All writes still go to single Primary (1K writes/sec). If writes increase to 10K writes/sec, Primary becomes bottleneck. Solution for future: Shard database (split by user_id) if writes increase. Current: Not needed (writes are low). COST ANALYSIS: Before: 1 database at $500/month. After: 1 Primary + 10 Replicas at $5,500/month. Increase: 11× cost for 10× read capacity. Per-read cost: Actually cheaper (spreading load). Business value: Fast response times → better user experience → more users. Decision: ROI positive if users value speed. ALTERNATIVE CONSIDERED: CACHING (REDIS). Could we use Redis instead of replicas? Yes, but: Redis: Cache hot data (e.g., top 1000 products). Good for: 80% of reads hit cache → Reduce DB load 80%. Bad for: Remaining 20% still hits database → May still be overloaded. Cache invalidation: Complexity when data changes. Verdict: Use BOTH: Redis cache (reduce load), Read replicas (handle remaining load). REAL-WORLD EXAMPLE: REDDIT. Before: Single PostgreSQL database, slow during peak hours. After: 1 Primary + 50 Read Replicas, Redis cache. Result: Handles 50M+ users, fast response times. FINAL RECOMMENDATION: Add 10 Read Replicas with async replication. Route reads to replicas via load balancer. Implement read-your-own-writes for consistency. Monitor replication lag actively. Add Redis cache for hot data (future optimization). Expected outcome: Response times drop from 2-3 seconds to 50-200ms, Handle 100K reads/sec easily, 10× capacity for growth.',
          keyPoints: [
            'Read-heavy workload: add read replicas (10 replicas for 100K reads/sec)',
            'Primary handles writes, replicas handle reads via load balancer',
            'Trade-off: eventual consistency (replication lag 1-5 seconds)',
            'Mitigation: read-your-own-writes, monitor lag, remove lagging replicas',
            'Cost: 11× infrastructure cost, but 10× read capacity (worth it for UX)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between synchronous and asynchronous replication. When would you use each? What are the risks of asynchronous replication?',
          sampleAnswer:
            'SYNCHRONOUS VS ASYNCHRONOUS REPLICATION: SYNCHRONOUS REPLICATION: How it works: Write confirmed ONLY after data written to Primary AND at least one Replica. Flow: (1) App writes to Primary. (2) Primary writes to disk. (3) Primary sends data to Replica. (4) Replica writes to disk. (5) Replica sends ACK to Primary. (6) Primary confirms write to App. Timing: Write latency = Primary write + Network round-trip + Replica write. Example: Primary write 5ms + Network 2ms + Replica write 5ms = 12ms total. Guarantees: STRONG CONSISTENCY: Replica always has same data as Primary. NO DATA LOSS: If Primary crashes after confirming write, data exists on Replica. DURABILITY: Data persisted on multiple machines before confirming. WHEN TO USE SYNCHRONOUS: Banking/Financial: Money transfers (data loss = customer money lost). Regulatory compliance: Healthcare records (HIPAA), financial records (SOX). Critical business data: Orders, transactions, user accounts. When: Correctness > Performance. PROS OF SYNCHRONOUS: Data safety: Zero data loss guarantee. Immediate failover: Replica ready to take over instantly. Strong consistency: Reads from replica always return latest data. Predictable: No replication lag. CONS OF SYNCHRONOUS: SLOWER WRITES: 2-3× slower than async (network latency). If Primary → Replica latency is 50ms: Each write takes extra 50ms. At 1000 writes/sec: 50ms per write is noticeable. AVAILABILITY RISK: If Replica is down or slow, writes block or fail. Network partition between Primary and Replica: Writes cannot complete. High availability hurt by synchronous replication. GEOGRAPHIC LIMITATIONS: Cross-region replication (US → EU): 100ms latency. Every write takes extra 100ms (unacceptable for most apps). Synchronous replication works only within same datacenter. ASYNCHRONOUS REPLICATION: How it works: Write confirmed IMMEDIATELY after data written to Primary. Replica updated later (in background). Flow: (1) App writes to Primary. (2) Primary writes to disk. (3) Primary confirms write to App IMMEDIATELY. (4) Background process sends data to Replica (async). (5) Replica eventually writes to disk. Timing: Write latency = Primary write only (5ms). Replication happens in background (user doesn\'t wait). Example: Primary write 5ms → User gets confirmation. Replica catches up 1-5 seconds later (user doesn\'t notice). Guarantees: EVENTUAL CONSISTENCY: Replica will eventually have data (not immediately). DATA LOSS POSSIBLE: If Primary crashes before replicating, recent writes lost. FAST WRITES: No wait for Replica. WHEN TO USE ASYNCHRONOUS: Most web applications: Social media (Twitter, Facebook, Instagram). E-commerce (Amazon, eBay). SaaS applications. When: Performance > Absolute consistency. Non-critical data: User preferences, caching, analytics. Read-heavy workloads: Replicas handle reads (slight staleness OK). Cross-region replication: Geographic distribution (async required for reasonable performance). PROS OF ASYNCHRONOUS: FAST WRITES: No replication overhead (2-3× faster than sync). PRIMARY AVAILABILITY: Writes succeed even if Replica down. CROSS-REGION: Works globally (US → EU replication feasible). SIMPLE: No complex coordination between nodes. CONS OF ASYNCHRONOUS: DATA LOSS RISK: If Primary fails, recent writes (last 1-5 seconds) lost. Example: User posts comment, Primary crashes 2 seconds later (before replicating) → Comment lost. Mitigation: Use battery-backed write cache, frequent checkpoints. REPLICATION LAG: Replica may be 1-5 seconds behind (or minutes under high load). User posts comment, refreshes, comment missing (lag) → Confusing UX. Mitigation: Read-your-own-writes pattern. STALE READS: Reading from Replica returns old data. Example: User updates profile, reads from Replica, sees old profile. Mitigation: Route reads to Primary for short time after write. UNPREDICTABLE: Replication lag varies with load. Normal: 1-5 seconds. High load: 10-60 seconds. Mitigation: Monitor lag, remove lagging Replicas from rotation. COMPARISON TABLE: Aspect / Synchronous / Asynchronous: Write Latency: Slow (2-3× baseline) / Fast (baseline). Data Loss Risk: None (guaranteed safe) / Possible (last few seconds). Consistency: Strong (Replica = Primary) / Eventual (Replica lags). Availability: Lower (Replica failure affects writes) / Higher (writes succeed even if Replica down). Geographic: Same datacenter only / Works cross-region. Use Case: Banking, critical data / Most web apps. RISKS OF ASYNCHRONOUS REPLICATION: RISK 1: DATA LOSS. Scenario: User submits form, Primary confirms, crashes before replicating. Result: Form data lost (user needs to resubmit). Frequency: Rare (Primary failures uncommon), but impactful. Mitigation: (1) Use semi-sync for critical tables (wait for 1 Replica). (2) Frequent checkpoints (reduce window of data loss). (3) Accept risk for non-critical data. RISK 2: USER CONFUSION FROM LAG. Scenario: User posts comment, refreshes page, comment missing (lag 3 seconds). Result: User thinks comment failed, posts again (duplicate). Mitigation: (1) Read-your-own-writes: Route user\'s reads to Primary for 5-10 seconds after write. (2) Show "saving..." then "saved" indicator (set expectations). (3) Sticky sessions: Route user to same Replica (lag affects everyone equally). RISK 3: REPLICATION LAG SPIKE. Scenario: Traffic spike → Primary overloaded → Replication slows → Lag increases to 60 seconds. Result: Reads return very stale data (1 minute old). Mitigation: (1) Monitor lag: Alert if >10 seconds. (2) Remove lagging Replicas from load balancer. (3) Auto-scaling: Add more Replicas when lag increases. RISK 4: FAILOVER DATA LOSS. Scenario: Primary crashes, Replica has not caught up → Lost writes. Example: Primary fails, Replica is 5 seconds behind → Last 5 seconds of writes lost (could be hundreds or thousands of transactions). Mitigation: (1) Use semi-sync replication (at least 1 Replica always up-to-date). (2) Accept data loss for non-critical systems. (3) Application-level tracking: Retry failed writes. SEMI-SYNCHRONOUS REPLICATION (MIDDLE GROUND): How it works: Wait for at least ONE Replica to confirm (not all). Flow: Primary → Replica1 (wait for ACK) + Replica2, 3, 4... (async). Benefits: Faster than full sync (wait for 1 Replica, not all). Safer than async (at least 1 Replica has data). Trade-off: 1 Replica may still lag (eventual consistency for reads from lagging Replicas). Use case: Production systems needing balance of speed and safety. REAL-WORLD EXAMPLES: SYNCHRONOUS: Banking: CitiBank, Chase (financial transactions). Healthcare: Epic, Cerner (patient records). ASYNCHRONOUS: Social Media: Facebook, Twitter, Instagram (posts, likes, comments). E-commerce: Amazon (product browsing, reviews). SaaS: Slack, Salesforce (messages, CRM data). SEMI-SYNC: MySQL semi-sync: Wait for 1 Replica in same datacenter, async to cross-region Replicas. PostgreSQL synchronous_commit = remote_apply: Similar to semi-sync. FINAL RECOMMENDATION: DEFAULT: Asynchronous replication (fast, works globally, suitable for most apps). UPGRADE: Semi-sync for critical data (balance speed and safety). SPECIAL: Sync only for absolutely critical data (banking, financial, compliance). Trade-off awareness: Understand consistency vs performance trade-off. Monitor always: Track replication lag regardless of mode.',
          keyPoints: [
            'Synchronous: wait for replica ACK (slow, strong consistency, no data loss)',
            'Asynchronous: confirm immediately (fast, eventual consistency, data loss risk)',
            'Use sync for: banking, financial, critical data (correctness > performance)',
            'Use async for: most web apps, social media, e-commerce (performance > immediate consistency)',
            'Risk mitigation: monitor lag, read-your-own-writes, semi-sync for balance',
          ],
        },
        {
          id: 'q3',
          question:
            'Your primary database just crashed. Walk through the failover process step-by-step. What challenges might you encounter? How do you prevent split-brain?',
          sampleAnswer:
            "FAILOVER PROCESS - STEP BY STEP: STEP 1: DETECTION (10-30 seconds). What happens: Monitoring system detects Primary is down. Detection methods: (1) Heartbeat failure: Primary stops sending heartbeat (every 1-2 seconds). (2) Health check timeout: HTTP /health endpoint not responding. (3) Connection failure: Cannot establish TCP connection to Primary. (4) Quorum: Majority of nodes agree Primary is down. Timing: Detection threshold: 3 consecutive failures (to avoid false positives). If heartbeat every 2 seconds, 3 failures = 6 seconds minimum. Add network delays, consensus: 10-30 seconds total. Challenges: False positive: Network glitch makes Primary appear down (but it's not). Solution: Require multiple failed health checks before declaring Primary down. Slow detection: Longer detection = longer downtime. Solution: Reduce heartbeat interval (but increases network overhead). STEP 2: CHOOSE NEW PRIMARY (ELECTION) (5-10 seconds). What happens: Select which Replica to promote to new Primary. Selection criteria: (1) Most up-to-date Replica (highest log position). (2) Lowest replication lag. (3) Geographic location (same datacenter as old Primary preferred). (4) Manual preference (priority weights). Algorithm: Raft consensus or Paxos for distributed election. MongoDB: Replica set election (highest priority, most recent oplog). MySQL: Manual selection or automated (Orchestrator, ProxySQL). Timing: Consensus algorithm: 5-10 seconds for cluster to agree. Challenges: No clear winner: Multiple Replicas have same log position. Solution: Use tie-breaker (node ID, datacenter, priority). Network partition: Split-brain risk (see below). STEP 3: PROMOTE REPLICA TO PRIMARY (10-20 seconds). What happens: Selected Replica transitions from read-only to read-write mode. Actions: (1) Stop replication: Replica stops receiving updates from old Primary. (2) Enable writes: Change database configuration to accept writes. (3) Update metadata: Mark node as Primary in cluster state. (4) Start replication: New Primary starts sending updates to other Replicas. Timing: Configuration change + restart: 10-20 seconds. Challenges: Incomplete replication: New Primary may be missing recent writes (if async replication). Example: Old Primary had 1000 writes, Replica only received 990 → 10 writes lost. Mitigation: Use semi-sync replication (at least 1 Replica always up-to-date). Accept data loss for async replication. STEP 4: RECONFIGURE APPLICATION (10-30 seconds). What happens: Update application to write to new Primary instead of old Primary. Methods: (1) DNS update: Change DNS entry to point to new Primary IP. (2) Load balancer: Update load balancer config to route to new Primary. (3) Service discovery: Update service registry (Consul, etcd, ZooKeeper). (4) Connection pool refresh: Application refreshes database connections. Timing: DNS: 30-60 seconds (DNS TTL). Load balancer: 5-10 seconds. Service discovery: 5-10 seconds (fastest). Challenges: DNS caching: Clients may cache old DNS (stale) for minutes. Solution: Low DNS TTL (10-30 seconds), but increases DNS query load. Connection pool: Existing connections to old Primary need to close/reconnect. Solution: Connection pool health checks detect failed connections quickly. Split writes: Some app instances write to old Primary, others to new Primary → Data divergence! Solution: Fence old Primary (make it unable to accept writes). STEP 5: FENCE OLD PRIMARY (PREVENT SPLIT-BRAIN) (immediate). What happens: Ensure old Primary cannot accept writes after being declared dead. Fencing methods: (1) STONITH (Shoot The Other Node In The Head): Physically power off old Primary via remote management (IPMI). (2) Network isolation: Block old Primary's network access. (3) Kill process: Force-stop database process on old Primary. (4) Revoke access: Remove old Primary's write permissions at storage level. (5) Epoch numbers: Use epoch/term numbers to reject writes from old Primary. Why critical: Prevents split-brain scenario (see below). Timing: Should happen BEFORE or simultaneously with promotion. Challenges: Old Primary may be network-isolated (can't reach it to fence). Solution: Fencing must be enforceable without network access (STONITH). STEP 6: OLD PRIMARY RECOVERY (when it comes back). What happens: Old Primary recovers, needs to rejoin as Replica. Actions: (1) Detect state: Old Primary discovers it's no longer Primary. (2) Discard divergent writes: Rollback any writes accepted during isolation (if split-brain occurred). (3) Resync data: Catch up with new Primary. (4) Join as Replica: Start replicating from new Primary. Timing: Data resync: Depends on divergence (minutes to hours for large divergence). Challenges: Data conflict: Old Primary accepted writes during split-brain. Solution: Discard old Primary's writes (data loss), or manual reconciliation. TOTAL FAILOVER TIME: Detection: 10-30 seconds. Election: 5-10 seconds. Promotion: 10-20 seconds. Reconfiguration: 10-30 seconds. Total: 35-90 seconds downtime (automated failover). Manual failover: Minutes to hours (human intervention). CHALLENGES ENCOUNTERED: CHALLENGE 1: SPLIT-BRAIN. Problem: Network partition isolates old Primary from cluster, but old Primary is still running and accepting writes. Scenario: (1) Network partition: Old Primary can't reach cluster. (2) Cluster thinks Primary is down, promotes Replica to new Primary. (3) Old Primary doesn't know it was demoted, keeps accepting writes. (4) Now TWO Primaries accepting writes → DATA DIVERGENCE! Impact: Write 1 goes to old Primary: User A updates account balance to $100. Write 2 goes to new Primary: User B updates account balance to $200. When network heals: Which is correct? $100 or $200? Both? Neither? Data integrity compromised. SOLUTION: FENCING (Prevent old Primary from accepting writes). PREVENTING SPLIT-BRAIN: SOLUTION 1: FENCING WITH STONITH. How: Use remote management to physically power off old Primary. Example: IPMI, iLO, iDRAC (out-of-band management). Pros: Guaranteed to work (physical power off). Cons: Requires special hardware support. SOLUTION 2: QUORUM WITH ODD NUMBER OF NODES. How: Require majority (quorum) to accept writes. Example: 5-node cluster, need 3 nodes to agree (majority). If network partition: 3 nodes on one side, 2 on other. Side with 3 nodes can accept writes (has quorum). Side with 2 nodes cannot accept writes (no quorum). Result: Only one side can be Primary. Pros: Mathematically prevents split-brain. Cons: Requires odd number of nodes (3, 5, 7). SOLUTION 3: EPOCH NUMBERS (FENCING TOKENS). How: Each Primary term has an epoch number (increments on each failover). New Primary: epoch = N + 1. Old Primary: epoch = N (stale). Storage layer: Only accept writes from highest epoch. Result: Old Primary's writes rejected (epoch too old). Pros: Elegant, no physical fencing needed. Cons: Requires storage layer support. SOLUTION 4: WITNESS NODE (TIE-BREAKER). How: Deploy lightweight witness node (doesn't store data, just participates in quorum). Example: 2 datacenters with 1 database each + 1 witness. DC1: Primary + Witness (2 nodes). DC2: Replica (1 node). If DC1 Primary fails: DC2 Replica can't get quorum (1 out of 3) → Can't promote. If DC2 Replica fails: DC1 Primary keeps working (2 out of 3 quorum). Result: Prevents split-brain with even number of databases. REAL-WORLD EXAMPLES: MONGODB REPLICA SETS: Automatic failover: Election takes 10-12 seconds. Fencing: Quorum-based (majority must agree). Split-brain prevention: Odd number of nodes required (3, 5, 7). MYSQL WITH ORCHESTRATOR: Automatic failover: Detection + promotion + reconfiguration = 30-60 seconds. Fencing: Optional STONITH via API. Split-brain prevention: Manual (administrator must ensure old Primary stopped). POSTGRESQL WITH PATRONI: Automatic failover: Uses etcd/Consul for consensus, ~30 seconds. Fencing: Watchdog timer (old Primary auto-terminates if can't reach etcd). Split-brain prevention: Distributed lock in etcd. CASSANDRA: No failover needed: Multi-master (no single Primary). Every node can accept writes. Split-brain: Not applicable (eventual consistency, no Primary). BEST PRACTICES FOR FAILOVER: (1) Automate failover: Human reaction time too slow (minutes). (2) Test regularly: Chaos engineering (simulate failures monthly). (3) Monitor actively: Track replication lag, health checks. (4) Use quorum: Odd number of nodes prevents split-brain. (5) Implement fencing: STONITH or epoch numbers mandatory. (6) Document runbooks: Manual failover procedure if automation fails. (7) Accept data loss: If using async replication, last few seconds may be lost. FINAL RECOMMENDATION: Use automated failover with quorum-based consensus. Implement fencing (STONITH or epoch numbers) to prevent split-brain. Monitor failover time (alert if >90 seconds). Test failover monthly (chaos engineering). Accept 35-90 seconds downtime during failover (better than manual hours). Document manual failover procedure as backup.",
          keyPoints: [
            'Failover steps: Detect (10-30s) → Elect (5-10s) → Promote (10-20s) → Reconfigure (10-30s)',
            'Total downtime: 35-90 seconds for automated failover',
            'Split-brain: Two primaries accepting writes after network partition (data divergence)',
            'Prevention: Fencing (STONITH, quorum, epoch numbers) to ensure only one primary',
            'Best practice: Automated failover + quorum + fencing + regular testing',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'Your application has 1 primary database and 5 read replicas using asynchronous replication. A user writes data to the primary, then immediately reads from a replica. What might happen?',
          options: [
            'The user will always see their write immediately (strong consistency)',
            'The user might not see their write yet due to replication lag (eventual consistency)',
            'The write will fail because replicas are read-only',
            'The read will automatically be routed to the primary',
          ],
          correctAnswer: 1,
          explanation:
            "With asynchronous replication, the user might not see their write immediately (replication lag 1-5 seconds typical). The write goes to primary and confirms immediately. Replication to replicas happens in background. If user reads from replica before replication completes, they see old data. This is eventual consistency. Solution: Read-your-own-writes pattern (route user's reads to primary for 5-10 seconds after they write).",
        },
        {
          id: 'mc2',
          question:
            'What is the main advantage of synchronous replication over asynchronous replication?',
          options: [
            'Synchronous replication is faster',
            'Synchronous replication has no data loss risk (replica always has latest data)',
            'Synchronous replication works better across geographic regions',
            'Synchronous replication is simpler to implement',
          ],
          correctAnswer: 1,
          explanation:
            'Synchronous replication guarantees no data loss: write confirmed only after data written to both primary and replica. If primary fails immediately after write, data exists on replica. Trade-off: Slower writes (must wait for replica acknowledgment). Asynchronous is faster but risks losing recent writes if primary fails. Synchronous does NOT work well across regions (high latency). Used for critical data (banking, financial).',
        },
        {
          id: 'mc3',
          question:
            'You have 1 primary database handling 5K writes/sec. You add 10 read replicas. What is the write capacity now?',
          options: [
            "5K writes/sec (same as before, replicas don't help writes)",
            '50K writes/sec (10 replicas × 5K each)',
            '55K writes/sec (primary + 10 replicas)',
            '15K writes/sec (primary + average of replicas)',
          ],
          correctAnswer: 0,
          explanation:
            'Read replicas do NOT improve write capacity! All writes still go to single primary (5K writes/sec). Replicas handle READS only (read-only copies of data). To scale writes, need to shard database (split data across multiple primary databases). Replication helps: Read scaling (add replicas → more read capacity). Availability (if primary fails, promote replica). But NOT write scaling.',
        },
        {
          id: 'mc4',
          question:
            'What is split-brain in database replication, and why is it dangerous?',
          options: [
            'The database uses only half its memory (performance issue)',
            'Two databases think they are both primary and accept conflicting writes (data divergence)',
            'Replication lag causes reads to return half the data',
            'The primary database fails and no replica is promoted',
          ],
          correctAnswer: 1,
          explanation:
            'Split-brain: After network partition, both old primary and new primary accept writes → data divergence. Example: Old primary receives Write A (user balance = $100), new primary receives Write B (user balance = $200). When network heals: Which is correct? Data integrity compromised. Prevention: Fencing (STONITH, quorum, epoch numbers) ensures only ONE primary can accept writes. This is critical for data consistency.',
        },
        {
          id: 'mc5',
          question:
            'Your application uses asynchronous replication. The primary database crashes. What data might be lost?',
          options: [
            'No data lost (asynchronous is just as safe as synchronous)',
            'All data in the database is lost',
            "Only the most recent writes (last 1-5 seconds) that weren't replicated yet",
            'Half of the data is lost',
          ],
          correctAnswer: 2,
          explanation:
            'With async replication, only recent writes not yet replicated are lost (typically last 1-5 seconds). Example: Primary receives 1000 writes, replicates 990, crashes → 10 writes lost. Most data is safe (already replicated). This is acceptable for most web apps (social media, e-commerce). For critical data (banking), use synchronous or semi-sync replication to prevent data loss.',
        },
      ],
    },
    {
      id: 'message-queues',
      title: 'Message Queues & Async Processing',
      content: `Message queues enable asynchronous communication between services, decoupling producers and consumers for better scalability and reliability.

## What is a Message Queue?

**Definition**: A message queue is a form of asynchronous service-to-service communication where messages are stored in a queue until the consumer is ready to process them.

### **Why Use Message Queues?**

**Without Message Queues:**
- Services tightly coupled (caller waits for response)
- If consumer is slow, caller is blocked
- If consumer is down, caller fails
- No buffering (spike in traffic overwhelms consumer)
- Difficult to scale independently

**With Message Queues:**
- Services decoupled (producer sends message and continues)
- Asynchronous processing (consumer processes when ready)
- Buffering (queue absorbs traffic spikes)
- Resilience (retry failed messages automatically)
- Independent scaling (scale producers and consumers separately)

**Real-world**: Amazon uses SQS (Simple Queue Service) to process millions of orders per day asynchronously.

---

## Core Concepts

### **Producer**

**Role**: Creates and sends messages to the queue.

**Example**: Web server receives user signup request, sends message to queue: "New user: email@example.com"

**Characteristics:**
- Fire and forget (doesn't wait for processing)
- Fast response to user
- Can continue handling other requests

---

### **Consumer**

**Role**: Reads messages from queue and processes them.

**Example**: Background worker reads "New user" message, sends welcome email, creates database record.

**Characteristics:**
- Pulls messages from queue (or queue pushes to consumer)
- Processes at its own pace
- Can scale independently (add more consumers for high load)

---

### **Queue**

**Role**: Stores messages between producer and consumer.

**Characteristics:**
- FIFO (First In First Out) - typically
- Persistent (messages not lost if consumer crashes)
- Durable (survives broker restarts)
- Configurable retention (messages expire after N days)

---

### **Message**

**Structure:**
- **Body**: Actual data (JSON, XML, binary)
- **Attributes**: Metadata (timestamp, message ID, priority)
- **Headers**: Routing information

**Example message:**
- Message ID: "abc123"
- Timestamp: "2023-10-15T10:30:00Z"  
- Body: userId=12345, email="user@example.com", action="signup"

---

## Common Use Cases

### **1. Asynchronous Processing**

**Problem**: User uploads video (takes 10 minutes to transcode). Can't make user wait.

**Solution**: 
1. User uploads video → Server responds immediately "Upload successful"
2. Server sends message to queue: "Transcode video 12345"
3. Background worker processes transcoding
4. Notify user when done (via email or websocket)

**Benefit**: Fast user experience, heavy processing in background.

---

### **2. Load Leveling (Traffic Spike Handling)**

**Problem**: Black Friday sale → 100K orders/sec, but order processing service only handles 10K/sec.

**Solution**:
1. Orders sent to queue (accepts 100K/sec)
2. Queue buffers 100K messages
3. Order processing service consumes at 10K/sec
4. Queue drains over time (takes ~10 seconds)

**Benefit**: System doesn't crash, orders processed eventually.

---

### **3. Service Decoupling**

**Problem**: Payment service needs to notify: Email service, Analytics service, Inventory service. If any service is down, payment fails.

**Solution**:
1. Payment service sends "Payment completed" message to queue
2. Email, Analytics, Inventory each subscribe to queue
3. Each service processes message independently
4. If one service is down, others still work

**Benefit**: Services independent, no single point of failure.

---

### **4. Task Distribution (Work Queue)**

**Problem**: 1000 background jobs need processing (e.g., send 1000 emails). Single worker is slow.

**Solution**:
1. Producer sends 1000 messages to queue
2. 10 workers consume messages in parallel
3. Each worker processes 100 messages
4. Total time reduced 10×

**Benefit**: Horizontal scaling, faster processing.

---

## Queue vs Topic (Pub/Sub)

### **Queue (Point-to-Point)**

**Model**: One producer → Queue → One consumer (or consumer group)

**Characteristics:**
- Each message consumed by ONE consumer only
- Once consumed, message deleted from queue
- Used for task distribution

**Example**: Order processing queue
- Producer: Web server sends "Process order 123"
- Consumers: 5 workers compete for messages
- Each order processed by exactly one worker

**Use case**: Background jobs, task queues.

---

### **Topic (Publish/Subscribe)**

**Model**: One producer → Topic → Multiple subscribers

**Characteristics:**
- Each message delivered to ALL subscribers
- Message not deleted until all subscribers consume it
- Used for event broadcasting

**Example**: Payment completed event
- Producer: Payment service publishes "Payment completed for order 123"
- Subscribers: Email service, Analytics service, Inventory service
- All three receive and process the message independently

**Use case**: Event notifications, fan-out scenarios.

---

## Message Delivery Guarantees

### **At-Most-Once**

**Guarantee**: Message delivered 0 or 1 times (may be lost, never duplicated).

**How it works:**
1. Producer sends message (no acknowledgment required)
2. Message may be lost in transit
3. Consumer receives message, processes it (no acknowledgment)

**Use case**: Metrics, logs (occasional loss acceptable).

**Trade-off**: Fast, but unreliable.

---

### **At-Least-Once**

**Guarantee**: Message delivered 1 or more times (never lost, may be duplicated).

**How it works:**
1. Producer sends message, waits for acknowledgment
2. If no ack, producer retries (may result in duplicate)
3. Consumer receives message, processes it, sends acknowledgment
4. If consumer crashes before ack, message redelivered

**Use case**: Most common (order processing, emails).

**Trade-off**: Reliable, but consumer must handle duplicates (idempotency).

**Idempotency**: Processing same message multiple times has same effect as processing once.

**Example**: Email service tracks sent emails by message ID, skips if already sent.

---

### **Exactly-Once**

**Guarantee**: Message delivered exactly 1 time (never lost, never duplicated).

**How it works:**
- Complex: Requires distributed transactions (2-phase commit) or deduplication
- Kafka: Transactional producer + idempotent consumer
- Very expensive in terms of performance

**Use case**: Financial transactions (critical correctness).

**Trade-off**: Slow, complex, but guaranteed correctness.

---

## Message Ordering

### **FIFO (First In First Out)**

**Guarantee**: Messages processed in order sent.

**Use case**: Order status updates (must process "Order created" before "Order shipped").

**Implementation:**
- Single consumer (parallel consumers break ordering)
- Or: Partition by key (messages with same key go to same consumer)

**Example**: User actions
- Message 1: User created account
- Message 2: User updated profile
- Message 3: User deleted account
- Must process in order!

**Challenge**: Single consumer = no parallelism (slower).

---

### **Unordered**

**Guarantee**: No order guarantee (messages may be processed out of order).

**Use case**: Independent tasks (send emails, no order required).

**Benefit**: High parallelism (many consumers).

**Example**: Email notifications
- 1000 "Welcome email" messages
- Order doesn't matter
- 10 workers process in parallel

---

## Handling Failures

### **Dead Letter Queue (DLQ)**

**Problem**: Message processing fails repeatedly (bad data, bug, external service down).

**Solution**: After N retries, move message to Dead Letter Queue.

**Flow:**
1. Consumer tries to process message
2. Processing fails (exception)
3. Message returned to queue
4. Retry 3 times (configurable)
5. After 3 failures, move to DLQ
6. Alert operations team
7. Manually inspect and fix

**Example**: Email service fails to send (invalid email address). After 3 retries, move to DLQ for manual review.

**Configuration:**
- Max retries: 3
- Retry backoff: Exponential (1s, 2s, 4s)
- DLQ retention: 14 days

---

### **Exponential Backoff**

**Problem**: Service temporarily down. Retrying immediately overwhelms it.

**Solution**: Increase retry delay exponentially.

**Example:**
- Retry 1: Wait 1 second
- Retry 2: Wait 2 seconds
- Retry 3: Wait 4 seconds
- Retry 4: Wait 8 seconds
- Max: 60 seconds

**Benefit**: Gives service time to recover, reduces load.

---

## Popular Message Queue Systems

### **RabbitMQ**

**Type**: Traditional message broker

**Features:**
- Supports queues and topics
- AMQP protocol
- Good performance (~50K messages/sec)
- Easy to set up
- Complex routing (exchange types: direct, topic, fanout)

**Use case**: General-purpose message queue.

**Pros:**
- Mature, reliable
- Good monitoring (management UI)
- Flexible routing

**Cons:**
- Single point of failure (needs clustering)
- Lower throughput than Kafka

---

### **Apache Kafka**

**Type**: Distributed streaming platform

**Features:**
- Extremely high throughput (millions of messages/sec)
- Distributed (scales horizontally)
- Persistent log (messages retained for days/weeks)
- Replay messages (consumers can rewind)
- Partitioning for parallelism

**Use case**: High-throughput event streaming, log aggregation.

**Pros:**
- Massive scale
- Durability (messages persisted to disk)
- Replay capability

**Cons:**
- Complex to set up and operate
- Overkill for simple use cases

---

### **AWS SQS**

**Type**: Managed cloud message queue

**Features:**
- Fully managed (no servers)
- Auto-scaling
- At-least-once delivery
- FIFO queues available

**Use case**: Cloud-native applications.

**Pros:**
- Zero maintenance
- Pay-per-use
- Integrates with AWS services

**Cons:**
- Vendor lock-in
- Limited features vs RabbitMQ/Kafka

---

### **Redis (with Pub/Sub or Streams)**

**Type**: In-memory data store with messaging

**Features:**
- Very fast (in-memory)
- Simple pub/sub
- Streams (similar to Kafka, added in Redis 5.0)

**Use case**: Real-time messaging, lightweight queues.

**Pros:**
- Extremely fast
- Simple to set up
- Multi-purpose (cache + queue)

**Cons:**
- Not persistent by default (messages may be lost)
- Limited features vs dedicated message queues

---

## Comparison Table

**RabbitMQ / Kafka / AWS SQS / Redis:**

**Throughput**: 50K msg/sec / Millions msg/sec / 300K msg/sec / Very high

**Durability**: High / Very high / High / Low (unless configured)

**Ordering**: FIFO per queue / Per partition / FIFO queues / Pub/sub unordered

**Setup**: Medium / Hard / Easy (managed) / Easy

**Use case**: General / High-throughput streaming / Cloud-native / Real-time, lightweight

---

## Best Practices

**1. Idempotent Consumers**

Consumer must handle duplicate messages gracefully.

**Example**: Email service checks if email already sent before sending again.

**Implementation**: Track processed message IDs in database or cache.

---

**2. Monitor Queue Depth**

Track number of messages in queue.

**Alert if**: Queue depth growing (consumers too slow or down).

**Metrics**: Messages in queue, consumer lag, processing rate.

---

**3. Set Message TTL**

Messages expire after N hours/days.

**Why**: Prevent old messages from being processed (e.g., "Send email about sale" after sale ended).

**Configuration**: TTL = 24 hours for time-sensitive tasks.

---

**4. Use Dead Letter Queues**

Move failed messages to DLQ after retries.

**Why**: Prevent bad messages from blocking queue forever.

**Operations**: Monitor DLQ size, alert if growing.

---

**5. Partition for Parallelism**

Use multiple queues or partitions for parallel processing.

**Example**: 10 queues, 10 consumers (one per queue) for 10× throughput.

**Trade-off**: Ordering only guaranteed within partition.

---

## Real-World Examples

### **Uber**

**Use case**: Ride matching, surge pricing calculations.

**System**: Kafka for high-throughput event streaming.

**Scale**: Millions of events per second (GPS updates, ride requests).

---

### **Netflix**

**Use case**: Video encoding, recommendations.

**System**: AWS SQS for task distribution.

**Scale**: Thousands of workers processing millions of encoding jobs.

---

### **Slack**

**Use case**: Message delivery, notifications.

**System**: Custom message queue (similar to Kafka).

**Scale**: Billions of messages per day.

---

## Interview Tips

### **Common Questions:**

**Q: "Design a system to send 1 million emails asynchronously."**

✅ Good answer: "Use message queue:
1. Web server receives request to send emails
2. Produce 1M messages to queue (each message = one email)
3. 100 worker instances consume messages in parallel
4. Each worker sends 10K emails (1M / 100)
5. Use at-least-once delivery (retry failures)
6. Implement idempotency (track sent emails)
7. DLQ for emails that fail after retries (invalid addresses)
8. Monitor queue depth and worker health"

**Q: "What happens if message queue goes down?"**

✅ Good answer: "Impact:
- Producers can't send messages (fail or buffer locally)
- Consumers can't receive messages (processing stops)

Mitigation:
1. Use managed service (AWS SQS) with built-in HA
2. Run RabbitMQ/Kafka in cluster (multiple brokers)
3. Producer-side buffering (local disk queue)
4. Fallback to synchronous processing temporarily
5. Monitor queue health, alert on failures"

**Q: "Queue vs Database for task storage?"**

✅ Good answer: "Queue advantages:
- Optimized for FIFO, fast enqueue/dequeue
- Built-in retry, DLQ
- At-least-once delivery guarantees
- Better for high-throughput (millions/sec)

Database advantages:
- Queryable (find tasks by status, date)
- Transactions (atomicity)
- Complex filtering
- Persistent (never lose tasks)

Use queue for: Simple task distribution, high throughput.
Use database for: Complex querying, audit requirements."

---

## Key Takeaways

1. **Message queues decouple services** for better scalability and resilience
2. **Asynchronous processing** improves user experience (fast responses)
3. **Load leveling** absorbs traffic spikes, prevents system overload
4. **At-least-once delivery** is most common (consumers must be idempotent)
5. **Dead Letter Queue** handles failed messages after retries
6. **Kafka** for high throughput, **RabbitMQ** for general use, **SQS** for cloud-native
7. **Monitor queue depth** to detect consumer issues early`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your e-commerce site receives 10K orders/sec during Black Friday, but your order processing service can only handle 1K orders/sec. Orders are currently failing. Design a solution using message queues.',
          sampleAnswer:
            'SOLUTION: MESSAGE QUEUE WITH LOAD LEVELING. PROBLEM ANALYSIS: Without queue: 10K orders/sec → Order processing service (capacity 1K/sec) → 9K orders/sec FAIL (90% failure rate). Users see errors, orders lost. PROPOSED ARCHITECTURE: Clients → Web servers → Message Queue (SQS/Kafka) → Order processing workers. IMPLEMENTATION: STEP 1: INTRODUCE MESSAGE QUEUE. When order received: Web server validates order (fast, <10ms). Web server sends message to queue: {"orderId": 12345, "userId": 789, ...}. Web server responds to user immediately: "Order received! Processing...". No waiting for order processing. STEP 2: CONFIGURE QUEUE. Queue type: AWS SQS (managed, no ops overhead). Queue capacity: Unlimited (can buffer millions of messages). Message retention: 4 days (if processing delayed). Delivery: At-least-once (no orders lost). STEP 3: ORDER PROCESSING WORKERS. Deploy workers to consume from queue. Workers: 10 instances, each handles 100 orders/sec = 1K orders/sec total. Workers: Pull messages from queue at their own pace. Workers: Process order (validate payment, update inventory, send confirmation email). Workers: Acknowledge message after successful processing. STEP 4: HANDLE PROCESSING. Flow: (1) Queue receives 10K orders/sec (no problem, just buffers). (2) Queue depth grows to 100K messages (10K/sec incoming - 1K/sec outgoing = 9K/sec accumulation). (3) Workers consume at 1K/sec. (4) After spike ends (e.g., 1 minute of 10K/sec), queue has 540K messages (9K/sec × 60sec = 540K). (5) Workers drain queue at 1K/sec: 540K messages / 1K per sec = 540 seconds = 9 minutes. Result: All orders processed within 9 minutes (acceptable for most e-commerce). BENEFITS: User experience: Instant response ("Order received"). No failures: Queue buffers all orders. Resilience: If workers crash, messages stay in queue (retry). Scalability: Can add more workers if queue depth too high. SCALING STRATEGY: Monitor queue depth in real-time. If queue depth > 100K: Auto-scale workers (add 10 more instances → 2K orders/sec). If queue depth < 10K: Scale down workers (save costs). Target: Keep processing within 5 minutes. FAILURE HANDLING: Worker crashes: Message not acknowledged → Returns to queue → Retried by another worker. Payment fails: Retry 3 times with exponential backoff (1s, 2s, 4s). After 3 failures: Move to Dead Letter Queue (DLQ). DLQ: Operations team reviews failed orders manually. IDEMPOTENCY: Problem: Message redelivered if worker crashes after processing but before acknowledging. Solution: Track processed order IDs in database. Before processing order: Check if order ID already processed. If yes, skip (already done). If no, process and mark as done. Result: Duplicate messages don\'t result in duplicate orders. MONITORING: Metrics: (1) Queue depth (messages in queue). (2) Message age (oldest message in queue). (3) Consumer lag (how far behind are workers). (4) Processing rate (orders/sec). (5) DLQ size (failed orders). Alerts: Queue depth > 500K (workers too slow). Message age > 1 hour (processing delayed). DLQ size > 100 (many failures). COST ANALYSIS: AWS SQS: $0.40 per million messages. Black Friday: 10K orders/sec × 3600 sec (1 hour) = 36M messages. Cost: 36M × $0.40 / 1M = $14.40 (cheap!). Workers: 10 × $0.10/hour (spot instances) = $1/hour. Total: $15.40 to handle 36M orders (negligible vs revenue). ALTERNATIVE CONSIDERED: Synchronous processing with more servers: Would need 10K orders/sec capacity = 100 instances running 24/7. Cost: 100 × $0.10/hour × 24 hours × 30 days = $7,200/month. Verdict: Queue approach much cheaper (workers scale down after spike). REAL-WORLD EXAMPLE: Amazon uses SQS to buffer orders during Prime Day. Queue absorbs traffic spikes (millions of orders). Workers process orders asynchronously. Users get instant confirmation, orders processed within minutes. FINAL RECOMMENDATION: Use AWS SQS (managed, reliable, cheap). Buffer orders in queue during spike. Workers consume at sustainable rate (1K/sec). Auto-scale workers based on queue depth. Implement idempotency (prevent duplicate orders). Monitor queue metrics and alert. Expected outcome: 100% orders processed, Zero failures, 9-minute processing time (acceptable).',
          keyPoints: [
            'Message queue buffers traffic spikes (10K orders/sec → queue → 1K orders/sec processing)',
            'Instant user response ("Order received"), processing happens asynchronously',
            'Auto-scale workers based on queue depth to reduce processing time',
            'Implement idempotency to handle message redelivery (track processed order IDs)',
            'Monitor queue depth and message age to detect processing delays',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between a Queue (point-to-point) and a Topic (pub/sub). Give a real-world scenario where you would use each.',
          sampleAnswer:
            'QUEUE (POINT-TO-POINT) VS TOPIC (PUB/SUB): QUEUE (POINT-TO-POINT): Model: One producer → Queue → One consumer (or consumer group). Each message consumed by exactly ONE consumer. Once consumed, message deleted from queue. Multiple consumers compete for messages (load distribution). EXAMPLE ARCHITECTURE: Producer: Web server sends messages. Queue: "email_tasks". Consumers: 5 email worker instances. Message: "Send welcome email to user@example.com". Flow: (1) Producer sends message to queue. (2) Consumer 1 pulls message from queue. (3) Consumer 1 processes message (sends email). (4) Consumer 1 acknowledges → Message deleted from queue. (5) Other consumers (2-5) never see this message. Result: Email sent exactly once (by one worker). USE CASE: TASK DISTRIBUTION (BACKGROUND JOBS). REAL-WORLD SCENARIO 1: IMAGE PROCESSING PIPELINE. Problem: Users upload images. Need to: resize image, generate thumbnails, optimize quality. Solution: Queue-based task distribution. Architecture: User uploads image → API server → Queue: "image_processing_tasks". Queue contains message: {"imageId": 123, "action": "process"}. 10 worker instances pull messages from queue. Each worker: Downloads image, processes it, uploads result, acknowledges message. Why queue (not topic)? Each image processed by ONE worker (avoid duplicate work). Load distributed evenly across 10 workers. Simple task distribution. TOPIC (PUBLISH/SUBSCRIBE): Model: One producer → Topic → Multiple subscribers. Each message delivered to ALL subscribers. Message not deleted until all subscribers consume it. Subscribers process message independently. EXAMPLE ARCHITECTURE: Producer: Payment service publishes messages. Topic: "payment_completed". Subscribers: (1) Email service (sends receipt). (2) Analytics service (tracks revenue). (3) Inventory service (updates stock). (4) Fraud detection service (analyzes transaction). Message: {"orderId": 456, "amount": 99.99, "userId": 789}. Flow: (1) Producer publishes message to topic. (2) Message delivered to ALL 4 subscribers simultaneously. (3) Each subscriber processes message independently. (4) Email service sends receipt. (5) Analytics service logs revenue. (6) Inventory service decrements stock. (7) Fraud detection analyzes transaction. Result: All 4 actions happen (fan-out). USE CASE: EVENT BROADCASTING. REAL-WORLD SCENARIO 2: ORDER COMPLETION EVENT. Problem: When order completed, need to: (1) Send confirmation email to customer. (2) Send notification to seller. (3) Update analytics dashboard. (4) Trigger inventory restock if low. (5) Log to audit system. Solution: Topic-based pub/sub. Architecture: Order service publishes "OrderCompleted" event to topic. 5 services subscribe to topic. Each service receives event and acts independently. Why topic (not queue)? Multiple actions needed for same event. Services independent (email service down doesn\'t affect analytics). Easy to add new subscribers later (no code change to producer). COMPARISON TABLE: Aspect / Queue / Topic: Message delivery: One consumer / All subscribers. Message deletion: After consumed / After all consume. Use case: Task distribution / Event broadcasting. Consumers: Compete for messages / Each gets copy. Example: Background jobs / Notifications. WHEN TO USE QUEUE: Task distribution: Process 1000 emails (each sent once). Load balancing: 10 workers share workload. No duplication: Each task done exactly once. Example: Video encoding, data processing, email sending. WHEN TO USE TOPIC: Event notification: Payment completed → Notify multiple services. Fan-out: One event, multiple actions. Loosely coupled services: Subscribers independent. Example: User registration, order events, system alerts. HYBRID ARCHITECTURE (QUEUE + TOPIC): Many systems use both. Example: E-commerce order flow. STEP 1: TOPIC for order created event. Producer: Order service publishes "OrderCreated" to topic. Subscribers: Email service, Analytics service, Fraud detection service. Each service receives event independently. STEP 2: QUEUE for order processing tasks. After fraud check passes: Fraud service sends message to "order_fulfillment_queue". 10 fulfillment workers compete for messages. Each order fulfilled by one worker. Why hybrid? Order creation is an EVENT (multiple services need to know) → Topic. Order fulfillment is a TASK (done once by one worker) → Queue. KAFKA EXAMPLE: Kafka Topics: Act like pub/sub (multiple consumer groups). Kafka Consumer Groups: Act like queue (within group, one consumer per message). Kafka combines both models! Topic: "user_signups". Consumer Group 1 (Email service): One consumer in group processes message. Consumer Group 2 (Analytics service): One consumer in group processes same message. Result: Message delivered to both groups, but only one consumer per group. REAL-WORLD EXAMPLES: Queue: Netflix: Video encoding tasks (one worker per video). Uber: Ride matching tasks (one matcher per ride). Topic: Slack: Message events (multiple services notified). Twitter: Tweet events (timeline, notifications, analytics). MISTAKES TO AVOID: Using queue for events: If multiple actions needed, use topic (not queue). Example: Don\'t send order event to email queue and analytics queue separately (use one topic with 2 subscribers). Using topic for tasks: If only one action needed, use queue (not topic). Example: Don\'t use topic for "process image" task (wastes resources if multiple services process same image). FINAL RECOMMENDATION: Queue: For task distribution, background jobs, load balancing. Topic: For event broadcasting, notifications, loosely coupled services. Hybrid: Use both when system has events (topic) and tasks (queue).',
          keyPoints: [
            'Queue: each message consumed by ONE consumer (task distribution, background jobs)',
            'Topic: each message delivered to ALL subscribers (event broadcasting, notifications)',
            'Use queue for: image processing, email sending (task done once)',
            'Use topic for: order events, payment completed (multiple services notified)',
            'Hybrid: Use both (topic for events, queue for tasks) in same system',
          ],
        },
        {
          id: 'q3',
          question:
            'Your message consumer processes payments. Due to a bug, processing fails for all messages. They keep retrying, filling the queue. How would you handle this situation?',
          sampleAnswer:
            'HANDLING FAILING MESSAGE CONSUMER (POISON PILL SCENARIO): SCENARIO: Bug in payment consumer code (e.g., NullPointerException). Consumer tries to process message → Fails → Message returned to queue → Retry → Fails → Retry... Loop continues forever. Queue fills up with retried messages. New messages can\'t be processed (queue full). System effectively down. IMMEDIATE ACTIONS (INCIDENT RESPONSE): ACTION 1: STOP THE CONSUMERS (CIRCUIT BREAKER). Immediately stop all consumer instances. Why: Prevent infinite retry loop, stop queue from filling further. How: Kill consumer processes or set consumer count to 0 in auto-scaling group. Result: Messages stay in queue (safe), no more retries (queue stabilizes). ACTION 2: INVESTIGATE THE ISSUE. Examine logs: Find error message/stack trace. Identify root cause: NullPointerException in payment processing code. Example: Code assumed field always present, but message missing field. Reproduce: Test with sample message locally. ACTION 3: APPLY IMMEDIATE FIX. If simple fix: Deploy hotfix (e.g., add null check). If complex: Implement temporary workaround (e.g., skip invalid messages). Deploy fix to staging, test thoroughly. Deploy fix to production. ACTION 4: RESTART CONSUMERS. Start consumer instances again. Monitor closely: Check if messages processing successfully. Watch queue depth: Should decrease steadily. Watch error rate: Should be 0% (or very low). LONG-TERM SOLUTIONS (PREVENTING RECURRENCE): SOLUTION 1: DEAD LETTER QUEUE (DLQ). Configure DLQ for failed messages. Max retries: 3 attempts. Backoff: Exponential (1s, 2s, 4s). After 3 failures: Move message to DLQ (stop retrying). Benefits: Poison pill messages don\'t block queue. DLQ for manual inspection and fixing. Configuration: Main Queue: "payments_queue" (max retries: 3). DLQ: "payments_dlq" (holds failed messages). Workflow: (1) Consumer fails to process message. (2) Message returned to queue (attempt 1). (3) Consumer fails again (attempt 2). (4) Consumer fails again (attempt 3). (5) Message moved to DLQ (stop retrying). (6) Alert operations team: "DLQ size increased". (7) Team inspects DLQ, identifies issue, fixes it. (8) Reprocess DLQ messages after fix. SOLUTION 2: MESSAGE VALIDATION. Validate message before processing. If invalid: Log error, move to DLQ immediately (don\'t retry). If valid: Process normally. Code example: def process_payment(message): # Validate message structure. if not message.get("amount") or not message.get("userId"): logger.error(f"Invalid message: {message}"). move_to_dlq(message). return. # Process payment. process_payment_logic(message). Benefits: Invalid messages detected early (don\'t retry). Clear error logging (easier to debug). SOLUTION 3: CIRCUIT BREAKER. If consumer fails N times in a row, stop processing temporarily. Wait T seconds, then retry. If still failing, alert and stop. Code example: consecutive_failures = 0. MAX_FAILURES = 10. def process_message(message): global consecutive_failures. try: process_payment(message). consecutive_failures = 0  # Reset on success. except Exception as e: consecutive_failures += 1. if consecutive_failures >= MAX_FAILURES: logger.critical("Circuit breaker opened! Too many failures."). stop_consumer()  # Stop processing. alert_ops_team(). raise. Benefits: Prevents infinite retry loop. Automatic stop on repeated failures. Operations team alerted quickly. SOLUTION 4: MONITORING & ALERTING. Track consumer metrics: Messages processed/sec (should be steady). Error rate (should be <1%). Queue depth (should be low). DLQ size (should be 0 or small). Consumer lag (time between message sent and processed). Alerts: Error rate > 5% for 5 minutes → Page on-call engineer. DLQ size > 100 → Alert (many failed messages). Queue depth > 10,000 → Warning (consumers too slow or failing). Consumer lag > 1 hour → Warning (processing delayed). Benefits: Early detection of issues. Fast response time (reduce downtime). SOLUTION 5: IDEMPOTENCY. Make consumer idempotent (handle duplicate messages). Track processed message IDs in database. Before processing: Check if message already processed (skip if yes). After processing: Mark message as processed. Why important for retries: If consumer processes payment, then crashes before acknowledging message, message redelivered. Without idempotency: User charged twice (bad!). With idempotency: Second processing skipped (user charged once). SOLUTION 6: GRADUAL ROLLOUT (CANARY DEPLOYMENT). When deploying new consumer code: Deploy to 1 instance first (canary). Monitor canary for 10 minutes (check error rate, queue depth). If healthy: Deploy to 10% of instances, monitor. If still healthy: Deploy to 100%. If any issues: Rollback canary, investigate. Benefits: Catch bugs early (affect small % of traffic). Fast rollback (only 1 instance affected). Prevent large-scale outages. SOLUTION 7: MESSAGE TTL (TIME TO LIVE). Set message expiration: Messages older than 24 hours deleted from queue. Why: If processing delayed significantly (e.g., consumer down for days), old messages may be obsolete. Example: "Send flash sale email" message from 3 days ago (sale over, don\'t send). Configuration: Message TTL: 24 hours. Result: Old messages auto-deleted (prevent processing stale data). RUNBOOK FOR FUTURE INCIDENTS: STEP 1: Detect issue (alert: high error rate or queue depth). STEP 2: Stop consumers (prevent further damage). STEP 3: Investigate logs (find root cause). STEP 4: Apply fix (hotfix or workaround). STEP 5: Test fix in staging. STEP 6: Restart consumers (1 instance first, then all). STEP 7: Monitor closely (error rate, queue depth). STEP 8: Inspect DLQ (reprocess valid messages after fix). STEP 9: Post-mortem (document incident, improve prevention). REAL-WORLD EXAMPLE: AWS: Customer had payment processor bug. Bug caused infinite retries (queue filled up). No DLQ configured (messages retried forever). Solution: Implemented DLQ, max retries = 3. Deployed circuit breaker (stop after 10 consecutive failures). Incident avoided in future deployments. FINAL RECOMMENDATION: Implement DLQ with max retries (3 attempts). Add circuit breaker (stop after 10 consecutive failures). Validate messages before processing (catch invalid data early). Monitor error rate and queue depth (alert on anomalies). Make consumer idempotent (handle duplicate messages). Test consumer code thoroughly before deployment (staging environment). Document runbook for handling consumer failures. Expected outcome: Future bugs caught early (circuit breaker stops processing). Failed messages isolated (DLQ), don\'t block queue. Fast incident resolution (runbook).',
          keyPoints: [
            'Immediate: Stop consumers to prevent infinite retry loop',
            'Dead Letter Queue (DLQ): Move failed messages after N retries (prevent blocking)',
            'Circuit breaker: Stop processing after N consecutive failures (alert ops team)',
            'Validation: Check message structure before processing (catch invalid data early)',
            'Monitoring: Track error rate, queue depth, DLQ size (alert on anomalies)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'Your web application sends 10K requests/sec to a payment processing service that can only handle 1K requests/sec. What will happen without a message queue?',
          options: [
            'The payment service will process all 10K requests/sec successfully',
            '9K requests/sec will fail or timeout (90% failure rate)',
            'The requests will be automatically queued by the network layer',
            'The web application will slow down to match the payment service',
          ],
          correctAnswer: 1,
          explanation:
            '9K requests/sec will fail or timeout (90% failure rate). Without a message queue, the payment service is overwhelmed: Can only process 1K/sec, remaining 9K/sec rejected or timeout. Users see errors, transactions lost. Solution: Add message queue between web app and payment service. Queue buffers requests, payment service consumes at its own pace (1K/sec). All 10K requests eventually processed (may take 10 seconds to drain queue).',
        },
        {
          id: 'mc2',
          question:
            'What is the main difference between "at-most-once" and "at-least-once" message delivery?',
          options: [
            'At-most-once is faster but may lose messages; at-least-once is slower but guarantees delivery',
            'At-most-once delivers to one consumer; at-least-once delivers to all consumers',
            'At-most-once uses queues; at-least-once uses topics',
            'At-most-once requires acknowledgments; at-least-once does not',
          ],
          correctAnswer: 0,
          explanation:
            'At-most-once: Fast, fire-and-forget, but message may be lost (no retries). At-least-once: Slower, with acknowledgments and retries, guarantees delivery but may deliver duplicates. At-least-once is most common in production (reliability > speed). Consumer must be idempotent (handle duplicates). At-most-once used for non-critical data (metrics, logs).',
        },
        {
          id: 'mc3',
          question: 'What is a Dead Letter Queue (DLQ), and when is it used?',
          options: [
            'A queue for messages that are processed successfully',
            'A queue for high-priority messages that need immediate processing',
            'A queue for messages that fail processing repeatedly after multiple retries',
            'A queue for messages that expired due to TTL',
          ],
          correctAnswer: 2,
          explanation:
            'Dead Letter Queue (DLQ) holds messages that fail processing after multiple retries. Example: Consumer tries to process message, fails 3 times (bad data, bug, external service down). After 3 failures, message moved to DLQ (stop retrying). Operations team inspects DLQ, fixes issue, reprocesses messages. Benefit: Prevents poison pill messages from blocking queue forever.',
        },
        {
          id: 'mc4',
          question: 'When should you use a Topic (pub/sub) instead of a Queue?',
          options: [
            'When you want each message processed by exactly one consumer (task distribution)',
            'When you want each message delivered to multiple subscribers (event broadcasting)',
            'When you need high throughput and low latency',
            'When messages need to be processed in order',
          ],
          correctAnswer: 1,
          explanation:
            'Use Topic (pub/sub) when one message needs to be delivered to multiple subscribers (event broadcasting). Example: "PaymentCompleted" event delivered to Email service, Analytics service, Inventory service. Each subscriber processes message independently. Use Queue when message should be processed by exactly one consumer (task distribution). Example: "Process video encoding" task consumed by one worker.',
        },
        {
          id: 'mc5',
          question:
            'Your message consumer must be idempotent. What does this mean?',
          options: [
            'The consumer must process messages in order',
            'The consumer must process messages very fast',
            'Processing the same message multiple times has the same effect as processing it once',
            'The consumer must acknowledge messages immediately',
          ],
          correctAnswer: 2,
          explanation:
            'Idempotent: Processing same message multiple times has same effect as processing once. Why needed: At-least-once delivery may deliver duplicates (if consumer crashes after processing but before acknowledging). Example: Payment consumer tracks processed message IDs. On duplicate, skips processing (payment already made). Non-idempotent example: Incrementing counter (duplicate message → counter incremented twice, wrong result).',
        },
      ],
    },
    {
      id: 'cdn',
      title: 'CDN (Content Delivery Network)',
      content: `A Content Delivery Network (CDN) is a geographically distributed network of servers that cache and deliver content to users from the nearest location, reducing latency and improving performance.

## What is a CDN?

**Definition**: A CDN is a network of edge servers strategically placed around the world that cache static content (images, videos, CSS, JavaScript) closer to end users.

### **Why Use a CDN?**

**Without CDN:**
- All users fetch content from origin server (e.g., US data center)
- Users in Asia experience high latency (200-300ms)
- Origin server handles all traffic (bandwidth expensive)
- Single point of failure

**With CDN:**
- Content cached on edge servers worldwide
- Users fetch from nearest edge server (10-50ms latency)
- Origin server handles less traffic (80-90% reduction)
- Distributed architecture (no single point of failure)

**Real-world**: Netflix uses CDN to deliver streaming video. 95% of traffic served from edge servers, not origin.

---

## How CDN Works

### **Basic Flow**

1. User requests image: example.com/logo.png
2. DNS resolves to nearest CDN edge server (not origin)
3. Edge server checks cache: Does it have logo.png?
4. If cached (HIT): Return image immediately (fast)
5. If not cached (MISS): Fetch from origin, cache it, return to user
6. Subsequent requests: Served from cache (no origin hit)

### **CDN Architecture**

**Components:**
- **Origin Server**: Your application server (where content originates)
- **Edge Servers**: CDN servers distributed globally (cache content)
- **PoPs (Points of Presence)**: Data centers with edge servers
- **DNS**: Routes users to nearest edge server

**Example topology:**
- Origin: US East (Virginia)
- Edge servers: 200+ locations worldwide
- User in Tokyo → Routes to Tokyo PoP (10ms latency vs 200ms to US)

---

## CDN Caching Strategies

### **Pull CDN (Origin Pull)**

**How it works**: Edge server pulls content from origin on first request (lazy loading).

**Flow:**
1. User requests file
2. Edge server: Cache miss
3. Edge server pulls from origin
4. Edge server caches and returns to user
5. Next user: Cache hit (served from edge)

**Pros:**
- Simple setup (no need to push content)
- Automatic (CDN handles everything)
- Only popular content cached (efficient)

**Cons:**
- First user experiences slow load (cache miss)
- Origin hit on every new file

**When to use**: Most common (dynamic content, websites).

---

### **Push CDN**

**How it works**: You proactively push content to edge servers before users request it.

**Flow:**
1. You upload files to CDN (via API or dashboard)
2. CDN distributes to all edge servers
3. User requests file
4. Edge server: Cache hit immediately (already there)

**Pros:**
- No cache misses (always cached)
- Predictable performance
- Control over what's cached

**Cons:**
- Manual effort (must push updates)
- Wastes cache space (unpopular content cached)
- Slower deploys (time to propagate to all edges)

**When to use**: Static sites, large files (videos), product launches.

---

## Cache Invalidation / Purging

**Problem**: You update logo.png on origin, but edge servers still serve old cached version.

### **Methods**

**1. TTL (Time To Live)**
- Set expiration time on cached content
- After TTL expires, edge refetches from origin
- Example: Cache images for 1 day (86400 seconds)

**2. Cache Purge (Invalidation)**
- Manually delete cached content from edge servers
- API call: "Purge /logo.png"
- All edge servers delete file
- Next request: Cache miss → Fetch from origin

**3. Versioned URLs**
- Change URL when content changes
- Old: /logo.png → New: /logo.png?v=2 or /logo-v2.png
- Edge servers see new URL → Cache miss → Fetch fresh version
- **Best practice** (no purge needed, instant update)

---

## CDN for Dynamic Content

**Challenge**: CDN typically caches static files. What about dynamic content (API responses, personalized pages)?

### **Solutions**

**1. Edge Computing (Serverless Functions)**
- Run code on edge servers (not just cache)
- Example: Cloudflare Workers, AWS Lambda@Edge
- Use case: A/B testing, authentication, personalization at edge

**2. Partial Caching**
- Cache page template (static)
- Fetch user-specific data via AJAX (dynamic)
- Hybrid approach

**3. Short TTL**
- Cache API responses for 1-5 seconds
- Reduces origin load significantly
- Slight staleness acceptable

---

## CDN Benefits

**1. Reduced Latency**
- Users fetch from nearest edge (10-50ms vs 200-300ms)
- Faster page loads → Better user experience

**2. Reduced Origin Load**
- 80-90% of requests served by CDN
- Origin handles only cache misses
- Lower bandwidth costs

**3. Improved Availability**
- Distributed architecture (no single point of failure)
- If origin down, cached content still served
- DDoS protection (CDN absorbs attack traffic)

**4. Scalability**
- CDN handles traffic spikes automatically
- No need to scale origin servers
- Example: Product launch, viral content

---

## Popular CDN Providers

### **Cloudflare**
- Free tier available
- Global network (200+ cities)
- DDoS protection included
- Best for: Small to medium websites

### **AWS CloudFront**
- Integrated with AWS ecosystem (S3, EC2)
- Pay-as-you-go pricing
- Lambda@Edge for edge computing
- Best for: AWS-native applications

### **Akamai**
- Largest CDN (300K+ servers)
- Enterprise-grade
- Expensive
- Best for: Large enterprises, media companies

### **Fastly**
- Real-time purging (instant cache invalidation)
- Edge computing (VCL scripting)
- Best for: High-performance apps, real-time updates

---

## CDN vs Origin Serving

**Comparison:**

**Aspect / Origin / CDN:**
- Latency: High (200-300ms globally) / Low (10-50ms)
- Bandwidth cost: High (all traffic hits origin) / Low (80-90% reduction)
- Availability: Single point of failure / Distributed (high availability)
- Scalability: Limited by origin capacity / Scales automatically
- Cost: Server + bandwidth costs / CDN fees (often cheaper at scale)

**When NOT to use CDN:**
- Very low traffic (<1000 visitors/day)
- All users in same region as origin
- Highly dynamic content (no caching benefit)

---

## Real-World Examples

### **Netflix**
- 95% of traffic served from CDN (Open Connect)
- Origin only for new/rare content
- Result: Massive bandwidth savings, fast streaming globally

### **Shopify**
- Uses CloudFront CDN for all stores
- Images, CSS, JS cached on edge
- Result: Fast page loads for global customers

### **Wikipedia**
- Uses Varnish CDN (self-hosted)
- Caches article pages
- Result: Handles 20 billion page views/month with minimal origin load

---

## Key Takeaways

1. **CDN caches content on edge servers worldwide** for low latency
2. **Pull CDN** most common (lazy loading, automatic)
3. **TTL + versioned URLs** for cache invalidation
4. **80-90% origin load reduction** typical with CDN
5. **Edge computing** enables dynamic content at CDN layer
6. **CDN provides DDoS protection** as side benefit`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your website serves users globally. Users in Asia complain about slow image loading (3-5 seconds). Images are served from your origin server in US-East. Propose a solution.',
          sampleAnswer:
            'SOLUTION: IMPLEMENT CDN. PROBLEM ANALYSIS: High latency for Asian users: US-East to Asia: ~250ms network latency (round-trip). Loading 10 images: 10 × 250ms = 2.5 seconds (network time alone). Plus image download time: Total 3-5 seconds. Root cause: Geographic distance between origin and users. PROPOSED SOLUTION: CDN (Content Delivery Network). Setup: Integrate CDN (CloudFront, Cloudflare, Fastly). Point domain to CDN. Configure CDN to pull from origin. Deploy edge servers in Asia (Tokyo, Singapore, Mumbai). IMPLEMENTATION: Step 1: Sign up for CDN provider. Step 2: Configure origin: Origin URL: your-origin-server.com. Step 3: Update DNS: Old: images.yoursite.com → Origin IP. New: images.yoursite.com → CDN endpoint (d1234.cloudfront.net). Step 4: CDN routes users: Asian users → Asian edge servers. US users → US edge servers. BENEFITS: Latency reduction: Before: Asia → US-East: 250ms. After: Asia → Tokyo edge: 10-20ms. Result: 12× faster (250ms → 20ms). Page load time: Before: 3-5 seconds. After: 300-500ms. 10× improvement! Origin load: Before: Origin serves all image requests. After: CDN serves 90% (only cache misses hit origin). Bandwidth: 90% reduction in origin bandwidth (cost savings). COST ANALYSIS: CDN cost: ~$0.085 per GB (CloudFront Asia). 100 GB/month: $8.50. Origin bandwidth savings: Was paying $0.12 per GB origin bandwidth. Now: 10 GB origin (90% reduction): $1.20. Net: $8.50 (CDN) - $10.80 (savings) = Save $2.30/month. Plus: Faster UX (more users, more revenue). MONITORING: Track metrics: Cache hit rate (target: >90%). Latency by region (should be <50ms). Origin requests (should decrease 90%). Alert if: Cache hit rate <80% (investigate). Latency >100ms in any region.',
          keyPoints: [
            'CDN caches images on edge servers worldwide (reduce latency 12×)',
            'Asian users fetch from Asian edge servers (10-20ms vs 250ms)',
            '90% cache hit rate → 90% less origin bandwidth',
            'Page load time: 3-5 seconds → 300-500ms (10× improvement)',
            'Setup: Configure CDN, update DNS to CDN endpoint, monitor hit rate',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between Pull CDN and Push CDN. When would you use each?',
          sampleAnswer:
            "PULL CDN (ORIGIN PULL): How it works: Edge server fetches content from origin on first request (lazy loading). Flow: (1) User in Tokyo requests /logo.png. (2) Tokyo edge server: Check cache. (3) Cache miss (first request). (4) Edge fetches from origin (US-East). (5) Edge caches logo.png locally. (6) Edge returns logo to user. (7) Next user in Tokyo: Cache hit (served from Tokyo edge immediately). Characteristics: Automatic: No manual action needed. On-demand: Only requested files cached. Efficient: Popular content cached, unpopular not. PROS: Simple setup: Just point CDN to origin. Automatic: CDN handles caching logic. Storage efficient: Only popular files cached. Dynamic: New content automatically cached when requested. CONS: First request slow: Cache miss hits origin. Unpredictable: Can't guarantee what's cached. Origin dependency: Cache misses require origin to be healthy. WHEN TO USE PULL CDN: Most common use case. Dynamic websites with many pages/images. Content changes frequently. Don't know which content will be popular. Examples: E-commerce sites, blogs, news sites. PUSH CDN: How it works: You proactively upload content to CDN before users request it. Flow: (1) You upload logo.png to CDN (via API/dashboard). (2) CDN distributes to ALL edge servers worldwide. (3) User in Tokyo requests /logo.png. (4) Tokyo edge: Cache hit immediately (already there). (5) No origin request ever needed. Characteristics: Proactive: You control what's cached. Predictable: All content available immediately. Manual: Requires you to push updates. PROS: No cache misses: Content always cached. Predictable performance: Consistent fast loads. No origin dependency: Can work even if origin down. Instant global availability: All edges have content immediately. CONS: Manual effort: Must push every file/update. Storage waste: All content cached even if unpopular. Slower deployments: Time to propagate to all edges (minutes). Management overhead: Track what's pushed, purge old versions. WHEN TO USE PUSH CDN: Static websites (rarely change). Large files (videos, software downloads). Product launches (know traffic spike coming). Marketing campaigns (predictable content). Mobile app binaries. Examples: Static documentation sites, video platforms, software distribution. COMPARISON TABLE: Aspect / Pull CDN / Push CDN: Caching: On-demand (first request) / Proactive (before request). Setup: Automatic / Manual. Popular content: Cached / Cached. Unpopular content: Not cached (efficient) / Cached (wasteful). First request: Slow (cache miss) / Fast (cache hit). Content updates: Automatic (via TTL) / Manual (re-push). Storage: Efficient / Wasteful. Use case: Dynamic sites / Static sites, predictable traffic. HYBRID APPROACH: Many sites use both. Example: Push CDN: Homepage, key landing pages (always fast). Pull CDN: User-generated content, long-tail pages (on-demand). Result: Best of both (fast for important content, efficient for rest). REAL-WORLD EXAMPLES: PULL CDN: Facebook: Billions of user photos (can't push all). Amazon: Millions of product images (pulls on demand). PUSH CDN: Apple: macOS updates (push to all edges before launch). Netflix: New movie releases (push before premiere). VERSIONING WITH PUSH CDN: Problem: Push logo.png, then update it. Old version still cached. Solution: Version URLs: /logo-v1.png, /logo-v2.png. Or query string: /logo.png?v=2. Result: Each version has unique URL, no cache collision. FINAL RECOMMENDATION: Default to Pull CDN (simpler, most flexible). Use Push CDN for: Static sites, predictable traffic, large files. Hybrid: Combine both for optimal performance.",
          keyPoints: [
            'Pull CDN: fetches from origin on first request (lazy, automatic, most common)',
            'Push CDN: you upload content before requests (proactive, manual, no cache misses)',
            'Use Pull for: dynamic sites, unpredictable traffic, many pages',
            'Use Push for: static sites, product launches, large files, predictable traffic',
            'Hybrid approach: Push for critical pages, Pull for long-tail content',
          ],
        },
        {
          id: 'q3',
          question:
            'You updated your logo.png on the origin server, but users still see the old logo (cached on CDN). Walk through 3 solutions to fix this.',
          sampleAnswer:
            'PROBLEM: STALE CACHE. Scenario: Updated logo.png on origin server. CDN edge servers still have old version cached. Users see old logo. Need to force CDN to serve new version. SOLUTION 1: CACHE PURGE (INVALIDATION). How it works: Manually delete cached file from CDN edge servers. Implementation: API call or dashboard: "Purge /logo.png". CDN broadcasts purge command to ALL edge servers. Edge servers delete cached logo.png. Next user request: Cache miss → Fetch from origin → Cache new version. Timing: Purge propagation: 1-5 minutes (all edges updated). User sees new logo: Within 5 minutes. PROS: Immediate effect: All users see new version soon. Works for emergency updates. CONS: Not instant: Takes 1-5 minutes to propagate. Costs money: Some CDNs charge per purge. Can cause traffic spike: All edges refetch simultaneously. Requires action: Must remember to purge on every update. WHEN TO USE: Emergency fixes (wrong logo uploaded). Infrequent updates. Small number of files. SOLUTION 2: VERSIONED URLs (CACHE BUSTING). How it works: Change URL when content changes. Edge servers see new URL → Cache miss → Fetch new version. Implementation OPTIONS: Option A: Query string versioning: Old: /logo.png. New: /logo.png?v=2 (or /logo.png?v=1234567890 timestamp). Option B: Filename versioning: Old: /logo.png. New: /logo-v2.png or /logo-20231015.png. Option C: Path versioning: Old: /assets/logo.png. New: /assets/v2/logo.png. Code example: HTML before: img src="/logo.png". HTML after: img src="/logo.png?v=2". Build process: Automatically append hash of file content. Result: /logo.png?v=abc123 (changes when file changes). PROS: INSTANT: Users see new version immediately (no cache, new URL). FREE: No purge costs. AUTOMATIC: Build tools handle versioning. ZERO DOWNTIME: Old and new versions coexist (gradual rollout). CONS: URL management: Must update all references to file. HTML changes: Need to deploy new HTML with new URLs. Best practice: Automate with build tools (Webpack, Vite). WHEN TO USE: BEST PRACTICE: Use this by default. Frequent updates. Critical for: JavaScript, CSS (breaking changes). SOLUTION 3: REDUCE TTL (TIME TO LIVE). How it works: Set short cache expiration time. After TTL expires, edge refetches from origin. Implementation: HTTP headers from origin: Before: Cache-Control: public, max-age=31536000 (1 year). After: Cache-Control: public, max-age=300 (5 minutes). Result: Edge servers cache for only 5 minutes. After 5 minutes: Edge refetches from origin. Max staleness: 5 minutes. PROS: Automatic: No manual purge needed. Predictable: Know exactly how long until update. CONS: Increased origin load: More frequent refetches. Slower for users: More cache misses. Not suitable for static content (defeats purpose of CDN). WHEN TO USE: Dynamic content: API responses, frequently changing pages. Acceptable staleness: Content can be 5 minutes old. Trade-off: Lower cache hit rate for fresher content. COMPARISON TABLE: Method / Speed / Cost / Effort / Best for: Purge: 1-5 min / Costs $ / Manual / Emergency. Versioned URLs: Instant / Free / Automated / Best practice. Short TTL: Minutes / Free / Automatic / Dynamic content. RECOMMENDED APPROACH: COMBINATION. For static assets (logo, CSS, JS): Use versioned URLs: /logo.png?v=abc123. Long TTL: Cache for 1 year (immutable). Build tool: Automatically append hash (Webpack, Vite). Result: Instant updates, maximum cache hit rate, free. For dynamic content: Short TTL: 5-60 seconds. Pull CDN: On-demand caching. Result: Fresh content, some caching benefit. For emergencies: Purge API: Available as fallback. Use rarely (only for mistakes). REAL-WORLD EXAMPLE: FACEBOOK: Versioned URLs for static assets: /static/logo.abc123.png. Long TTL: 1 year (immutable). Build: Automated hash generation. Purge: Only for emergencies (privacy issues, DMCA). IMPLEMENTATION WITH WEBPACK: webpack.config.js: output: { filename: \'[name].[contenthash].js\',  // Auto-versioning. }. Result: main.abc123.js, logo.def456.png. HTML: Automatically updated by HtmlWebpackPlugin. Deployment: Upload new files → Deploy new HTML → Instant updates. FINAL RECOMMENDATION: USE VERSIONED URLS as primary strategy. Automate with build tools (zero manual effort). Set long TTL for static assets (max cache hit rate). Keep purge API for emergencies only. Educate team: Always version URLs, never purge.',
          keyPoints: [
            'Solution 1: Purge (delete cached file, 1-5 min propagation, costs money)',
            'Solution 2: Versioned URLs (change URL, instant, free, BEST PRACTICE)',
            'Solution 3: Short TTL (auto-expire cache, increased origin load)',
            'Recommended: Versioned URLs + long TTL for static assets',
            'Automate versioning with build tools (Webpack contenthash)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the primary benefit of using a CDN?',
          options: [
            'Reduced database load',
            'Reduced latency by serving content from geographically closer servers',
            'Increased security through encryption',
            'Automatic content compression',
          ],
          correctAnswer: 1,
          explanation:
            "CDN primary benefit: Reduced latency by caching content on edge servers worldwide. Users fetch from nearest edge (10-50ms) vs origin (200-300ms). Secondary benefits: Reduced origin bandwidth (80-90%), DDoS protection, high availability. Not primarily for database (that's caching), security (CDN can help but not primary), or compression (origin responsibility).",
        },
        {
          id: 'mc2',
          question:
            'Your CDN cache hit rate is 60% (target is 90%). What could be the problem?',
          options: [
            'TTL is too short (content expires too quickly)',
            'Origin server is too slow',
            'Users are geographically distributed',
            'CDN is overloaded',
          ],
          correctAnswer: 0,
          explanation:
            "Low cache hit rate (60%) typically caused by: TTL too short (content expires too quickly, edge refetches too often). Solution: Increase TTL for static content. Or: Frequent cache purges. Or: Many unique URLs (query strings not cached). Origin speed doesn't affect hit rate (affects cache miss latency). Geographic distribution is good (CDN handles it). CDN overload would cause slow responses, not low hit rate.",
        },
        {
          id: 'mc3',
          question:
            'What is the recommended way to update cached content on a CDN?',
          options: [
            'Manually purge the cache every time you update content',
            'Set TTL to 1 second so content updates frequently',
            'Use versioned URLs (cache busting) like /logo.png?v=2',
            'Restart the CDN servers',
          ],
          correctAnswer: 2,
          explanation:
            'Best practice: Versioned URLs (/logo.png?v=2 or /logo-abc123.png). Benefits: Instant updates (no wait for purge), free (no purge costs), automatic (build tools), immutable (long TTL, high hit rate). Purging works but slow (1-5 min), costs money, manual effort. Short TTL defeats CDN purpose (low hit rate). Restarting CDN not possible (managed service).',
        },
        {
          id: 'mc4',
          question: 'When should you NOT use a CDN?',
          options: [
            'Your website has global users across multiple continents',
            'You serve large video files',
            'You have very low traffic (<100 visitors/day) and all users are in the same city as your server',
            'You want to reduce origin server bandwidth costs',
          ],
          correctAnswer: 2,
          explanation:
            "Don't use CDN when: Very low traffic (<100 visitors/day), all users near origin server (no latency benefit), CDN cost > benefit. Use CDN when: Global users (latency reduction), large files (bandwidth savings), high traffic (origin offload), DDoS protection needed. Option 2 is low traffic + same city = CDN overkill (adds latency for DNS lookups).",
        },
        {
          id: 'mc5',
          question:
            'What is the typical cache hit rate target for a well-configured CDN?',
          options: ['50-60%', '70-80%', '90-95%', '99-100%'],
          correctAnswer: 2,
          explanation:
            '90-95% cache hit rate is typical target. This means 90-95% of requests served from edge cache (fast), 5-10% from origin (cache misses). 99-100% unrealistic (uncacheable content, new content, low-traffic pages). 50-60% indicates problem (TTL too short, too many purges, poor cache config). Monitor hit rate and optimize if <85%.',
        },
      ],
    },
    {
      id: 'api-gateway',
      title: 'API Gateway',
      content: `An API Gateway is a server that acts as a single entry point for all client requests, routing them to appropriate backend services while providing cross-cutting functionality like authentication, rate limiting, and monitoring.

## What is an API Gateway?

**Definition**: An API Gateway sits between clients and backend microservices, providing a unified interface and handling common tasks centrally.

### **Why Use an API Gateway?**

**Without API Gateway:**
- Clients call microservices directly (tight coupling)
- Each service implements authentication separately (duplication)
- No centralized rate limiting or logging
- Complex client logic (knows all service endpoints)
- Security risks (services exposed directly)

**With API Gateway:**
- Single entry point for all requests
- Centralized authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Load balancing and service discovery
- Simplified client code

**Real-world**: Netflix uses Zuul API Gateway to handle billions of requests per day.

---

## Core Responsibilities

### **1. Request Routing**

Route requests to appropriate backend services based on URL path.

**Example:**
- GET /api/users/123 → User Service
- GET /api/orders/456 → Order Service
- GET /api/products/789 → Product Service

**Benefits:** Clients only know one endpoint (gateway), not all microservices.

---

### **2. Authentication & Authorization**

Verify user identity and permissions before forwarding requests.

**Flow:**
1. Client sends request with JWT token
2. Gateway validates token
3. If valid: Extract user info, forward to backend
4. If invalid: Return 401 Unauthorized

**Benefits:** Authentication logic centralized (not in every microservice).

---

### **3. Rate Limiting & Throttling**

Limit number of requests per user/API key to prevent abuse.

**Example:**
- Free tier: 1000 requests/day
- Premium tier: 100,000 requests/day

**Implementation:** Track request count in Redis, reject if limit exceeded.

**Benefits:** Protect backend from overload, enforce fair usage.

---

### **4. Request/Response Transformation**

Modify requests or responses (format conversion, header manipulation).

**Example:**
- Client sends XML → Gateway converts to JSON for backend
- Backend returns v2 API → Gateway adapts to v1 for legacy clients

**Benefits:** Backward compatibility, protocol translation.

---

### **5. Load Balancing**

Distribute requests across multiple backend instances.

**Example:**
- User Service has 5 instances
- Gateway load balances requests across all 5

**Benefits:** Horizontal scaling, high availability.

---

### **6. Caching**

Cache responses to reduce backend load.

**Example:**
- GET /api/products/123 → Cache for 60 seconds
- Subsequent requests served from cache

**Benefits:** Lower latency, reduced backend load.

---

### **7. Logging & Monitoring**

Centralized logging for all API requests.

**Metrics:**
- Request count per endpoint
- Response time (latency)
- Error rate (4xx, 5xx)
- Top users by request count

**Benefits:** Single place to monitor all API traffic.

---

## API Gateway vs Load Balancer

**Comparison:**

**Load Balancer:**
- Layer 4 or Layer 7
- Simple traffic distribution
- No business logic
- Routes to same service type

**API Gateway:**
- Layer 7 only (HTTP/HTTPS)
- Intelligent routing (different services per path)
- Business logic (auth, rate limiting, transformation)
- Routes to different services

**When to use both:** Gateway handles application logic, Load Balancer handles traffic distribution.

---

## API Gateway Patterns

### **1. Backend for Frontend (BFF)**

Separate gateway per client type (web, mobile, IoT).

**Architecture:**
- Web BFF: Optimized for web browsers
- Mobile BFF: Optimized for mobile apps (smaller payloads)
- IoT BFF: Optimized for IoT devices (minimal data)

**Benefits:** Tailored responses per client, better performance.

---

### **2. Aggregation**

Gateway fetches data from multiple services, aggregates, returns to client.

**Example:**
- Client requests user profile
- Gateway calls: User Service, Order Service, Recommendation Service
- Gateway aggregates responses into single response

**Benefits:** Reduced client-side complexity, fewer round trips.

---

### **3. GraphQL Gateway**

Gateway exposes GraphQL API, translates to REST calls for backends.

**Benefits:** Clients query exactly what they need, no over-fetching.

---

## Popular API Gateway Solutions

### **AWS API Gateway**
- Fully managed (serverless)
- Integrates with Lambda, EC2, other AWS services
- Built-in authentication (IAM, Cognito)
- Pay-per-request pricing

**Best for:** AWS-native applications

---

### **Kong**
- Open-source (also has enterprise version)
- Plugin architecture (auth, rate limiting, logging)
- Built on NGINX (high performance)
- Self-hosted or cloud-managed

**Best for:** On-premise or hybrid deployments

---

### **NGINX**
- Not a dedicated API gateway, but can be configured as one
- High performance (handles 10K+ requests/sec)
- Reverse proxy + load balancer + API gateway

**Best for:** Simple use cases, high performance needs

---

### **Apigee (Google)**
- Enterprise API management platform
- Advanced analytics and monetization
- Developer portal
- Expensive

**Best for:** Large enterprises, API monetization

---

## API Gateway Challenges

### **1. Single Point of Failure**

**Problem:** If gateway down, all services unavailable.

**Solution:** 
- Deploy multiple gateway instances behind load balancer
- Health checks and auto-scaling
- Circuit breaker patterns

---

### **2. Performance Bottleneck**

**Problem:** All traffic goes through gateway (latency increase).

**Solution:**
- Scale gateway horizontally (add more instances)
- Optimize gateway code (avoid heavy processing)
- Cache aggressively

---

### **3. Complexity**

**Problem:** Gateway becomes complex with too many responsibilities.

**Solution:**
- Keep gateway thin (essential logic only)
- Push business logic to services
- Use plugins/middleware for modularity

---

## Security Best Practices

**1. Authentication:** Verify user identity (JWT, OAuth)

**2. Authorization:** Check user permissions (RBAC, ABAC)

**3. Rate Limiting:** Prevent DDoS and abuse

**4. Input Validation:** Sanitize requests (prevent injection attacks)

**5. HTTPS Only:** Encrypt all traffic (TLS/SSL)

**6. IP Whitelisting:** Restrict access by IP (for admin APIs)

**7. API Keys:** Require keys for programmatic access

---

## Real-World Examples

### **Netflix (Zuul)**
- Handles billions of requests/day
- Dynamic routing (A/B testing, canary deployments)
- Resilience patterns (circuit breaker, retries)

### **Uber**
- API Gateway for mobile apps
- Authentication and rate limiting
- Request logging and analytics

### **Shopify**
- API Gateway for merchants
- Rate limiting per shop
- Request transformation (API versioning)

---

## Key Takeaways

1. **API Gateway = single entry point** for all client requests
2. **Centralized cross-cutting concerns:** auth, rate limiting, logging
3. **Request routing:** Route to appropriate backend service by URL path
4. **Not a replacement for load balancer:** Use both together
5. **Security:** Authenticate, authorize, rate limit at gateway
6. **High availability:** Deploy multiple gateway instances`,
      quiz: [
        {
          id: 'q1',
          question:
            'Your microservices architecture has 10 services. Each client (web, mobile) calls services directly. This causes: (1) Clients need to know all 10 endpoints, (2) Each service implements authentication separately. Propose a solution.',
          sampleAnswer:
            'SOLUTION: IMPLEMENT API GATEWAY. PROBLEM ANALYSIS: Current architecture issues: (1) Tight coupling: Clients know all service endpoints. (2) Duplicate auth: Each service validates tokens (10× duplication). (3) Complex clients: Web/mobile apps have complex service discovery logic. (4) Security: Services exposed directly (attack surface). (5) No rate limiting: Services vulnerable to abuse. PROPOSED SOLUTION: API Gateway as single entry point. ARCHITECTURE: Before: Client → Service 1, Service 2, ..., Service 10 (10 connections). After: Client → API Gateway → Service 1, Service 2, ..., Service 10 (1 connection). IMPLEMENTATION: Deploy API Gateway (Kong, AWS API Gateway, NGINX). Configure routing: /api/users/* → User Service. /api/orders/* → Order Service. /api/products/* → Product Service. Configure authentication: Gateway validates JWT tokens (once). Gateway forwards requests with user context. Configure rate limiting: 1000 requests/hour per user. Gateway returns 429 if exceeded. Configure logging: Centralized request logs. BENEFITS: Simplified clients: Only one endpoint (gateway.example.com). No auth duplication: Authentication at gateway only (1× not 10×). Rate limiting: Centralized (protect all services). Monitoring: Single place to log/monitor all traffic. Security: Services not exposed directly (internal network only). DETAILED FLOW: Example: Client requests GET /api/orders/123. Step 1: Client → Gateway (gateway.example.com/api/orders/123). Step 2: Gateway validates JWT token. If invalid: Return 401 Unauthorized. If valid: Extract user ID from token. Step 3: Gateway checks rate limit. If exceeded: Return 429 Too Many Requests. Step 4: Gateway routes request to Order Service (internal network). Request: GET http://order-service-internal:8080/orders/123. Headers: X-User-ID: 456 (from token). Step 5: Order Service processes request (no auth needed, trusts gateway). Step 6: Order Service returns response to Gateway. Step 7: Gateway logs request and returns response to Client. BENEFITS QUANTIFIED: Auth logic: Before: 10 services × 100 lines auth code = 1000 lines. After: Gateway: 100 lines. Services: 0 lines. Reduction: 90% less code. Performance: Before: Client makes 10 requests (one per service). Latency: 10 × 200ms = 2000ms (serial). After: Client makes 1 request to Gateway. Gateway aggregates if needed. Latency: 200ms (gateway) + parallel service calls. Security: Before: 10 attack surfaces (services exposed). After: 1 attack surface (only gateway exposed). TRADE-OFFS: Single point of failure: If gateway down, all services unavailable. Mitigation: Deploy multiple gateway instances behind load balancer. Added latency: Extra hop through gateway (~5-10ms). Mitigation: Deploy gateway close to services (same datacenter). Gateway becomes bottleneck: High traffic overloads gateway. Mitigation: Scale gateway horizontally (add instances). COST: API Gateway cost (AWS): ~$3.50 per million requests. 10M requests/month: $35. Engineering savings: Reduced auth code, easier monitoring. Net: Positive ROI (simplicity + security > cost). MIGRATION STRATEGY: Phase 1: Deploy gateway alongside existing architecture. Phase 2: Route 10% traffic through gateway (canary). Phase 3: Monitor for issues (latency, errors). Phase 4: Gradually increase to 100%. Phase 5: Remove direct client→service connections. MONITORING: Track metrics: Request count per endpoint. Response time (p50, p95, p99). Error rate (4xx, 5xx). Auth failures (invalid tokens). Rate limit hits. Alert if: Error rate >1%. Latency p95 >500ms. FINAL RECOMMENDATION: Implement API Gateway immediately. Centralize authentication (remove from services). Add rate limiting (protect from abuse). Monitor closely (single point of failure). Expected outcome: Simpler clients, 90% less auth code, better security.',
          keyPoints: [
            'API Gateway = single entry point for all clients (simplified client code)',
            'Centralized auth: validate JWT once at gateway (not in every service)',
            'Rate limiting: protect all services with centralized throttling',
            'Trade-off: Single point of failure (mitigate with multiple gateway instances)',
            'Result: 90% less auth code, better security, easier monitoring',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain the difference between an API Gateway and a Load Balancer. When would you use each, and can you use both together?',
          sampleAnswer:
            'API GATEWAY VS LOAD BALANCER: LOAD BALANCER: Purpose: Distribute traffic across multiple instances of SAME service. Layer: Layer 4 (TCP/UDP) or Layer 7 (HTTP). Intelligence: Minimal (just route to healthy instances). Business logic: None. Example: 5 instances of User Service. Load balancer distributes requests evenly across all 5. Use case: Scale single service horizontally. API GATEWAY: Purpose: Route requests to DIFFERENT services based on URL path. Layer: Layer 7 (HTTP/HTTPS) only. Intelligence: High (routing, auth, rate limiting, transformation). Business logic: Yes (authentication, rate limiting, request transformation). Example: /api/users → User Service. /api/orders → Order Service. /api/products → Product Service. Use case: Microservices architecture (single entry point). COMPARISON TABLE: Aspect / Load Balancer / API Gateway: Routing: Same service (multiple instances) / Different services (by path). Layer: L4 or L7 / L7 only. Auth: No / Yes. Rate limiting: No / Yes. Transformation: No / Yes. Caching: Minimal / Yes. Monitoring: Basic / Advanced. Complexity: Low / High. WHEN TO USE LOAD BALANCER: Scaling single service: 10 instances of Order Service. High availability: If one instance fails, route to others. Simple traffic distribution: No business logic needed. Example: Single monolithic app with 5 replicas. WHEN TO USE API GATEWAY: Microservices: Multiple services behind single endpoint. Cross-cutting concerns: Auth, rate limiting, logging. Request routing: Different paths → different services. Client simplification: One endpoint instead of many. Example: E-commerce with User, Order, Product, Payment services. USING BOTH TOGETHER (COMMON): Architecture: Client → API Gateway → Load Balancer → Service Instances. Flow: (1) Client sends request to API Gateway. (2) Gateway: Authenticates, routes by path. (3) Gateway forwards to Order Service Load Balancer. (4) Load Balancer distributes across 5 Order Service instances. (5) One instance handles request. (6) Response back through Load Balancer → Gateway → Client. Why use both? Gateway: Handles application-level concerns (auth, routing). Load Balancer: Handles traffic distribution (scaling, availability). Result: Best of both worlds. EXAMPLE: NETFLIX ARCHITECTURE: Layer 1: AWS ELB (Load Balancer) → Distributes to Zuul instances. Layer 2: Zuul (API Gateway) → Routes by path, auth, rate limiting. Layer 3: Service Load Balancers → Distribute to service instances. Layer 4: Service Instances → Process requests. Benefits: Gateway scales independently (multiple Zuul instances). Services scale independently (multiple instances each). Gateway provides intelligence, Load Balancers provide distribution. DETAILED EXAMPLE: UBER: Client request: GET /api/rides/123. Architecture: (1) Client → Cloudflare (Global Load Balancer, CDN). (2) Cloudflare → AWS ELB (Regional Load Balancer). (3) ELB → API Gateway Instance (1 of 10). (4) API Gateway: Validates JWT, checks rate limit. (5) API Gateway routes: /api/rides → Ride Service Load Balancer. (6) Ride Service LB → Ride Service Instance (1 of 20). (7) Ride Service processes request. Result: Client makes 1 request. 4 layers of routing/distribution. Fast, reliable, scalable. CAN YOU SKIP ONE? Skip Load Balancer? If only 1 instance per service: Yes, go directly to service. If multiple instances: No, need load balancer. Skip API Gateway? If single monolithic app: Yes, just use load balancer. If microservices: No, need gateway for routing. RECOMMENDATION FOR DIFFERENT SCENARIOS: Scenario 1: Monolithic App (1 service, 10 instances). Use: Load Balancer only. Why: No need for routing (same service), just distribution. Scenario 2: Microservices (10 services, 1 instance each). Use: API Gateway only. Why: Need routing (different services), no need for distribution (1 instance each). Scenario 3: Microservices (10 services, 10 instances each). Use: API Gateway + Load Balancers. Why: Need routing (different services) AND distribution (multiple instances). This is most common in production. COST COMPARISON: Load Balancer only: AWS ELB: $20/month + data transfer. Simple, cheap. API Gateway only: AWS API Gateway: $3.50 per million requests. More expensive at scale. Both together: ELB: $20/month. API Gateway: $3.50 per million requests. Service LBs: $20/month each × 10 services = $200/month. Total: $220/month + $3.50/million requests. Worth it? Yes, for microservices (complexity reduction, security). TRADE-OFFS: Using both: Pros: Clean separation, scales independently, best practices. Cons: Added cost, more complexity, extra latency hop. Using only gateway: Pros: Simpler architecture, lower cost. Cons: Gateway must handle load balancing (added responsibility). Using only load balancer: Pros: Simple, cheap. Cons: No microservices support, no auth/rate limiting. FINAL RECOMMENDATION: Microservices: Use API Gateway + Load Balancers (industry standard). Monolith: Use Load Balancer only (simpler). Hybrid: Start with gateway only (1 instance per service), add load balancers when scaling (>3 instances per service).',
          keyPoints: [
            'Load Balancer: distributes traffic across instances of SAME service',
            'API Gateway: routes requests to DIFFERENT services by URL path',
            'Load Balancer: no business logic, just distribution',
            'API Gateway: auth, rate limiting, transformation',
            'Use both: Gateway for routing/auth, Load Balancer for scaling (industry standard)',
          ],
        },
        {
          id: 'q3',
          question:
            'Your API Gateway is becoming a performance bottleneck (high latency, 500ms+ response times). What could be the causes and how would you optimize?',
          sampleAnswer:
            "API GATEWAY PERFORMANCE OPTIMIZATION: SYMPTOMS: High latency: p95 = 500ms (should be <50ms). Slow requests: Users complain about lag. Gateway CPU: 80-90% utilization (bottleneck). DIAGNOSIS: Step 1: Measure where time is spent. Use APM tool (New Relic, Datadog) to trace requests. Break down latency: Gateway processing: ?ms. Backend service call: ?ms. Network: ?ms. COMMON CAUSES & SOLUTIONS: CAUSE 1: SYNCHRONOUS BLOCKING I/O. Problem: Gateway makes sequential backend calls. Example: Request to /api/user-profile. Gateway calls: (1) User Service: 100ms. (2) Order Service: 150ms. (3) Recommendation Service: 100ms. Total: 350ms (sequential). Solution: Parallel backend calls (async I/O). Implementation: Use async/await (Node.js) or CompletableFuture (Java). Make calls in parallel: Promise.all([userService(), orderService(), recommendationService()]). Total: max(100ms, 150ms, 100ms) = 150ms. Improvement: 350ms → 150ms (2.3× faster). CAUSE 2: NO CACHING. Problem: Every request hits backend services. Solution: Cache responses at gateway. Implementation: Cache GET requests: GET /api/products/123 → Cache for 60 seconds. Use Redis for cache (fast, distributed). Cache hit: 1-2ms (vs 100ms backend). Cache hit rate: 80-90% (10× load reduction). Example: Before: 100% requests hit backend (100ms each). After: 90% cached (2ms), 10% backend (100ms). Average: 0.9 × 2ms + 0.1 × 100ms = 11.8ms. Improvement: 100ms → 11.8ms (8.5× faster). CAUSE 3: INEFFICIENT AUTH VALIDATION. Problem: Gateway validates JWT on every request (expensive). Example: JWT validation: Parse token, verify signature, check expiration: 50ms. Solution: Cache auth results. Implementation: Cache user permissions in Redis. Key: token_hash → user_id + permissions. TTL: 5 minutes. Auth flow: (1) Hash token. (2) Check Redis cache. (3) If hit: Use cached user info (1ms). (4) If miss: Validate JWT, cache result (50ms). Cache hit rate: 95% (most users make multiple requests). Improvement: 50ms → 2.5ms (20× faster). CAUSE 4: TOO MANY GATEWAY RESPONSIBILITIES. Problem: Gateway does too much: Auth, rate limiting, logging, transformation, aggregation. Each adds latency. Solution: Offload non-critical logic. Move to backends: Complex transformations. Business logic. Aggregation (use GraphQL). Keep in gateway: Auth (critical for security). Rate limiting (protect backends). Simple routing. Improvement: Reduce gateway processing from 100ms to 10ms. CAUSE 5: INSUFFICIENT GATEWAY INSTANCES. Problem: Single gateway instance overloaded. CPU: 90% (can't handle more requests). Solution: Horizontal scaling (add more instances). Implementation: Deploy 10 gateway instances (instead of 1). Load balancer distributes traffic: Each instance handles 10% of traffic. CPU per instance: 9% (plenty of headroom). Improvement: Can handle 10× more traffic. CAUSE 6: SLOW BACKEND SERVICES. Problem: Gateway fast, but backend services slow. Example: Order Service takes 500ms to respond. Solution: Not a gateway problem, optimize backends. But gateway can help: (1) Circuit breaker: If backend slow/down, fail fast (don't wait). (2) Timeout: Set aggressive timeout (e.g., 200ms). (3) Retry: Retry failed requests (with backoff). (4) Fallback: Return cached/default response if backend unavailable. OPTIMIZATION CHECKLIST: 1. Enable caching: Redis cache for GET requests. Cache hit rate target: 90%. TTL: 60 seconds for static data. 2. Parallel backend calls: Use async I/O for multiple services. Reduce sequential latency. 3. Cache auth results: JWT validation result cached. TTL: 5 minutes. 4. Horizontal scaling: Add more gateway instances. Target: CPU <50% per instance. 5. Reduce responsibilities: Move complex logic to backends. Keep gateway thin. 6. Optimize logging: Async logging (don't block requests). Sample logs (e.g., 10% of requests). 7. Use HTTP/2: Multiplexing reduces connection overhead. 8. Connection pooling: Reuse backend connections (avoid TCP handshake). MONITORING: Track metrics: Latency: p50, p95, p99 (target: p95 <50ms). Throughput: Requests/sec (capacity planning). Cache hit rate: >90% (optimize caching). Error rate: <0.1% (reliability). CPU utilization: <50% (headroom). Alert if: p95 latency >100ms. Error rate >1%. CPU >80%. REAL-WORLD EXAMPLE: NETFLIX (ZUUL): Optimization: Async I/O (Netty framework, non-blocking). Caching (aggressive caching of metadata). Horizontal scaling (thousands of Zuul instances). Circuit breaker (fail fast if backend slow). Result: p99 latency <100ms. Handles billions of requests/day. EXPECTED IMPROVEMENTS: Before: p95 latency: 500ms. After optimizations: Caching: 500ms → 50ms (10× cache hit rate). Parallel calls: 50ms → 20ms (parallel backend calls). Auth caching: 20ms → 10ms (cached auth). Result: p95 latency: 10ms (50× improvement). COST: Horizontal scaling: 10 instances × $100/month = $1000/month. Redis cache: $200/month. Total: $1200/month. Benefit: Handle 50× more traffic, better UX, fewer customer complaints. ROI: Positive (avoid losing customers due to slow site). FINAL RECOMMENDATION: Profile gateway to identify bottleneck (use APM). Enable caching (biggest impact: 10× latency reduction). Parallelize backend calls (2-3× improvement). Scale horizontally (add instances as needed). Keep gateway thin (push logic to backends). Monitor continuously (p95 latency, cache hit rate).",
          keyPoints: [
            'Caching: Biggest impact (10× latency reduction, 90% cache hit rate)',
            'Parallel backend calls: 2-3× faster (async I/O instead of sequential)',
            'Horizontal scaling: Add more gateway instances (reduce CPU load)',
            'Cache auth results: 20× faster JWT validation',
            'Keep gateway thin: Move complex logic to backend services',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the primary purpose of an API Gateway in a microservices architecture?',
          options: [
            'To scale backend services horizontally',
            'To provide a single entry point and handle cross-cutting concerns like authentication and rate limiting',
            'To store data for microservices',
            'To replace all backend services with a monolith',
          ],
          correctAnswer: 1,
          explanation:
            "API Gateway primary purpose: Single entry point for clients + handle cross-cutting concerns (auth, rate limiting, logging, request routing). Not for scaling (that's load balancer), data storage (that's database), or replacing services (gateway routes to services, doesn't replace them).",
        },
        {
          id: 'mc2',
          question:
            'Your API Gateway validates JWT tokens on every request (50ms per validation). Requests are slow. What optimization would have the biggest impact?',
          options: [
            'Use a faster JWT library',
            'Cache authentication results in Redis (validate once, cache for 5 minutes)',
            'Skip authentication for some endpoints',
            'Use HTTP instead of HTTPS',
          ],
          correctAnswer: 1,
          explanation:
            'Caching auth results has biggest impact: Validate JWT once (50ms), cache result in Redis. Subsequent requests: Check cache (1ms) instead of validating (50ms). Cache hit rate: 95%+ (users make multiple requests). Result: 50ms → 2.5ms (20× faster). Option 1 might save 5-10ms. Option 3 is security risk. Option 4 is security risk (never skip HTTPS).',
        },
        {
          id: 'mc3',
          question:
            'When should you use BOTH an API Gateway and Load Balancer?',
          options: [
            'Never, they serve the same purpose',
            'When you have a monolithic application with multiple instances',
            'When you have microservices with multiple instances per service',
            'Only when you have more than 100 users',
          ],
          correctAnswer: 2,
          explanation:
            'Use both when: Microservices (multiple services) + multiple instances per service. Architecture: Client → Gateway (routes by path: /api/users, /api/orders) → Load Balancer per service (distributes across instances) → Service instances. Gateway handles routing/auth. Load Balancer handles distribution. This is industry standard for production microservices.',
        },
        {
          id: 'mc4',
          question: 'What is the main risk of using an API Gateway?',
          options: [
            'It makes clients more complex',
            'It eliminates the need for authentication',
            'It becomes a single point of failure if not properly configured',
            'It prevents horizontal scaling',
          ],
          correctAnswer: 2,
          explanation:
            'Main risk: Single point of failure. If gateway down, ALL services unavailable (even if services healthy). Mitigation: (1) Deploy multiple gateway instances behind load balancer. (2) Health checks and auto-scaling. (3) Circuit breaker patterns. Gateway actually simplifies clients (option 1 wrong), enables auth (option 2 wrong), and enables scaling (option 3 wrong).',
        },
        {
          id: 'mc5',
          question:
            'Your API Gateway needs to call User Service, Order Service, and Recommendation Service for a single client request. What is the BEST approach?',
          options: [
            'Call services sequentially (User → Order → Recommendation)',
            'Call services in parallel using async I/O',
            'Cache all responses permanently to avoid calling services',
            'Have the client call each service directly',
          ],
          correctAnswer: 1,
          explanation:
            'Best approach: Parallel async calls. Sequential: 100ms + 150ms + 100ms = 350ms total. Parallel: max(100ms, 150ms, 100ms) = 150ms total (2.3× faster). Use Promise.all or similar. Option 3 (permanent cache) stale data. Option 4 defeats purpose of gateway (client complexity).',
        },
      ],
    },
    {
      id: 'proxies',
      title: 'Proxies (Forward & Reverse)',
      content: `A proxy server acts as an intermediary between clients and servers, forwarding requests and responses. Forward proxies serve clients, while reverse proxies serve servers.

## What is a Proxy?

**Definition**: A proxy is an intermediate server that sits between a client and a server, forwarding requests and responses.

**Types:**
- **Forward Proxy**: Serves clients (hides client identity from servers)
- **Reverse Proxy**: Serves servers (hides server identity from clients)

---

## Forward Proxy

### **Purpose**

Forward proxy sits between clients and the internet, forwarding client requests to external servers.

**Use cases:**
- Hide client IP addresses (anonymity)
- Bypass geographic restrictions (access blocked content)
- Content filtering (block certain websites)
- Caching (reduce bandwidth)

### **How it Works**

**Flow:**
1. Client configures browser to use proxy
2. Client sends request to proxy: "GET google.com"
3. Proxy forwards request to google.com (on behalf of client)
4. Google sees proxy IP, not client IP
5. Google sends response to proxy
6. Proxy forwards response to client

**From server's perspective:** All requests come from proxy IP (not client IP).

### **Examples**

**Corporate Proxy:**
- Company employees use company proxy
- Proxy filters: Blocks social media, porn sites
- Proxy logs: Tracks employee browsing (compliance)

**VPN (Virtual Private Network):**
- Acts as forward proxy
- Encrypts traffic (privacy)
- Changes apparent location (bypass geo-restrictions)

**Tor Network:**
- Chain of proxies (onion routing)
- Maximum anonymity
- Very slow (multiple hops)

---

## Reverse Proxy

### **Purpose**

Reverse proxy sits in front of backend servers, forwarding client requests to appropriate servers.

**Use cases:**
- Load balancing (distribute traffic)
- SSL termination (decrypt HTTPS at proxy)
- Caching (cache static content)
- Security (hide backend servers)
- Compression (gzip responses)

### **How it Works**

**Flow:**
1. Client sends request to example.com
2. DNS resolves to reverse proxy IP (not backend server)
3. Reverse proxy receives request
4. Proxy forwards to backend server (internal network)
5. Backend server processes request
6. Backend sends response to proxy
7. Proxy forwards response to client

**From client's perspective:** Talking directly to example.com (doesn't know proxy exists).

### **Examples**

**NGINX as Reverse Proxy:**
- Most common use case
- Load balances across backend servers
- Terminates SSL (HTTPS → HTTP to backends)
- Caches static content
- Serves static files directly (no backend hit)

**Cloudflare:**
- Global reverse proxy (CDN)
- DDoS protection
- SSL termination
- Caching and optimization

**AWS ELB (Elastic Load Balancer):**
- Managed reverse proxy
- Load balancing
- Health checks
- SSL termination

---

## Forward vs Reverse Proxy

**Comparison:**

**Forward Proxy:**
- Purpose: Serves clients
- Location: Client-side (client's network)
- Clients know: Yes (configured in browser)
- Servers know: No (sees proxy IP)
- Use case: Anonymity, content filtering
- Example: Corporate proxy, VPN

**Reverse Proxy:**
- Purpose: Serves servers
- Location: Server-side (server's network)
- Clients know: No (transparent)
- Servers know: Yes (receives proxy requests)
- Use case: Load balancing, SSL termination, caching
- Example: NGINX, Cloudflare, AWS ELB

---

## Common Proxy Features

### **1. SSL Termination**

Proxy decrypts HTTPS traffic, forwards HTTP to backend.

**Benefits:**
- Offload SSL computation from backends (CPU-intensive)
- Centralized certificate management (one place)
- Backend servers simpler (no SSL config)

**Flow:**
- Client → Proxy: HTTPS (encrypted)
- Proxy → Backend: HTTP (unencrypted, internal network)

---

### **2. Caching**

Proxy caches responses to reduce backend load.

**Example:**
- GET /logo.png → Proxy caches for 1 hour
- Subsequent requests served from cache (fast)

**Benefits:** Lower latency, reduced backend load.

---

### **3. Compression**

Proxy compresses responses before sending to client.

**Example:**
- Backend sends 1MB HTML
- Proxy compresses to 100KB (gzip)
- Client downloads 100KB (10× faster)

**Benefits:** Reduced bandwidth, faster page loads.

---

### **4. Load Balancing**

Reverse proxy distributes requests across multiple backend servers.

**Example:**
- 3 backend servers
- Proxy round-robins requests

**Benefits:** Horizontal scaling, high availability.

---

### **5. Security**

**Reverse Proxy:**
- Hides backend server IPs (security through obscurity)
- Blocks malicious requests (WAF - Web Application Firewall)
- Rate limiting (DDoS protection)

**Forward Proxy:**
- Content filtering (block malicious sites)
- Logging (audit user access)

---

## Real-World Examples

### **Forward Proxy: Corporate Network**

**Setup:**
- All employee traffic goes through company proxy
- Proxy logs all requests (compliance)
- Proxy blocks: Social media, gambling, adult content

**Benefits:** Productivity, security, compliance.

---

### **Reverse Proxy: E-commerce Website**

**Setup:**
- NGINX reverse proxy in front of 10 app servers
- Proxy handles: SSL termination, load balancing, static files
- App servers focus on business logic

**Benefits:** Simplified app servers, better performance, centralized SSL.

---

## Proxy vs Load Balancer vs API Gateway

**Proxy:**
- General purpose intermediary
- Forwarding + caching + security
- Can be forward or reverse

**Load Balancer:**
- Specific type of reverse proxy
- Focus: Traffic distribution
- Minimal business logic

**API Gateway:**
- Advanced reverse proxy
- Focus: API management (auth, rate limiting, routing)
- Business logic

**Relationship:** API Gateway > Load Balancer > Reverse Proxy (increasingly specialized).

---

## NGINX Example Configuration

**Simple Reverse Proxy:**

listen 80;
server_name example.com;

location / {
    proxy_pass http://backend:8080;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

**With Load Balancing:**

upstream backend {
    server backend1:8080;
    server backend2:8080;
    server backend3:8080;
}

server {
    location / {
        proxy_pass http://backend;
    }
}

**With Caching:**

proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m;

location / {
    proxy_cache my_cache;
    proxy_cache_valid 200 1h;
    proxy_pass http://backend;
}

---

## Key Takeaways

1. **Forward Proxy serves clients** (hides client from server)
2. **Reverse Proxy serves servers** (hides server from client)
3. **Reverse proxy use cases:** Load balancing, SSL termination, caching, security
4. **NGINX most common reverse proxy** (high performance, flexible)
5. **Proxies enable centralization** of concerns (SSL, caching, security)
6. **Clients know about forward proxy, not reverse proxy**`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the difference between Forward Proxy and Reverse Proxy. Give a real-world example of each.',
          sampleAnswer:
            "FORWARD PROXY VS REVERSE PROXY: FORWARD PROXY (CLIENT-SIDE): Definition: Proxy sits between clients and the internet, forwarding client requests to external servers. Who it serves: Clients (hides client identity from servers). Location: Client's network (corporate network, home network). Client awareness: Clients know proxy exists (configured in browser). Server awareness: Servers don't know client (sees proxy IP). Purpose: Anonymity, content filtering, bypass restrictions. REAL-WORLD EXAMPLE: CORPORATE PROXY. Scenario: Company with 1000 employees. Setup: All employee internet traffic routed through company proxy (proxy.company.com). Proxy configuration: Employees configure browser: HTTP Proxy: proxy.company.com:8080. Or: Automatic via network settings (PAC file). Flow: (1) Employee visits facebook.com. (2) Browser sends request to proxy: GET facebook.com. (3) Proxy checks policy: Is facebook.com allowed? (4) Proxy blocks: Return \"Access Denied\" page. (5) Employee visits work-related site (allowed). (6) Proxy forwards request to external site. (7) External site sees proxy IP (not employee IP). (8) Proxy receives response, forwards to employee. Use cases: Content filtering: Block social media, gambling, adult sites. Logging: Track all employee web access (compliance, security). Bandwidth savings: Cache common resources (OS updates, popular sites). Security: Scan for malware in downloads. Compliance: Ensure employees don't access illegal content. Benefits for company: Productivity: Employees can't waste time on social media. Security: Malware scanned before reaching employee machines. Compliance: Audit trail of all web access. REVERSE PROXY (SERVER-SIDE): Definition: Proxy sits in front of backend servers, forwarding client requests to appropriate servers. Who it serves: Servers (hides server identity from clients). Location: Server's network (in front of backend servers). Client awareness: Clients don't know proxy exists (transparent). Server awareness: Servers know proxy exists (receive proxy requests). Purpose: Load balancing, SSL termination, caching, security. REAL-WORLD EXAMPLE: NGINX FOR E-COMMERCE SITE. Scenario: E-commerce website (example.com) with 10 backend servers. Setup: NGINX reverse proxy in front of backend servers. Client requests go to NGINX first. Architecture: Client → NGINX (reverse proxy) → Backend servers (app1, app2, ..., app10). Flow: (1) Client visits example.com. (2) DNS resolves to NGINX IP (not backend servers). (3) Client sends: GET /products/123 (HTTPS). (4) NGINX receives request. (5) NGINX: SSL termination (decrypt HTTPS → HTTP). (6) NGINX: Check cache. If cached: Return immediately (fast). If not: Forward to backend server. (7) NGINX: Load balance (choose backend server via round-robin). (8) NGINX forwards to backend1: GET http://backend1:8080/products/123. (9) Backend1 processes request (fetch product from database). (10) Backend1 returns response to NGINX. (11) NGINX caches response (for future requests). (12) NGINX encrypts response (HTTP → HTTPS). (13) NGINX sends response to client. Use cases: Load balancing: Distribute traffic across 10 backends. SSL termination: NGINX handles SSL (backends don't need to). Caching: Static content cached at NGINX (images, CSS, JS). Static files: NGINX serves directly (no backend hit). Security: Hide backend IPs (only NGINX exposed). Compression: NGINX compresses responses (gzip). Benefits for site: Performance: SSL offloaded from backends (CPU-intensive). Scalability: Easy to add more backend servers. Security: Backend servers not directly exposed (attack surface reduced). Simplified backends: Backends don't handle SSL/caching (focus on business logic). High availability: If backend crashes, NGINX routes to healthy servers. COMPARISON TABLE: Aspect / Forward Proxy / Reverse Proxy: Serves: Clients / Servers. Location: Client network / Server network. Client knows: Yes (configured) / No (transparent). Server knows: No (sees proxy IP) / Yes (receives from proxy). Purpose: Anonymity, filtering / Load balancing, caching. Example: Corporate proxy, VPN / NGINX, Cloudflare. WHEN TO USE FORWARD PROXY: Corporate network: Content filtering, logging. Privacy: Hide IP address (VPN, Tor). Bypass restrictions: Access geo-blocked content. Bandwidth: Cache common resources. WHEN TO USE REVERSE PROXY: Web applications: Load balancing, SSL termination. Microservices: Single entry point (like API Gateway). High traffic: Caching, compression. Security: Hide backend servers, DDoS protection. BOTH TOGETHER: Some architectures use both. Example: Corporate employee (behind forward proxy) accesses website (behind reverse proxy). Flow: Employee → Forward Proxy (corporate) → Internet → Reverse Proxy (website) → Backend Server. FINAL RECOMMENDATION: Forward Proxy: Use for client-side needs (filtering, anonymity). Reverse Proxy: Use for server-side needs (load balancing, caching). Most web applications use reverse proxy (NGINX, Cloudflare). Some enterprises use forward proxy (content filtering).",
          keyPoints: [
            'Forward Proxy: serves clients, hides client IP from servers (corporate proxy, VPN)',
            'Reverse Proxy: serves servers, hides server IP from clients (NGINX, load balancer)',
            'Forward: clients know proxy exists (configured in browser)',
            "Reverse: clients don't know proxy exists (transparent)",
            'Use forward for: content filtering, anonymity; Use reverse for: load balancing, SSL termination',
          ],
        },
        {
          id: 'q2',
          question:
            'Your backend servers are overwhelmed with SSL encryption/decryption (high CPU usage). Propose a solution using a reverse proxy.',
          sampleAnswer:
            "SOLUTION: SSL TERMINATION AT REVERSE PROXY. PROBLEM ANALYSIS: Backend servers handling SSL/TLS: Each HTTPS request: (1) TCP handshake: 1 round trip. (2) TLS handshake: 2-3 round trips (negotiate cipher, exchange keys). (3) Decrypt request: CPU-intensive. (4) Process request: Business logic. (5) Encrypt response: CPU-intensive. SSL overhead: CPU: 10-15% per connection (encryption/decryption). Latency: 100-300ms for TLS handshake. Certificate management: Each backend needs SSL certificate. Impact: Backend servers at 80-90% CPU (SSL overhead). Can't handle more traffic (CPU bottleneck). Slow response times (encryption overhead). PROPOSED SOLUTION: SSL TERMINATION AT NGINX REVERSE PROXY. Architecture: Client → NGINX (HTTPS) → Backend (HTTP, internal network). IMPLEMENTATION: STEP 1: Deploy NGINX reverse proxy in front of backends. STEP 2: Configure NGINX for SSL termination. Install SSL certificate on NGINX (Let's Encrypt, commercial cert). NGINX config: server { listen 443 ssl; ssl_certificate /etc/nginx/ssl/cert.pem; ssl_certificate_key /etc/nginx/ssl/key.pem; ssl_protocols TLSv1.2 TLSv1.3; ssl_ciphers HIGH:!aNULL:!MD5; location / { proxy_pass http://backend-servers; proxy_set_header X-Forwarded-For $remote_addr; proxy_set_header X-Forwarded-Proto https; } }. STEP 3: Backend servers accept HTTP only (port 8080). Remove SSL config from backends. Backends trust NGINX (internal network). BENEFITS: Backend CPU reduction: Before: 80-90% CPU (SSL overhead). After: 50-60% CPU (no SSL). Savings: 30% CPU freed (can handle more traffic). Simplified backends: No SSL configuration needed. Focus on business logic (not encryption). Certificate management: Centralized: One certificate on NGINX (not 10 certificates on backends). Easy renewal: Update NGINX only (not all backends). Performance: SSL handshake: Done once at NGINX (not per backend). Reuse: NGINX reuses SSL sessions (TLS session resumption). Latency: Reduced (fewer handshakes). Scalability: Backends scale independently (add more backends easily). No SSL overhead per backend. DETAILED FLOW: BEFORE (NO PROXY): Client → Backend Server (HTTPS): (1) TCP handshake: 50ms. (2) TLS handshake: 150ms. (3) Decrypt request: 10ms CPU. (4) Process: 50ms. (5) Encrypt response: 10ms CPU. Total: 270ms + CPU overhead. AFTER (WITH SSL TERMINATION): Client → NGINX (HTTPS): (1) TCP handshake: 50ms. (2) TLS handshake: 150ms (once). NGINX → Backend (HTTP, internal): (3) HTTP request: 5ms (no encryption). (4) Process: 50ms. (5) HTTP response: 5ms (no encryption). Total: 260ms (slightly faster). CPU on backend: 70% reduction in SSL CPU (10ms → 0ms per request). SECURITY CONSIDERATIONS: Internal network encryption: NGINX → Backend is HTTP (unencrypted). Risk: If internal network compromised, traffic visible. Mitigation: (1) Use private network/VPC (isolated). (2) Or: Use mTLS (mutual TLS) between NGINX and backends (added complexity). Recommendation: HTTP internally is standard practice (AWS, Google, Netflix do this). Trust model: Backends trust NGINX (NGINX authenticates clients). NGINX forwards user context: Headers: X-Forwarded-For (client IP), X-Forwarded-Proto (https), X-User-ID (if authenticated). NGINX OPTIMIZATIONS: SSL session caching: ssl_session_cache shared:SSL:10m;. ssl_session_timeout 10m;. Reuse: TLS sessions reused for returning clients (skip handshake). Hardware acceleration: Use SSL acceleration (Intel AES-NI, hardware offload). OCSP stapling: ssl_stapling on;. Reduces client-side OCSP lookups. HTTP/2: http2 on;. Multiplexing: Multiple requests over one connection. COST ANALYSIS: NGINX instance: Medium server: $100/month. Handles: 10K concurrent connections. Backend savings: Before: 10 backends at 80% CPU → Need 15 backends (scale up). After: 10 backends at 50% CPU → Keep 10 backends. Savings: 5 backends × $100/month = $500/month. Net: $100 (NGINX) - $500 (savings) = Save $400/month. Plus: Simplified operations (one certificate, not 10). MONITORING: Track metrics: NGINX CPU: Should be <70% (SSL load). Backend CPU: Should decrease 30% (SSL offloaded). SSL handshake time: <200ms. TLS session reuse rate: >80% (caching working). Alert if: NGINX CPU >80% (need to scale NGINX). Backend CPU not reduced (misconfiguration). MIGRATION STRATEGY: Phase 1: Deploy NGINX with SSL termination (in parallel). Phase 2: Route 10% traffic through NGINX (canary). Phase 3: Monitor backend CPU (should drop). Phase 4: Gradually increase to 100%. Phase 5: Remove SSL from backends (simplify config). REAL-WORLD EXAMPLES: NETFLIX: SSL termination at ELB (AWS Load Balancer). Backends: HTTP only. Result: Simplified backends, better performance. CLOUDFLARE: Global SSL termination at edge. Origin servers: HTTP (Cloudflare encrypts). Result: Free SSL for customers, offloaded SSL. FINAL RECOMMENDATION: Implement NGINX reverse proxy with SSL termination. Backends accept HTTP only (internal network). Monitor backend CPU reduction (30% expected). Scale NGINX horizontally if needed (multiple NGINX instances). Expected outcome: Backend CPU: 80% → 50% (30% reduction). Simplified backends (no SSL config). Centralized certificate management (easy renewals). Can handle 50% more traffic (CPU freed).",
          keyPoints: [
            'SSL Termination: NGINX handles SSL, backends use HTTP (internal)',
            'Backend CPU reduction: 30% (SSL offloaded to NGINX)',
            'Centralized certificates: One cert on NGINX (not 10 on backends)',
            'Internal HTTP acceptable: Private network/VPC (industry standard)',
            'Result: 50% more traffic capacity, simplified backend configuration',
          ],
        },
        {
          id: 'q3',
          question:
            'A company wants employees to be unable to access social media during work hours. Design a solution using a forward proxy.',
          sampleAnswer:
            'SOLUTION: CORPORATE FORWARD PROXY WITH CONTENT FILTERING. REQUIREMENTS: Block: Social media sites (Facebook, Twitter, Instagram, TikTok). Allow: Work-related sites (email, productivity tools, research). Time-based: Block during work hours (9am-5pm), allow after hours. Logging: Track all employee web access (compliance). Exceptions: Marketing team needs social media access (whitelist). ARCHITECTURE: All employee traffic → Forward Proxy → Internet. No direct internet access (firewall blocks). Proxy is bottleneck (all traffic goes through). IMPLEMENTATION: STEP 1: DEPLOY PROXY SERVER. Software: Squid Proxy (open-source, widely used). Or: Blue Coat ProxySG (commercial, advanced features). Or: Cloud-based: Zscaler, Cisco Umbrella. Server: Medium instance (4 CPUs, 8GB RAM). Handles: 1000 concurrent employees. STEP 2: CONFIGURE PROXY SETTINGS. Network configuration: Firewall blocks direct internet (force proxy usage). DHCP: Automatically configure employee devices: HTTP Proxy: proxy.company.com:8080. HTTPS Proxy: proxy.company.com:8080. Or: PAC file (Proxy Auto-Config): Auto-detect corporate network, apply proxy. Employee devices: Browsers configured automatically. No manual configuration needed. STEP 3: CONFIGURE BLOCKED SITES. Blocklist: facebook.com, twitter.com, instagram.com, tiktok.com, youtube.com (entertainment), reddit.com, pinterest.com. Wildcard: *.facebook.com (blocks all Facebook subdomains). m.facebook.com, www.facebook.com, etc. Categories: Social media, entertainment, gambling, adult content. Use pre-built lists: Squid blacklists (community-maintained). Commercial feeds (Blue Coat, Symantec). STEP 4: CONFIGURE TIME-BASED RULES. Work hours (9am-5pm): Block social media. Allow work sites. After hours (5pm-9am): Allow all sites (personal use OK). Weekends: Allow all sites. Squid config: acl work_hours time MTWHF 09:00-17:00. acl social_media dstdomain .facebook.com .twitter.com. http_access deny social_media work_hours. http_access allow all. STEP 5: CONFIGURE WHITELIST (EXCEPTIONS). Marketing team: Needs social media access (Facebook, Twitter for campaigns). Implementation: Create group: marketing_team. Members: User IDs of marketing employees. Rule: Allow social_media for marketing_team (even during work hours). Squid config: acl marketing_team src 10.0.1.0/24 # Marketing subnet. http_access allow social_media marketing_team. STEP 6: CONFIGURE LOGGING. Log all requests: Timestamp, user ID, URL, action (allow/block), bytes transferred. Store logs: Centralized logging server (Splunk, ELK). Retention: 90 days (compliance). Analysis: Generate reports: Top blocked sites, top users by bandwidth, policy violations. BENEFITS: Productivity: Employees can\'t waste time on social media. Bandwidth: Reduce non-work traffic (save bandwidth costs). Security: Block malicious sites (malware, phishing). Compliance: Audit trail of all web access. Policy enforcement: Automatic (no manual monitoring). EXAMPLE BLOCKED REQUEST: Scenario: Employee tries to access facebook.com during work hours. Flow: (1) Employee browser: GET facebook.com. (2) Browser sends to proxy: GET http://facebook.com. (3) Proxy checks rules: URL: facebook.com. Time: 2pm (work hours). User: john@company.com (not marketing team). (4) Proxy finds match: social_media + work_hours → BLOCK. (5) Proxy returns: HTTP 403 Forbidden. Body: "Access to social media blocked during work hours. Contact IT for exceptions.". (6) Employee sees block page. (7) Proxy logs: 2024-10-15 14:00:00, john@company.com, facebook.com, BLOCKED. EXAMPLE ALLOWED REQUEST: Scenario: Employee accesses work email (gmail.com). Flow: (1) Employee: GET gmail.com. (2) Proxy checks rules: URL: gmail.com (not in blocklist). (3) Proxy: ALLOW. (4) Proxy forwards to gmail.com: GET http://gmail.com (on behalf of employee). (5) Gmail responds to proxy. (6) Proxy forwards response to employee. (7) Proxy logs: 2024-10-15 14:00:00, john@company.com, gmail.com, ALLOWED. BYPASS PREVENTION: Employees try to bypass: VPN: Proxy blocks VPN sites (openvpn.com, nordvpn.com). Tor: Proxy blocks Tor (torproject.org). Proxy websites: Proxy blocks proxy sites (hidemyass.com). HTTPS: Proxy does SSL inspection (decrypt, inspect, re-encrypt). Caution: Privacy concerns (inspect personal data). Recommendation: Warn employees (acceptable use policy). Mobile hotspot: Employee uses phone as hotspot (bypass corporate network). Prevention: Require VPN for remote work (force traffic through proxy). MONITORING & REPORTING: Daily reports: Top blocked sites (understand employee behavior). Top bandwidth users (identify heavy users). Policy violations (repeated access attempts). Monthly reports: Compliance: All web access logged. Trends: Popular sites, bandwidth usage. Alerts: Unusual activity: Employee accesses 100 blocked sites/hour (malware?). Bandwidth spike: Employee downloads 10GB (investigation). EMPLOYEE COMMUNICATION: Acceptable use policy: Document: "Social media blocked during work hours. Exceptions for marketing team. Violations logged.". Training: Explain policy, consequences. Transparency: Employees know proxy exists (not secret). COST: Proxy server: Medium instance: $200/month. Or: Cloud-based (Zscaler): $5/user/month × 1000 = $5000/month. Software: Squid: Free (open-source). Blue Coat: $10K/year (commercial). Maintenance: IT admin: 10 hours/month × $50/hour = $500/month. Total: ~$700/month (self-hosted). Benefits: Productivity: Hard to quantify, but significant. Bandwidth: 20-30% reduction (social media blocked). Compliance: Audit trail (avoid fines). ALTERNATIVES CONSIDERED: DNS filtering: Block social media at DNS level (cloudflare-gateway.com). Pros: Simple, no proxy needed. Cons: Easy to bypass (change DNS), no logging. Verdict: Proxy better (can\'t bypass, detailed logs). Endpoint software: Install agent on employee devices (block sites locally). Pros: Works off-network (remote employees). Cons: Can be disabled by employee, maintenance burden. Verdict: Use both (proxy on-network, agent for remote). FINAL RECOMMENDATION: Deploy Squid forward proxy on corporate network. Block social media during work hours (9am-5pm). Whitelist marketing team (exceptions). Log all requests (90-day retention). Prevent bypasses (block VPN, Tor, proxy sites). Communicate policy to employees (transparency). Expected outcome: 30-40% reduction in non-work web traffic. Improved productivity. Compliance with acceptable use policy.',
          keyPoints: [
            'Forward proxy filters content: block social media during work hours (9am-5pm)',
            'Network enforced: firewall blocks direct internet, all traffic goes through proxy',
            'Time-based rules: block during work hours, allow after hours',
            'Whitelist exceptions: marketing team needs social media access',
            'Logging: track all employee web access for compliance (90-day retention)',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What is the main difference between a Forward Proxy and a Reverse Proxy?',
          options: [
            'Forward proxy is faster than reverse proxy',
            'Forward proxy serves clients (hides client from server), reverse proxy serves servers (hides server from client)',
            'Forward proxy only works with HTTP, reverse proxy works with HTTPS',
            'Forward proxy is for internal networks, reverse proxy is for external networks',
          ],
          correctAnswer: 1,
          explanation:
            "Forward Proxy: Serves clients, sits on client-side, hides client IP from servers. Example: Corporate proxy, VPN. Reverse Proxy: Serves servers, sits on server-side, hides server IP from clients. Example: NGINX, load balancer. Clients know about forward proxy (configured in browser), don't know about reverse proxy (transparent).",
        },
        {
          id: 'mc2',
          question:
            'What is SSL Termination, and why would you use it at a reverse proxy?',
          options: [
            'Blocking SSL connections for security',
            'Decrypting HTTPS at the proxy, forwarding HTTP to backends',
            'Encrypting all traffic end-to-end',
            'Removing SSL certificates from servers',
          ],
          correctAnswer: 1,
          explanation:
            'SSL Termination: Reverse proxy decrypts HTTPS (from clients), forwards HTTP to backends (internal network). Benefits: (1) Offload CPU-intensive SSL from backends (30% CPU reduction). (2) Centralized certificate management (one cert on proxy). (3) Simplified backends (no SSL config). Used by: Netflix, Google, AWS. Security: Internal HTTP acceptable (private network/VPC).',
        },
        {
          id: 'mc3',
          question:
            'A corporate network uses a forward proxy. An employee configures their browser to bypass the proxy. What happens?',
          options: [
            'The employee can access the internet normally',
            'The firewall blocks the direct connection (proxy is mandatory)',
            'The proxy automatically reconfigures the browser',
            'The employee gets faster internet access',
          ],
          correctAnswer: 1,
          explanation:
            "Properly configured corporate network: Firewall blocks all direct internet access. Only proxy allowed (whitelist proxy IP). Employee bypassing proxy: Browser tries direct connection. Firewall blocks (no route to internet). Result: No internet access. This enforces proxy usage (employees can't bypass). Some networks allow direct access (poor security).",
        },
        {
          id: 'mc4',
          question:
            'Which of the following is NOT typically a responsibility of a reverse proxy?',
          options: [
            'Load balancing across backend servers',
            'SSL termination',
            'Content filtering (blocking websites)',
            'Caching static content',
          ],
          correctAnswer: 2,
          explanation:
            "Reverse proxy responsibilities: Load balancing (distribute traffic), SSL termination (offload encryption), caching (reduce backend load), compression, security. Content filtering (blocking websites): Forward proxy responsibility (client-side). Reverse proxy doesn't filter based on destination (it proxies to known backends). Option 3 is forward proxy task.",
        },
        {
          id: 'mc5',
          question:
            'Your application uses NGINX as a reverse proxy. Where should SSL certificates be installed?',
          options: [
            'On each backend server',
            'On the NGINX server (proxy) only',
            'On both NGINX and backend servers',
            'On the client browsers',
          ],
          correctAnswer: 1,
          explanation:
            'With SSL termination at reverse proxy: Install certificate on NGINX only (not backends). NGINX: Handles HTTPS from clients, decrypts, forwards HTTP to backends. Backends: Accept HTTP only (no SSL config needed). Benefits: Centralized certificate management, simplified backends, lower CPU on backends. Backends trust NGINX (internal network).',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Load balancers distribute traffic across multiple servers for availability and horizontal scaling',
    'Algorithm choice matters: Round Robin for simple/homogeneous servers, Least Connections for long-lived connections, Weighted for heterogeneous capacity',
    'Layer 4 (fast, protocol-agnostic) vs Layer 7 (intelligent HTTP routing, SSL termination) - choose based on needs',
    'Health checks are critical: detect and remove unhealthy servers automatically',
    'Avoid sticky sessions: design stateless servers with shared session storage (Redis)',
    'Caching dramatically reduces database load: 90% cache hit rate = 10× less database QPS',
    'Cache-aside most common pattern: check cache first, query DB on miss, populate cache',
    'LRU most common eviction policy: keeps hot data, evicts least recently used',
    'Cache invalidation is hard: use TTL + explicit invalidation for consistency',
    'Prevent cache stampede: jittered TTL or probabilistic early refresh',
    'Sharding = horizontal partitioning: splits data across machines by rows',
    'Partition key critical: choose high-cardinality, uniformly distributed key',
    'Consistent hashing: industry standard, minimal rebalancing when adding shards',
    'Avoid cross-shard joins: denormalize or colocate related data',
    'Replication = copying data to multiple databases for availability and read scaling',
    'Async replication: fast but eventual consistency (most common in practice)',
    'Sync replication: slow but strong consistency (use for critical data)',
    'Failover: automatic preferred, 30-90 second downtime typical',
    'Message queues decouple services, enable async processing, and absorb traffic spikes',
    'At-least-once delivery most common (idempotent consumers required)',
    'Dead Letter Queue handles failed messages after retries',
    'Queue for task distribution, Topic for event broadcasting',
    'CDN caches content on edge servers worldwide for low latency (10-50ms vs 200-300ms)',
    'Pull CDN most common (lazy loading, automatic), Push CDN for predictable traffic',
    'Versioned URLs best practice for cache invalidation (instant, free, automated)',
    '90-95% cache hit rate typical for well-configured CDN',
    'API Gateway = single entry point for all client requests (centralized auth, rate limiting)',
    'Use API Gateway + Load Balancers together (gateway for routing/auth, LB for distribution)',
    'Forward Proxy serves clients (hides client from server, content filtering)',
    'Reverse Proxy serves servers (hides server from client, load balancing, SSL termination)',
    'SSL termination at reverse proxy: 30% backend CPU reduction, centralized certificates',
  ],
  learningObjectives: [
    "Understand what load balancing is and why it's critical for distributed systems",
    'Master different load balancing algorithms and when to use each',
    'Learn the difference between Layer 4 and Layer 7 load balancing',
    'Implement health checks to detect and handle server failures',
    'Design stateless systems without relying on sticky sessions',
    'Master caching strategies to dramatically reduce database load',
    'Understand cache reading patterns and eviction policies',
    'Understand data partitioning (sharding) for horizontal scalability',
    'Master sharding strategies: hash-based, consistent hashing, range-based',
    'Choose effective partition keys and handle cross-shard operations',
    'Understand database replication for availability and read scalability',
    'Differentiate between synchronous and asynchronous replication',
    'Master failover processes and preventing split-brain scenarios',
    'Understand message queues for asynchronous service communication',
    'Differentiate between Queues (point-to-point) and Topics (pub/sub)',
    'Implement idempotent consumers and Dead Letter Queues',
    'Understand CDN architecture and how it reduces latency globally',
    'Master Pull vs Push CDN strategies and when to use each',
    'Implement cache invalidation using versioned URLs',
    'Understand API Gateway role in microservices architecture',
    'Master API Gateway responsibilities: routing, auth, rate limiting',
    'Learn when to use API Gateway vs Load Balancer vs both together',
    'Understand Forward Proxy vs Reverse Proxy differences',
    'Implement SSL termination at reverse proxy for performance',
    'Design corporate content filtering using forward proxy',
  ],
};
