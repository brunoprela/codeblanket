/**
 * Load Balancing Section
 */

export const loadbalancingSection = {
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
};
