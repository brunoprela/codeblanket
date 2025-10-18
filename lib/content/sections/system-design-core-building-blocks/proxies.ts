/**
 * Proxies (Forward & Reverse) Section
 */

export const proxiesSection = {
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
};
