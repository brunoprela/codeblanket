/**
 * DNS (Domain Name System) Section
 */

export const dnssystemSection = {
  id: 'dns-system',
  title: 'DNS (Domain Name System)',
  content: `DNS is one of the most critical yet often overlooked components of the internet. Understanding DNS deeply is essential for system design, especially for scalability and reliability discussions.

## What is DNS?

**DNS (Domain Name System)** translates human-readable domain names into IP addresses.

**Example**:
\`\`\`
User types: www.example.com
DNS resolves: 93.184.216.34
Browser connects to: 93.184.216.34
\`\`\`

**Why DNS exists**:
- Humans remember names better than numbers
- IP addresses can change without affecting the domain
- One domain can map to multiple IPs (load balancing)
- Abstraction layer for infrastructure changes

---

## DNS Hierarchy

DNS is a **distributed hierarchical system**:

\`\`\`
Root DNS Servers (.)
    ↓
TLD Servers (.com, .org, .net)
    ↓
Authoritative Name Servers (example.com)
    ↓
Subdomain Servers (api.example.com)
\`\`\`

### **Root Servers**
- 13 root server addresses (a.root-servers.net through m.root-servers.net)
- Actually 1000+ physical servers via Anycast
- Know where to find TLD servers
- Rarely queried directly (heavily cached)

### **TLD (Top-Level Domain) Servers**
- Manage .com, .org, .net, .edu, country codes (.uk, .de)
- Know where to find authoritative servers for each domain
- Operated by domain registrars

### **Authoritative Name Servers**
- Final source of truth for a domain
- Return the actual IP address
- Managed by domain owner or DNS provider

### **Recursive Resolvers**
- Your ISP's DNS or public DNS (8.8.8.8, 1.1.1.1)
- Does the hard work of querying the hierarchy
- Caches results

---

## DNS Query Flow

### **Recursive Query** (most common)

User\'s perspective: One request, one response.

\`\`\`
User
  ↓
"What is example.com?"
  ↓
Recursive Resolver (8.8.8.8)
  ↓
[Resolver does all the work]
  ↓
"93.184.216.34"
  ↓
User
\`\`\`

Behind the scenes:

\`\`\`
1. Recursive Resolver → Root Server
   Q: "Where is .com?"
   A: "Ask 192.5.6.30 (TLD server)"

2. Recursive Resolver → TLD Server
   Q: "Where is example.com?"
   A: "Ask 199.43.135.53 (authoritative server)"

3. Recursive Resolver → Authoritative Server
   Q: "What is example.com?"
   A: "93.184.216.34"

4. Recursive Resolver → User
   A: "93.184.216.34"
\`\`\`

**Latency**: 3 round trips (without caching)

### **Iterative Query**

Resolver tells client where to ask next:

\`\`\`
User → Root: "Where is example.com?"
Root → User: "Ask TLD server at 192.5.6.30"

User → TLD: "Where is example.com?"
TLD → User: "Ask authoritative at 199.43.135.53"

User → Authoritative: "What is example.com?"
Authoritative → User: "93.184.216.34"
\`\`\`

**Rarely used** (recursive is standard for end users)

---

## DNS Record Types

### **A Record** (Address)
Maps domain to IPv4 address.

\`\`\`
example.com.  IN  A  93.184.216.34
\`\`\`

### **AAAA Record**
Maps domain to IPv6 address.

\`\`\`
example.com.  IN  AAAA  2606:2800:220:1:248:1893:25c8:1946
\`\`\`

### **CNAME Record** (Canonical Name)
Alias from one domain to another.

\`\`\`
www.example.com.  IN  CNAME  example.com.
\`\`\`

**Use case**: Point multiple subdomains to main domain.

**Limitation**: Can't have CNAME at root (example.com can't be CNAME)

### **MX Record** (Mail Exchange)
Specifies mail server for domain.

\`\`\`
example.com.  IN  MX  10  mail.example.com.
example.com.  IN  MX  20  mail2.example.com.
\`\`\`

**Priority**: Lower number = higher priority

### **TXT Record**
Arbitrary text data.

**Use cases**:
- Domain verification (Google, Microsoft)
- SPF (email authentication)
- DKIM (email signing)
- DMARC (email policy)

\`\`\`
example.com.  IN  TXT  "v=spf1 include:_spf.google.com ~all"
\`\`\`

### **NS Record** (Name Server)
Specifies authoritative name servers for domain.

\`\`\`
example.com.  IN  NS  ns1.example.com.
example.com.  IN  NS  ns2.example.com.
\`\`\`

### **SOA Record** (Start of Authority)
Administrative information about zone.

\`\`\`
example.com.  IN  SOA  ns1.example.com. admin.example.com. (
                         2024010101 ; Serial
                         7200       ; Refresh
                         3600       ; Retry
                         1209600    ; Expire
                         86400 )    ; Minimum TTL
\`\`\`

### **SRV Record** (Service)
Specifies location of services.

\`\`\`
_http._tcp.example.com.  IN  SRV  10  60  80  server.example.com.
\`\`\`

### **CAA Record** (Certification Authority Authorization)
Specifies which CAs can issue certificates.

\`\`\`
example.com.  IN  CAA  0  issue  "letsencrypt.org"
\`\`\`

---

## DNS Caching & TTL

### **TTL (Time to Live)**

How long can DNS response be cached?

\`\`\`
example.com.  300  IN  A  93.184.216.34
              ^^^
              TTL in seconds (5 minutes)
\`\`\`

**Trade-offs**:

**Short TTL (60-300 seconds)**:
- ✅ Can change IP quickly
- ✅ Good for failover
- ❌ More DNS queries (higher load)
- ❌ Slower (more lookups)

**Long TTL (3600-86400 seconds)**:
- ✅ Fewer DNS queries
- ✅ Faster (cached longer)
- ❌ Slow to update (takes hours)
- ❌ Can't failover quickly

**Common strategy**: 
- Normal: 1 hour TTL
- Before change: Lower to 5 minutes
- After change: Raise back to 1 hour

### **DNS Caching Layers**

\`\`\`
Browser Cache (short, minutes)
    ↓
OS Cache (hours)
    ↓
Router Cache (ISP, hours)
    ↓
Recursive Resolver Cache (hours)
    ↓
[Query authoritative if not cached]
\`\`\`

**Impact**: ~95% of DNS queries answered from cache.

---

## DNS Load Balancing

### **Round-Robin DNS**

Return multiple A records, client picks one:

\`\`\`
example.com.  IN  A  1.2.3.4
example.com.  IN  A  5.6.7.8
example.com.  IN  A  9.10.11.12
\`\`\`

**Client behavior**: Uses first IP, or randomizes.

**Pros**:
- Simple
- No additional infrastructure

**Cons**:
- No health checking (send traffic to dead server)
- Uneven distribution (depends on TTL/caching)
- Can't route based on geography

### **GeoDNS / Geo-Routing**

Return different IPs based on user location:

\`\`\`
User in US      → 1.2.3.4 (US server)
User in Europe  → 5.6.7.8 (EU server)
User in Asia    → 9.10.11.12 (Asia server)
\`\`\`

**Providers**: AWS Route53, Cloudflare, NS1

**Benefits**:
- Lower latency (geographically closer)
- Compliance (data sovereignty)

### **Weighted Routing**

Control traffic distribution:

\`\`\`
Server A (weight 80) → 80% of traffic
Server B (weight 20) → 20% of traffic
\`\`\`

**Use cases**:
- Gradual rollout (canary deployment)
- A/B testing
- Cost optimization (send less traffic to expensive region)

### **Latency-Based Routing**

Route to lowest-latency endpoint for user.

AWS Route 53 measures latency and routes accordingly.

### **Failover Routing**

Health check endpoints, route away from unhealthy:

\`\`\`
Primary: 1.2.3.4 (healthy)
Secondary: 5.6.7.8 (standby)

If primary fails health check → route to secondary
\`\`\`

---

## DNS Security

### **DNS Spoofing / Cache Poisoning**

**Attack**: Attacker injects fake DNS response.

\`\`\`
User → Resolver: "What is bank.com?"
Attacker → Resolver: "It\'s 6.6.6.6" (fake)
Resolver caches fake response
User connects to attacker's server
\`\`\`

**Mitigation**:
- Use random source ports (harder to guess)
- Use random transaction IDs
- DNSSEC (cryptographic signatures)

### **DNSSEC (DNS Security Extensions)**

Cryptographically signs DNS responses.

**How it works**:
1. Authoritative server signs DNS records with private key
2. Publishes public key in DNS (DNSKEY record)
3. Resolver verifies signature

**Chain of trust**: Root → TLD → Domain

**Pros**:
- Prevents spoofing
- Guarantees authenticity

**Cons**:
- Complex to set up
- Larger DNS responses
- Low adoption (~30% of domains)

### **DNS over HTTPS (DoH)**

Encrypts DNS queries over HTTPS.

**Standard DNS**:
\`\`\`
User → 8.8.8.8:53 (UDP, plaintext)
ISP can see: "User is looking up adult-site.com"
\`\`\`

**DNS over HTTPS**:
\`\`\`
User → https://dns.google/resolve?name=example.com
ISP sees: "User is making HTTPS request to dns.google"
ISP can't see which domain
\`\`\`

**Pros**:
- Privacy (ISP can't see queries)
- Prevents DNS manipulation

**Cons**:
- Bypasses corporate DNS filtering
- Slightly higher latency (HTTPS overhead)

**Providers**:
- Google: https://dns.google/dns-query
- Cloudflare: https://1.1.1.1/dns-query
- Mozilla Firefox uses Cloudflare by default

### **DNS over TLS (DoT)**

Similar to DoH but uses dedicated port 853.

\`\`\`
User → 1.1.1.1:853 (TLS encrypted)
\`\`\`

**Difference from DoH**: Easier for firewalls to block (dedicated port).

---

## DNS Propagation

**Problem**: DNS changes take time to propagate globally.

**Why**:
1. TTL must expire on all caches
2. Some resolvers ignore TTL (cached longer)
3. Distributed system (13 root servers, thousands of resolvers)

**Timeline**:
- Minimum: TTL value (e.g., 5 minutes)
- Typical: 1-4 hours
- Maximum: 24-48 hours (worst case)

**How to speed up**:
1. Lower TTL before making changes
2. Wait for old TTL to expire
3. Make changes
4. Test from multiple locations
5. Raise TTL back to normal

**Check propagation**: whatsmydns.net

---

## DNS in System Design

### **Design Consideration 1: Failover**

Use DNS for automatic failover:

\`\`\`
Primary: us-east-1.example.com (1.2.3.4)
Secondary: us-west-2.example.com (5.6.7.8)

Health check every 30 seconds
If primary fails → switch DNS to secondary
\`\`\`

**Requirements**:
- Short TTL (60-300 seconds for fast failover)
- Health checking
- Automatic DNS update

**AWS Route53 Example**:
- Health check endpoint
- Failover policy: Primary/Secondary
- If health check fails 3 times → route to secondary

### **Design Consideration 2: Global Traffic Management**

Route users to nearest region:

\`\`\`
example.com → GeoDNS
    ├─ US users → us.example.com (1.2.3.4)
    ├─ EU users → eu.example.com (5.6.7.8)
    └─ Asia users → asia.example.com (9.10.11.12)
\`\`\`

**Benefits**:
- Lower latency (50-200ms reduction)
- Better user experience
- Compliance (data in-region)

### **Design Consideration 3: Blue-Green Deployment**

Use DNS for zero-downtime deployments:

\`\`\`
Step 1: example.com → blue environment (1.2.3.4)
Step 2: Deploy green environment (5.6.7.8)
Step 3: Test green thoroughly
Step 4: Switch DNS: example.com → green (5.6.7.8)
Step 5: Monitor, rollback to blue if issues
\`\`\`

**Challenge**: TTL means gradual switchover, not instant.

### **Design Consideration 4: Subdomain Strategy**

Organize services with subdomains:

\`\`\`
example.com → Main website
api.example.com → API
admin.example.com → Admin panel
cdn.example.com → Static assets
\`\`\`

**Benefits**:
- Different TTLs per service
- Independent scaling
- Easier to move services
- Better security (separate cookies)

---

## Real-World Examples

### **Netflix**

- Uses AWS Route53
- GeoDNS routes to nearest CDN edge
- Health checks for failover
- Short TTLs for quick updates

### **Cloudflare**

- Own authoritative DNS network
- 1.1.1.1 recursive resolver
- ~200ms average DNS response time
- Handles 20+ million DNS queries per second

### **Facebook**

- Custom DNS infrastructure
- Anycast for resilience
- Heavy caching (billions of lookups saved)

---

## DNS Performance Optimization

### **1. Use Anycast**

Same IP announced from multiple locations, network routes to closest:

\`\`\`
8.8.8.8 announced from:
  - US: 10 locations
  - Europe: 15 locations
  - Asia: 12 locations

User in Japan → routed to Tokyo instance
User in US → routed to New York instance
\`\`\`

**Result**: Lower latency (~20-50ms vs 200ms+)

### **2. Pre-resolve DNS**

\`\`\`html
<!-- Hint browser to resolve DNS early -->
<link rel="dns-prefetch" href="//api.example.com">
<link rel="dns-prefetch" href="//cdn.example.com">
\`\`\`

### **3. Minimize DNS Lookups**

Fewer domains = fewer lookups:

\`\`\`
Bad:
  cdn1.example.com
  cdn2.example.com
  cdn3.example.com
  (3 DNS lookups)

Good:
  cdn.example.com (with multiple IPs)
  (1 DNS lookup)
\`\`\`

### **4. Use CDN with Smart DNS**

Cloudflare, CloudFront automatically route to optimal edge.

---

## Common Mistakes

### ❌ **Long TTL before infrastructure change**

\`\`\`
TTL: 86400 (24 hours)
Change IP address
Users stuck on old IP for 24 hours!
\`\`\`

**Fix**: Lower TTL to 60 seconds, wait 24 hours, then change.

### ❌ **CNAME at root**

\`\`\`
example.com.  CNAME  other.com.  ← INVALID
\`\`\`

**Why**: RFC forbids CNAME at zone apex (conflicts with SOA, NS records).

**Fix**: Use A record or ALIAS record (Route53).

### ❌ **Not considering DNS propagation**

Deploy at 5pm, DNS hasn't propagated, users see errors.

**Fix**: Deploy during low-traffic hours, use gradual rollout.

### ❌ **Single point of failure**

Only one DNS provider → Provider outage = your site down.

**Fix**: Use multiple DNS providers (Route53 + Cloudflare).

---

## Interview Tips

### **Question: "How does DNS work?"**

**Good answer structure**:
1. User requests domain
2. Recursive resolver queries root, TLD, authoritative servers
3. Returns IP address
4. Cached with TTL
5. Mention: Most queries answered from cache

### **Question: "Design a globally distributed application"**

**Include DNS**:
- GeoDNS to route users to nearest region
- Health checks for automatic failover
- Short TTL for quick updates
- Multiple DNS providers for reliability

### **Question: "How would you handle DNS-based DDoS?"**

- Use Anycast (distributes load)
- Rate limiting at DNS level
- Cloudflare or similar DDoS protection
- Don't expose authoritative servers directly

### **Question: "Why is DNS slow sometimes?"**

- Cold cache (first lookup takes 3 round trips)
- Authoritative server far away
- Resolver overloaded
- Network issues

**Mitigation**:
- Use fast public DNS (1.1.1.1, 8.8.8.8)
- Anycast for geographic proximity
- Longer TTLs for popular domains

---

## Key Takeaways

1. **DNS translates** domain names to IP addresses via hierarchical system
2. **Recursive query**: Resolver does all work (root → TLD → authoritative)
3. **Caching with TTL** means ~95% of queries answered without hitting authoritative
4. **Record types**: A (IPv4), AAAA (IPv6), CNAME (alias), MX (mail), TXT (text)
5. **DNS load balancing**: Round-robin, geo-routing, weighted, latency-based, failover
6. **Security**: DNSSEC (signatures), DoH (encryption), prevent cache poisoning
7. **System design**: Use for failover, global routing, blue-green deployments
8. **Performance**: Anycast, short TTL for changes, minimize lookups`,
};
