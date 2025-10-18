/**
 * CDN (Content Delivery Network) Section
 */

export const cdnSection = {
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
};
