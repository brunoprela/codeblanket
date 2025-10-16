/* eslint-disable no-useless-escape */
import { Module } from '../types';

export const systemDesignNetworkingModule: Module = {
  id: 'system-design-networking',
  title: 'Networking & Communication',
  description:
    'Master networking protocols, communication patterns, and distributed system communication',
  icon: '🌐',
  category: 'System Design',
  difficulty: 'Medium',
  estimatedTime: '3-4 hours',
  sections: [
    {
      id: 'http-https-fundamentals',
      title: 'HTTP/HTTPS Fundamentals',
      content: `HTTP (Hypertext Transfer Protocol) is the foundation of data communication on the web. Understanding HTTP/HTTPS deeply is essential for system design interviews and building distributed systems.

## What is HTTP?

**HTTP** is an **application-layer protocol** for transmitting hypermedia documents (like HTML). It's the protocol that powers the World Wide Web.

**Key Characteristics**:
- **Stateless**: Each request is independent; server doesn't remember previous requests
- **Client-Server**: Clear separation between requester (client) and provider (server)
- **Text-based**: Human-readable (unlike binary protocols)
- **Request-Response**: Client sends request, server sends response

---

## HTTP Request Structure

Every HTTP request consists of:

### 1. Request Line
\`\`\`
GET /api/users/123 HTTP/1.1
\`\`\`
- **Method**: GET (what action to perform)
- **Path**: /api/users/123 (which resource)
- **Version**: HTTP/1.1

### 2. Headers
\`\`\`
Host: api.example.com
User-Agent: Mozilla/5.0
Accept: application/json
Authorization: Bearer eyJhbGc...
Content-Type: application/json
\`\`\`

### 3. Body (optional)
\`\`\`json
{
  "name": "John Doe",
  "email": "john@example.com"
}
\`\`\`

---

## HTTP Methods (Verbs)

### **GET** - Retrieve data
- **Idempotent**: Multiple identical requests have same effect as one
- **Safe**: Doesn't modify server state
- **Cacheable**: Responses can be cached

**Example**:
\`\`\`
GET /api/users/123
\`\`\`

### **POST** - Create new resources
- **Not idempotent**: Multiple requests create multiple resources
- **Not safe**: Modifies server state
- **Usually not cacheable**

**Example**:
\`\`\`
POST /api/users
Content-Type: application/json

{
  "name": "Alice",
  "email": "alice@example.com"
}
\`\`\`

### **PUT** - Update/replace resource
- **Idempotent**: Multiple identical requests = one request
- **Not safe**: Modifies state
- **Full replacement**: Replaces entire resource

**Example**:
\`\`\`
PUT /api/users/123
Content-Type: application/json

{
  "name": "Alice Updated",
  "email": "alice.new@example.com"
}
\`\`\`

### **PATCH** - Partial update
- **Not necessarily idempotent** (depends on implementation)
- **Not safe**
- **Partial modification**: Only updates specified fields

**Example**:
\`\`\`
PATCH /api/users/123
Content-Type: application/json

{
  "email": "alice.new@example.com"
}
\`\`\`

### **DELETE** - Remove resource
- **Idempotent**: Deleting multiple times has same effect
- **Not safe**

**Example**:
\`\`\`
DELETE /api/users/123
\`\`\`

### **HEAD** - Like GET but no body
- Used to check if resource exists
- Get metadata without downloading content

### **OPTIONS** - Describe communication options
- Used for CORS preflight requests

---

## HTTP Status Codes

### **1xx - Informational**
- **100 Continue**: Client should continue request
- **101 Switching Protocols**: Upgrading to WebSocket

### **2xx - Success**
- **200 OK**: Request succeeded
- **201 Created**: Resource created successfully (POST)
- **202 Accepted**: Request accepted, processing async
- **204 No Content**: Success, but no content to return (DELETE)

### **3xx - Redirection**
- **301 Moved Permanently**: Resource permanently moved, update bookmarks
- **302 Found**: Temporary redirect
- **304 Not Modified**: Cached version is still valid
- **307 Temporary Redirect**: Like 302, but preserve HTTP method

### **4xx - Client Errors**
- **400 Bad Request**: Invalid request syntax
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource doesn't exist
- **405 Method Not Allowed**: GET requested on POST-only endpoint
- **409 Conflict**: Request conflicts with current state
- **429 Too Many Requests**: Rate limit exceeded

### **5xx - Server Errors**
- **500 Internal Server Error**: Generic server error
- **502 Bad Gateway**: Invalid response from upstream server
- **503 Service Unavailable**: Server overloaded or down
- **504 Gateway Timeout**: Upstream server timeout

**Interview Tip**: Know the difference between 401 and 403, 301 and 302!

---

## Important HTTP Headers

### **Caching Headers**

**Cache-Control**:
\`\`\`
Cache-Control: max-age=3600, public
Cache-Control: no-cache, no-store, must-revalidate
\`\`\`
- **max-age**: How long to cache (seconds)
- **public**: Can be cached by any cache (CDN, browser)
- **private**: Only browser cache (contains user-specific data)
- **no-cache**: Must revalidate before using cached version
- **no-store**: Never cache (sensitive data)

**ETag** (Entity Tag):
\`\`\`
ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
\`\`\`
- Version identifier for resource
- Client sends \`If-None-Match\` header on subsequent requests
- Server returns 304 if unchanged

**Example Flow**:
\`\`\`
1. GET /api/users/123
   Response: ETag: "abc123", body: {...}

2. GET /api/users/123
   Request: If-None-Match: "abc123"
   Response: 304 Not Modified (no body, use cached version)
\`\`\`

### **Security Headers**

**Authorization**:
\`\`\`
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=
\`\`\`

**Content-Security-Policy**:
\`\`\`
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'
\`\`\`
- Prevents XSS attacks

**Strict-Transport-Security** (HSTS):
\`\`\`
Strict-Transport-Security: max-age=31536000; includeSubDomains
\`\`\`
- Forces HTTPS

### **Content Negotiation Headers**

**Content-Type**:
\`\`\`
Content-Type: application/json; charset=utf-8
Content-Type: text/html
Content-Type: multipart/form-data
\`\`\`

**Accept**:
\`\`\`
Accept: application/json, text/plain, */*
\`\`\`

**Content-Encoding**:
\`\`\`
Content-Encoding: gzip
\`\`\`

### **CORS Headers**

**Access-Control-Allow-Origin**:
\`\`\`
Access-Control-Allow-Origin: *
Access-Control-Allow-Origin: https://example.com
\`\`\`

**Access-Control-Allow-Methods**:
\`\`\`
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
\`\`\`

---

## HTTPS (HTTP Secure)

**HTTPS = HTTP + TLS/SSL**

### Why HTTPS Matters

**Without HTTPS** (HTTP):
- ❌ Data sent in plaintext
- ❌ Anyone on network can read data (passwords, credit cards)
- ❌ Man-in-the-middle attacks possible
- ❌ No guarantee you're talking to real server

**With HTTPS**:
- ✅ All data encrypted
- ✅ Server authenticity verified (certificates)
- ✅ Data integrity (tampering detected)
- ✅ SEO benefits (Google ranks HTTPS higher)

### TLS/SSL Handshake (Simplified)

\`\`\`
Client                                  Server
  |                                       |
  |------ ClientHello ------------------>|
  |       (supported cipher suites)      |
  |                                       |
  |<----- ServerHello -------------------|
  |       (chosen cipher)                |
  |<----- Certificate -------------------|
  |       (public key + CA signature)    |
  |                                       |
  |-- Verify certificate --------------->|
  |   (check CA signature)               |
  |                                       |
  |------ Client Key Exchange ---------->|
  |       (encrypted with server's       |
  |        public key)                   |
  |                                       |
  |<----- Both parties derive ---------->|
  |       session keys                   |
  |                                       |
  |====== Encrypted communication =======|
\`\`\`

### SSL/TLS Certificates

**Components**:
- **Domain name**: example.com
- **Organization**: Company Inc.
- **Public key**: Used for encryption
- **Certificate Authority (CA) signature**: Proves authenticity

**Certificate Types**:
1. **Domain Validated (DV)**: Basic, verifies domain ownership only
2. **Organization Validated (OV)**: Verifies organization details
3. **Extended Validation (EV)**: Highest validation, shows green bar (less common now)

**Where to get certificates**:
- Let's Encrypt (free, automated)
- DigiCert, GlobalSign (paid)
- AWS Certificate Manager (free for AWS services)

---

## HTTP/1.1 vs HTTP/2 vs HTTP/3

### **HTTP/1.1** (1997)

**Characteristics**:
- One request per TCP connection (or sequential)
- Head-of-line blocking
- Text-based protocol
- Header redundancy (same headers sent repeatedly)

**Performance issues**:
\`\`\`
Time →
Connection 1: [Request 1 ----][Response 1 ----]
Connection 2:                   [Request 2 ----][Response 2 ----]
Connection 3:                                     [Request 3 ----][Response 3 ----]
\`\`\`

**Workarounds**:
- Domain sharding (use multiple domains)
- Concatenate files (bundle.js instead of 10 separate files)
- Sprite images
- Inlining small resources

### **HTTP/2** (2015)

**Major improvements**:

**1. Multiplexing**:
- Multiple requests/responses simultaneously over one connection
- No head-of-line blocking at HTTP layer

\`\`\`
Single TCP Connection:
[Req1 chunk][Req2 chunk][Req3 chunk][Res1 chunk][Res2 chunk]...
\`\`\`

**2. Header Compression (HPACK)**:
- Compresses headers using Huffman encoding
- Maintains header table (sends only differences)
- Saves bandwidth

**3. Server Push**:
- Server can push resources before client requests
- Example: Server knows HTML needs style.css, pushes it immediately

**4. Binary Protocol**:
- Frames instead of text
- More efficient parsing

**Benefits**:
- ✅ Faster page loads (50% improvement typical)
- ✅ Single connection (less overhead)
- ✅ Better mobile performance

### **HTTP/3** (2022)

**Major change**: Uses **QUIC** (UDP-based) instead of TCP

**Why UDP?**:
- TCP has head-of-line blocking at transport layer
- Packet loss blocks entire connection
- QUIC has independent streams

**Improvements over HTTP/2**:
- ✅ No head-of-line blocking (even at transport layer)
- ✅ Faster connection establishment (0-RTT resumption)
- ✅ Better mobile performance (survives IP changes)
- ✅ Built-in encryption (TLS 1.3 integrated)

**Adoption**: Growing (Google, Cloudflare, Facebook use it)

---

## HTTP Performance Optimization

### 1. **Enable Compression**
\`\`\`
Content-Encoding: gzip
\`\`\`
- Reduces response size by 70-90%
- Use gzip or Brotli

### 2. **Use Caching Effectively**
\`\`\`
Cache-Control: public, max-age=31536000, immutable
\`\`\`
- Cache static assets aggressively
- Use content hashing for cache busting (style.a1b2c3.css)

### 3. **Use ETags**
- Avoid sending unchanged resources

### 4. **Enable HTTP/2**
- Most modern servers support it
- Requires HTTPS

### 5. **Use CDN**
- Serve static assets from edge locations
- Reduces latency

### 6. **Optimize Images**
- WebP format (smaller than JPEG/PNG)
- Lazy loading
- Responsive images (srcset)

### 7. **Connection: keep-alive**
- Reuse TCP connections (default in HTTP/1.1)

---

## Real-World Examples

### **Twitter API**
\`\`\`
GET https://api.twitter.com/2/tweets/1234567890
Authorization: Bearer YOUR_TOKEN
Accept: application/json

Response:
{
  "data": {
    "id": "1234567890",
    "text": "Hello world!"
  }
}
\`\`\`

### **GitHub API**
Uses HTTP status codes effectively:
- **200**: Success
- **201**: Repository created
- **304**: Use cached version
- **422**: Validation failed (body explains which fields)

### **Stripe API**
- Idempotency keys for safe retries
- Comprehensive error codes
- Versioning via headers (\`Stripe-Version: 2022-11-15\`)

---

## Common Mistakes

### ❌ **Using GET for mutations**
\`\`\`
GET /api/users/delete/123  ← WRONG
\`\`\`
- Search engines/proxies might prefetch GET requests
- Should be \`DELETE /api/users/123\`

### ❌ **Not using proper status codes**
\`\`\`
Response: 200 OK
{
  "error": "User not found"  ← WRONG, should be 404
}
\`\`\`

### ❌ **Exposing sensitive data in URLs**
\`\`\`
GET /api/reset-password?token=secret123  ← Bad (logged in server logs)
\`\`\`
- Use POST with body instead

### ❌ **Not enabling HTTPS**
- Even for "non-sensitive" sites
- Session cookies can be stolen

### ❌ **Ignoring caching**
- Not setting Cache-Control headers
- Missing out on massive performance gains

---

## Interview Tips

### **Question: "How does HTTPS work?"**

**Good answer structure**:
1. HTTPS = HTTP + TLS
2. TLS handshake: Client and server negotiate cipher suite
3. Server sends certificate (verified by CA)
4. Key exchange establishes session keys
5. All subsequent data encrypted with symmetric encryption

### **Question: "Why use HTTP/2?"**

**Hit these points**:
- Multiplexing (multiple requests on one connection)
- Header compression
- Server push
- Binary protocol
- Concrete benefit: "Reduces page load time by ~50%"

### **Question: "What's the difference between 401 and 403?"**

- **401 Unauthorized**: You haven't authenticated (need to log in)
- **403 Forbidden**: You're authenticated, but don't have permission

### **Question: "How would you design an API for rate limiting?"**

- Return **429 Too Many Requests**
- Include headers:
  - \`X-RateLimit-Limit: 100\`
  - \`X-RateLimit-Remaining: 0\`
  - \`X-RateLimit-Reset: 1640000000\`
  - \`Retry-After: 60\`

---

## Best Practices

### **1. Use the right HTTP method**
- GET for reads
- POST for creates
- PUT for full updates
- PATCH for partial updates
- DELETE for deletes

### **2. Use proper status codes**
- Don't return 200 for everything
- Clients rely on status codes for error handling

### **3. Version your API**
- \`/v1/users\`, \`/v2/users\`
- Or use headers: \`Accept: application/vnd.myapi.v2+json\`

### **4. Always use HTTPS in production**
- Use Let's Encrypt for free certificates
- Redirect HTTP to HTTPS

### **5. Implement idempotency**
- Use idempotency keys for POST requests
- Prevents duplicate charges, duplicate records

### **6. Set appropriate cache headers**
- Immutable static assets: \`Cache-Control: public, max-age=31536000, immutable\`
- Dynamic data: \`Cache-Control: private, max-age=60\`
- Sensitive data: \`Cache-Control: no-store\`

### **7. Use compression**
- Enable gzip/Brotli on server
- Saves bandwidth and improves performance

---

## Key Takeaways

1. **HTTP is stateless**, request-response, text-based protocol
2. **HTTP methods** have specific semantics (GET is safe/idempotent, POST is not)
3. **Status codes** communicate outcome (2xx success, 4xx client error, 5xx server error)
4. **HTTPS** encrypts data and verifies server authenticity via TLS/SSL
5. **HTTP/2** multiplexes requests, compresses headers (50% faster)
6. **HTTP/3** uses QUIC (UDP) to eliminate head-of-line blocking
7. **Headers** control caching, security, content negotiation
8. **Performance**: Compression, caching, CDN, HTTP/2 are key optimizations`,
      multipleChoice: [
        {
          id: 'http-idempotent',
          question:
            'Which of the following HTTP methods is both safe and idempotent?',
          options: ['POST', 'GET', 'PATCH', 'All of the above'],
          correctAnswer: 1,
          explanation:
            "GET is both safe (doesn't modify server state) and idempotent (multiple identical requests have the same effect as one). POST is neither safe nor idempotent. PATCH modifies state so it's not safe, and its idempotency depends on implementation.",
        },
        {
          id: 'http-status-codes',
          question:
            "A user is authenticated but tries to access a resource they don't have permission for. What status code should the API return?",
          options: [
            '401 Unauthorized',
            '403 Forbidden',
            '404 Not Found',
            '400 Bad Request',
          ],
          correctAnswer: 1,
          explanation:
            "403 Forbidden is correct because the user is authenticated but not authorized. 401 Unauthorized means authentication is required (user hasn't logged in). This is a common interview question!",
        },
        {
          id: 'http2-benefit',
          question:
            'What is the primary advantage of HTTP/2 multiplexing over HTTP/1.1?',
          options: [
            'It uses UDP instead of TCP',
            'It eliminates the need for SSL/TLS',
            'It allows multiple requests/responses simultaneously on one connection',
            'It compresses the request body',
          ],
          correctAnswer: 2,
          explanation:
            'HTTP/2 multiplexing allows multiple requests and responses to be sent simultaneously over a single TCP connection, eliminating head-of-line blocking at the HTTP layer. HTTP/2 still uses TCP (not UDP), requires TLS, and header compression (not body compression) is a separate feature.',
        },
        {
          id: 'https-tls',
          question:
            'During the TLS handshake, which key type is used to encrypt the session key exchange?',
          options: [
            "The server's private key",
            "The server's public key (from the certificate)",
            'A symmetric session key',
            "The client's private key",
          ],
          correctAnswer: 1,
          explanation:
            "The client uses the server's public key (from the certificate) to encrypt the pre-master secret. The server then decrypts it using its private key. This allows secure key exchange over an insecure channel. After this exchange, both parties derive symmetric session keys for efficient encryption.",
        },
        {
          id: 'http-caching',
          question:
            'Which Cache-Control directive ensures a resource is NEVER cached, even in the browser?',
          options: [
            'Cache-Control: no-cache',
            'Cache-Control: private',
            'Cache-Control: no-store',
            'Cache-Control: max-age=0',
          ],
          correctAnswer: 2,
          explanation:
            'Cache-Control: no-store tells browsers and intermediary caches to never store the response. "no-cache" means you must revalidate before using cached version (it still caches). "private" means only browser can cache (not CDNs). "max-age=0" means expired immediately but still cached.',
        },
      ],
      quiz: [
        {
          id: 'http-api-design',
          question:
            "You're designing a RESTful API for an e-commerce platform. Explain how you would design the endpoints for managing shopping carts, including which HTTP methods you'd use, what status codes you'd return, and how you'd handle errors. Discuss trade-offs between different approaches.",
          sampleAnswer: `I would design the shopping cart API following REST principles with clear resource modeling and appropriate HTTP semantics:

**Endpoint Design**:

1. **GET /api/carts/:userId** - Retrieve user's cart
   - Returns 200 with cart contents
   - Returns 404 if cart doesn't exist yet
   - Idempotent and safe operation

2. **POST /api/carts/:userId/items** - Add item to cart
   - Body: { "productId": "123", "quantity": 2 }
   - Returns 201 Created with updated cart
   - Returns 400 if product doesn't exist or invalid quantity
   - Returns 409 Conflict if item exceeds available inventory

3. **PATCH /api/carts/:userId/items/:itemId** - Update quantity
   - Body: { "quantity": 5 }
   - Returns 200 with updated cart
   - Use PATCH (not PUT) since we're partially updating
   - Returns 404 if item not in cart

4. **DELETE /api/carts/:userId/items/:itemId** - Remove item
   - Returns 204 No Content
   - Idempotent (multiple deletes same as one)

**Error Handling**:
- 400 Bad Request: Invalid input (missing fields, negative quantity)
- 401 Unauthorized: User not authenticated
- 403 Forbidden: User trying to modify another user's cart
- 404 Not Found: Cart or item doesn't exist
- 409 Conflict: Business rule violation (out of stock)
- 429 Too Many Requests: Rate limit exceeded
- Include error body with details: { "error": "OUT_OF_STOCK", "message": "Only 3 items remaining" }

**Trade-offs**:

*Approach 1: Cart as subresource of user*
- GET /users/:userId/cart
- Pro: Clear ownership relationship
- Con: Longer URLs

*Approach 2: Cart as top-level resource*
- GET /carts/:cartId (cartId could be userId)
- Pro: Shorter URLs, better separation of concerns
- Con: Requires additional mapping

*Idempotency Keys*:
For adding items, consider accepting idempotency keys to prevent duplicate adds:
- Header: Idempotency-Key: uuid
- Prevents user clicking "Add to Cart" twice from adding duplicate items
- Trade-off: Adds complexity, requires storing keys temporarily

**Performance Considerations**:
- Cache cart contents with short TTL (30 seconds)
- Return ETag header for cart version
- Client sends If-None-Match to avoid unnecessary data transfer
- For high-traffic, consider write-through cache to Redis

I'd choose Approach 2 with idempotency keys for production, as it provides the best scalability and prevents common user errors while maintaining clear REST semantics.`,
          keyPoints: [
            'Use appropriate HTTP methods (GET for read, POST for add, PATCH for update, DELETE for remove)',
            'Return meaningful status codes (2xx success, 4xx client errors, 5xx server errors)',
            'Include detailed error messages in response body for client debugging',
            "Consider idempotency keys for operations that shouldn't be duplicated",
            'Discuss trade-offs between different URL structures and resource modeling',
            'Consider caching strategies and ETags for performance',
          ],
        },
        {
          id: 'http2-migration',
          question:
            "Your company is considering migrating from HTTP/1.1 to HTTP/2. Explain the benefits, potential challenges, and what changes you'd need to make to your existing infrastructure. Would you recommend HTTP/3, and why or why not?",
          sampleAnswer: `**HTTP/2 Benefits**:

1. **Multiplexing**: Multiple requests over single connection
   - Eliminates need for domain sharding
   - Reduces connection overhead (no TCP handshake per request)
   - Expected improvement: 30-50% faster page loads

2. **Header Compression (HPACK)**: 
   - Typical header size reduction: 80-90%
   - Significant for APIs with many small requests

3. **Server Push**: 
   - Push CSS/JS when HTML is requested
   - Reduces round trips

4. **Binary Protocol**: 
   - More efficient parsing
   - Less error-prone than text

**Migration Challenges**:

1. **HTTPS Required**: 
   - Must obtain SSL/TLS certificates
   - Cost: Let's Encrypt is free, but need automation
   - Slight CPU overhead for encryption

2. **Server Compatibility**: 
   - Need HTTP/2-capable servers (NGINX 1.9.5+, Apache 2.4.17+)
   - May require server upgrades

3. **Load Balancer Support**: 
   - Ensure load balancers support HTTP/2
   - AWS ALB supports it, but some older LBs don't

4. **CDN Compatibility**: 
   - Verify CDN supports HTTP/2 to origin
   - Most modern CDNs do (CloudFront, Cloudflare)

5. **Monitoring/Debugging**: 
   - Binary protocol harder to debug
   - Need tools that understand HTTP/2 frames

**Required Changes**:

1. Enable HTTPS across all services
2. Update server configurations (nginx.conf: \`listen 443 ssl http2;\`)
3. Update load balancers to support HTTP/2
4. Remove HTTP/1.1 optimizations that hurt HTTP/2:
   - Domain sharding (now anti-pattern)
   - File concatenation (defeats multiplexing)
   - Image sprites (defeats caching)
5. Update monitoring tools

**HTTP/3 Consideration**:

I would **NOT** recommend HTTP/3 immediately for these reasons:

1. **Less mature ecosystem**: 
   - Fewer servers/clients support it
   - Debugging tools less mature

2. **UDP may be blocked**: 
   - Corporate firewalls often block UDP
   - Falls back to HTTP/2, but adds complexity

3. **Benefits mainly for mobile/high-latency**: 
   - Connection migration (IP changes)
   - Less impactful for desktop users on stable connections

4. **Cost vs benefit**: 
   - HTTP/2 gives 80% of the benefit with 20% of the risk

**Recommendation Strategy**:

1. **Phase 1**: Migrate to HTTP/2 (next quarter)
   - Clear benefits, mature ecosystem
   - ROI: 30-50% performance improvement
   
2. **Phase 2**: Monitor HTTP/3 adoption (next 1-2 years)
   - Wait for broader support
   - Evaluate again when 50% of traffic supports it
   
3. **Phase 3**: Consider HTTP/3 for mobile-heavy products
   - If 40%+ traffic is mobile
   - High-latency markets (emerging countries)

**Metrics to Track**:
- Time to First Byte (TTFB)
- Page Load Time
- Number of TCP connections
- Header overhead reduction

Expected outcome: 40% faster page loads with HTTP/2, acceptable migration cost.`,
          keyPoints: [
            'HTTP/2 provides multiplexing, header compression, server push, and binary protocol benefits',
            'Migration requires HTTPS, server updates, load balancer compatibility, and removal of HTTP/1.1 workarounds',
            'HTTP/3 is less mature and primarily benefits mobile/high-latency scenarios',
            'Phase migration approach: HTTP/2 first, then evaluate HTTP/3 later',
            'Consider cost vs benefit: HTTP/2 offers significant improvement with manageable risk',
            'Monitor key metrics like TTFB, page load time, and connection count to measure success',
          ],
        },
        {
          id: 'https-security',
          question:
            'Explain how HTTPS protects against man-in-the-middle attacks. Walk through the TLS handshake process and explain what would happen if an attacker tried to intercept the connection. What additional security headers would you implement for a banking application?',
          sampleAnswer: `**HTTPS Protection Against MITM Attacks**:

HTTPS uses TLS/SSL to provide three security guarantees:
1. **Encryption**: Data can't be read
2. **Authentication**: You're talking to the real server
3. **Integrity**: Data can't be modified without detection

**TLS Handshake Process**:

1. **Client Hello**: 
   - Client sends supported cipher suites (encryption algorithms)
   - Random nonce for this session

2. **Server Hello**: 
   - Server chooses cipher suite (e.g., TLS_AES_128_GCM_SHA256)
   - Sends its certificate containing:
     * Server's public key
     * Domain name
     * Digital signature from Certificate Authority (CA)

3. **Certificate Verification**: 
   - **Critical step**: Client verifies certificate
   - Checks CA signature using pre-installed CA public keys
   - Validates domain matches certificate
   - Checks certificate hasn't expired
   - Checks certificate hasn't been revoked (OCSP/CRL)

4. **Key Exchange**: 
   - Client generates pre-master secret
   - Encrypts it with server's public key (from certificate)
   - Only server's private key can decrypt

5. **Session Keys Derived**: 
   - Both parties derive symmetric session keys
   - Used for fast encryption (symmetric is 1000x faster than asymmetric)

6. **Encrypted Communication**: 
   - All subsequent data encrypted with session keys

**How MITM Attack is Prevented**:

**Scenario**: Attacker sits between client and server

*Attack Attempt 1: Intercept and forward*:
- Attacker receives encrypted data
- Can't decrypt without session keys
- Can't derive session keys without server's private key
- **Result**: Attack blocked by encryption

*Attack Attempt 2: Pose as server*:
- Attacker sends own certificate to client
- Client verifies certificate signature
- Attacker's certificate not signed by trusted CA
- Browser shows warning: "Your connection is not private"
- **Result**: Attack blocked by authentication

*Attack Attempt 3: Create fake certificate*:
- Attacker would need CA's private key to sign certificate
- CAs guard private keys extremely carefully (HSMs)
- Compromising a CA is extremely difficult
- **Result**: Attack blocked by PKI infrastructure

**Additional Security Headers for Banking Application**:

1. **Strict-Transport-Security (HSTS)**:
\`\`\`
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
\`\`\`
- Forces HTTPS for 1 year
- Prevents downgrade attacks (attacker redirecting to HTTP)
- \`preload\`: Submit to Chrome's HSTS preload list

2. **Content-Security-Policy**:
\`\`\`
Content-Security-Policy: 
  default-src 'self'; 
  script-src 'self' 'nonce-random123'; 
  style-src 'self'; 
  img-src 'self' https:; 
  connect-src 'self'; 
  frame-ancestors 'none';
\`\`\`
- Prevents XSS attacks
- Only allows scripts from same origin
- Blocks clickjacking (frame-ancestors 'none')

3. **X-Frame-Options**:
\`\`\`
X-Frame-Options: DENY
\`\`\`
- Prevents clickjacking
- Bank page can't be embedded in iframe

4. **X-Content-Type-Options**:
\`\`\`
X-Content-Type-Options: nosniff
\`\`\`
- Prevents MIME type sniffing attacks

5. **Referrer-Policy**:
\`\`\`
Referrer-Policy: no-referrer
\`\`\`
- Doesn't leak URLs to external sites
- Important if URLs contain sensitive tokens

6. **Permissions-Policy**:
\`\`\`
Permissions-Policy: geolocation=(), microphone=(), camera=()
\`\`\`
- Disables unnecessary browser features

**Additional Banking Security Measures**:

1. **Certificate Pinning**: 
   - Mobile apps pin specific certificates
   - Even compromised CA can't MITM

2. **Mutual TLS (mTLS)**: 
   - Client also presents certificate
   - Proves client's identity

3. **Perfect Forward Secrecy (PFS)**: 
   - Use ephemeral keys (ECDHE)
   - Even if server's private key compromised, past sessions safe

4. **Implement CAA DNS records**: 
\`\`\`
example.com. CAA 0 issue "letsencrypt.org"
\`\`\`
- Only specified CAs can issue certificates
- Prevents rogue certificate issuance

**Monitoring**:
- Monitor Certificate Transparency logs for unauthorized certificates
- Set up alerts for certificate expiration
- Regular security audits (penetration testing)

This defense-in-depth approach makes MITM attacks effectively impossible against properly implemented HTTPS.`,
          keyPoints: [
            'TLS provides encryption, authentication, and integrity through certificate validation',
            "MITM attacks are blocked because attacker can't forge valid CA-signed certificates",
            'Certificate verification is critical - checks CA signature, domain, expiration, revocation',
            'Banking apps need additional headers: HSTS, CSP, X-Frame-Options, etc.',
            'Advanced measures include certificate pinning, mTLS, PFS, and CAA records',
            'Defense-in-depth: multiple layers of security working together',
          ],
        },
      ],
    },
    {
      id: 'tcp-vs-udp',
      title: 'TCP vs UDP',
      content: `TCP and UDP are the two primary transport-layer protocols in the Internet Protocol Suite. Understanding when to use each is crucial for system design decisions.

## The Transport Layer

The **transport layer** (Layer 4 in OSI model) is responsible for:
- End-to-end communication between applications
- Reliability (if needed)
- Flow control
- Multiplexing (multiple applications on one host)

Two main protocols: **TCP** and **UDP**

---

## TCP (Transmission Control Protocol)

**TCP** is a **connection-oriented**, **reliable**, **ordered** delivery protocol.

### Key Characteristics

**1. Connection-Oriented**:
- Must establish connection before sending data
- Three-way handshake

**2. Reliable**:
- Guarantees delivery
- Retransmits lost packets
- Acknowledges received packets

**3. Ordered**:
- Packets arrive in order sent
- Reorders out-of-order packets

**4. Flow Control**:
- Prevents sender from overwhelming receiver
- Sliding window protocol

**5. Congestion Control**:
- Detects network congestion
- Reduces send rate to avoid packet loss

---

## TCP Three-Way Handshake

Before data can be sent, TCP establishes a connection:

\`\`\`
Client                                Server
  |                                     |
  |------ SYN (seq=1000) -------------->|
  |       "Let's establish connection"  |
  |                                     |
  |<----- SYN-ACK (seq=5000, ack=1001) -|
  |       "Acknowledged, here's my seq" |
  |                                     |
  |------ ACK (ack=5001) -------------->|
  |       "Got it, let's start"         |
  |                                     |
  |====== Data transmission ============|
\`\`\`

**Why three-way?**
- Prevents old duplicate connections from causing confusion
- Both sides agree on initial sequence numbers
- Ensures both sides are ready

**Overhead**: 
- 1 round-trip time (RTT) before data can be sent
- For high-latency connections (e.g., 100ms RTT), that's 100ms delay
- This is why TCP can be slow for short requests

---

## TCP Data Transfer

Once connected, TCP ensures reliable delivery:

\`\`\`
Client                                Server
  |                                     |
  |------ Packet 1 (seq=1000) --------->|
  |       Data: "Hello"                 |
  |                                     |
  |<----- ACK (ack=1005) ---------------|
  |       "Got it"                      |
  |                                     |
  |------ Packet 2 (seq=1005) --------->|
  |       Data: "World"                 |
  |                                     |
  |  X--- Packet lost ---------X        |
  |                                     |
  |       [Timeout]                     |
  |                                     |
  |------ Retransmit Packet 2 --------->|
  |       Data: "World"                 |
  |                                     |
  |<----- ACK (ack=1010) ---------------|
\`\`\`

**Key mechanisms**:
- **Sequence numbers**: Track which bytes have been sent
- **Acknowledgments (ACKs)**: Confirm receipt
- **Retransmission timer**: Resend if ACK not received
- **Duplicate ACKs**: Signal lost packet (fast retransmit)

---

## TCP Flow Control

**Problem**: Sender might be faster than receiver

**Solution**: **Sliding window**

\`\`\`
Sender's view:
[Sent & ACKed][Sent, awaiting ACK][Can send now][Can't send yet]
              Window size = what receiver can handle
\`\`\`

Receiver tells sender: "I have 64KB buffer available"
Sender won't send more than 64KB before getting ACKs

**TCP window size**:
- Advertised in every ACK
- Dynamically adjusted based on receiver's buffer

---

## TCP Congestion Control

**Problem**: Too much traffic can conquer the network

**Solution**: TCP congestion control algorithms

### Slow Start

Start sending slowly, increase exponentially:
\`\`\`
Round 1: Send 1 packet
Round 2: Send 2 packets (if ACKed)
Round 3: Send 4 packets
Round 4: Send 8 packets
...
\`\`\`

Increase until:
- Packet loss occurs (network congested)
- Slow start threshold reached

### Congestion Avoidance

After slow start, increase linearly:
- Add 1 MSS (Maximum Segment Size) per RTT

### Fast Retransmit / Fast Recovery

If 3 duplicate ACKs received:
- Assume packet lost (don't wait for timeout)
- Retransmit immediately
- Cut congestion window in half

**Algorithms**:
- **TCP Reno**: Basic congestion control
- **TCP Cubic**: Modern (Linux default), better for high-bandwidth networks
- **BBR** (Bottleneck Bandwidth and RTT): Google's algorithm, optimizes for throughput

---

## UDP (User Datagram Protocol)

**UDP** is **connectionless**, **unreliable**, **unordered** protocol.

### Key Characteristics

**1. Connectionless**:
- No handshake
- Just send packets (datagrams)
- No connection state

**2. Unreliable**:
- No delivery guarantee
- Packets can be lost
- No retransmission

**3. Unordered**:
- Packets may arrive out of order
- No reordering

**4. No Flow Control**:
- Sender sends as fast as it wants

**5. No Congestion Control**:
- Doesn't adapt to network conditions

**6. Minimal Overhead**:
- 8-byte header (vs TCP's 20+ bytes)
- No connection setup latency

---

## UDP Datagram Structure

\`\`\`
Client                                Server
  |                                     |
  |------ Datagram 1 ------------------>|
  |       Data: "Hello"                 |
  |                                     |
  |------ Datagram 2 ------------------>|
  |       Data: "World"                 |
  |                                     |
  |------ Datagram 3 ------------------>|
  |  X--- Lost -----------------X       |
  |                                     |
  |       No retransmission!            |
  |       No notification!              |
\`\`\`

**No guarantees**:
- Datagram 2 might arrive before Datagram 1
- Datagram 3 might never arrive
- Application must handle these cases

---

## TCP vs UDP Comparison

| Feature | TCP | UDP |
|---------|-----|-----|
| **Connection** | Connection-oriented | Connectionless |
| **Reliability** | Guaranteed delivery | No guarantee |
| **Ordering** | Ordered | Unordered |
| **Speed** | Slower (overhead) | Faster (minimal overhead) |
| **Header size** | 20-60 bytes | 8 bytes |
| **Flow control** | Yes | No |
| **Congestion control** | Yes | No |
| **Use case** | Reliable delivery needed | Speed more important than reliability |
| **Examples** | HTTP, FTP, SSH, Email | DNS, VoIP, Video streaming, Gaming |

---

## When to Use TCP

Use TCP when:
- **Reliability is critical**
- **Order matters**
- **You can tolerate latency**

### Use Cases

**1. HTTP/HTTPS** (Web traffic):
- Must receive entire HTML page
- Order matters (can't render page with missing parts)
- User can wait 100ms

**2. File Transfer (FTP, SFTP)**:
- Can't have corrupted files
- Every byte must arrive
- Speed less critical than reliability

**3. Email (SMTP, IMAP)**:
- Can't lose emails
- Order not critical, but reliability is

**4. SSH/Remote Access**:
- Every keystroke must arrive
- Out-of-order commands would be confusing

**5. Database Connections**:
- Queries and results must be reliable
- Transactions require reliability

---

## When to Use UDP

Use UDP when:
- **Speed is more important than reliability**
- **Real-time is critical**
- **You can handle packet loss**

### Use Cases

**1. DNS (Domain Name System)**:
- Queries are small (single packet)
- If lost, just retry (application-level retry)
- TCP handshake would double latency
\`\`\`
TCP DNS: 
  1. SYN, SYN-ACK, ACK (1 RTT)
  2. Query, Response (1 RTT)
  Total: 2 RTT

UDP DNS:
  1. Query, Response (1 RTT)
  Total: 1 RTT
\`\`\`

**2. Video Streaming (Live)**:
- Old frames are useless (timestamp passed)
- Better to skip lost frame than wait for retransmit
- Example: Zoom, YouTube Live, Twitch

**3. Online Gaming**:
- Player position updates 60 times/second
- If packet lost, next update is coming in 16ms anyway
- Retransmitting old position is useless
- Example: Counter-Strike, Call of Duty

**4. VoIP (Voice over IP)**:
- Human voice can tolerate small losses
- Delay causes awkward pauses
- Better to have slight audio glitch than 200ms delay

**5. IoT Sensor Data**:
- Sending temperature every second
- Lost reading not critical (next one coming soon)
- Millions of devices, TCP overhead prohibitive

**6. Metrics/Logging**:
- StatsD sends metrics via UDP
- Losing occasional metric acceptable
- High volume, low individual importance

---

## Hybrid Approaches

### **QUIC (Quick UDP Internet Connections)**

- **Used by**: HTTP/3, Google services
- **Idea**: UDP + reliability built in application layer

**Why QUIC?**

TCP has limitations:
- Head-of-line blocking (one lost packet blocks everything)
- Slow connection setup (1-2 RTT)
- Hard to update (baked into OS kernel)

QUIC advantages:
- **0-RTT connection**: Establish connection + send data in one round trip
- **No head-of-line blocking**: Independent streams
- **Survives IP changes**: Mobile devices switching networks
- **Encrypted by default**: Built-in TLS 1.3

\`\`\`
TCP + TLS:
  1. TCP handshake (SYN, SYN-ACK, ACK)
  2. TLS handshake
  Total: 2-3 RTT before data

QUIC:
  1. Connection + encryption + data
  Total: 0-1 RTT
\`\`\`

**Adoption**:
- HTTP/3 uses QUIC
- Google search, YouTube, Gmail use QUIC
- ~25% of internet traffic now uses QUIC

### **RTP (Real-time Transport Protocol)**

- Built on UDP
- Adds sequence numbers and timestamps
- Used for VoIP, video conferencing
- Application decides what to do with lost packets

### **SCTP (Stream Control Transmission Protocol)**

- Combines benefits of TCP and UDP
- Multiple streams in one connection
- No head-of-line blocking
- Less widely supported

---

## TCP Optimizations

### **TCP Fast Open (TFO)**

Skip the full 3-way handshake on subsequent connections:
\`\`\`
First connection: Normal 3-way handshake
Server gives client a cookie

Subsequent connections:
Client sends SYN + Cookie + Data (all in one packet!)
Server validates cookie, sends SYN-ACK + Response
\`\`\`

**Benefit**: Saves 1 RTT

**Adoption**: Supported by Linux, macOS, iOS

### **TCP BBR (Bottleneck Bandwidth and RTT)**

- Google's congestion control algorithm
- Measures actual bottleneck bandwidth
- Better than Cubic for high-bandwidth networks
- 2-5x faster for YouTube, Google services

### **TCP Keepalive**

- Sends periodic probes to check if connection alive
- Prevents middleboxes from timing out connection
- Configurable interval (default: 2 hours)

---

## Real-World Examples

### **Netflix**

- **Video streaming**: TCP (not UDP!)
- Why? Video is pre-recorded, not live
- Can buffer ahead
- Every byte must arrive (or video corrupted)
- Uses adaptive bitrate over TCP

### **Zoom**

- **Video/audio**: UDP primary, TCP fallback
- UDP for real-time with custom reliability layer
- Falls back to TCP if firewall blocks UDP
- Implements own jitter buffer, packet loss concealment

### **Discord**

- **Voice chat**: UDP with custom reliability
- **Text chat**: TCP (must be reliable)
- **File transfer**: TCP

### **Google Meet**

- Uses WebRTC over UDP (SRTP protocol)
- Falls back to TCP if UDP blocked
- Implements own congestion control

### **Cloudflare**

- Supports QUIC (HTTP/3) for websites
- Falls back to TCP (HTTP/2) if client doesn't support

---

## Common Mistakes

### ❌ **Using UDP without considering packet loss**
\`\`\`python
# Bad: Just send and forget
udp_socket.sendto(critical_data, address)
# If packet lost, data gone forever!
\`\`\`

**Fix**: Implement application-level acknowledgments for critical data

### ❌ **Using TCP for real-time when UDP better**
- Example: Live video streaming over TCP
- Problem: Retransmits delay entire stream

### ❌ **Assuming UDP is always faster**
- For large data transfers, TCP's congestion control helps
- UDP can congest network if sending too fast

### ❌ **Not handling TCP connection failures**
- TCP connections can break
- Need timeouts, reconnection logic

### ❌ **Forgetting about firewalls**
- Many corporate firewalls block UDP
- Need TCP fallback mechanism

---

## Interview Tips

### **Question: "When would you use UDP over TCP?"**

**Good answer structure**:
1. "UDP is appropriate when speed/latency is more important than reliability"
2. Give concrete examples:
   - Live video streaming (old frames useless)
   - Online gaming (position updates every 16ms)
   - DNS (small request, can retry if lost)
3. Mention you'd handle packet loss at application layer if needed
4. Note that UDP is connectionless, so lower overhead

### **Question: "Explain TCP three-way handshake"**

**Hit these points**:
1. Client sends SYN with initial sequence number
2. Server responds SYN-ACK (acknowledges + sends its sequence)
3. Client sends ACK
4. Purpose: Agree on sequence numbers, ensure both sides ready
5. Overhead: 1 RTT before data can be sent

### **Question: "How does TCP ensure reliability?"**

**Key mechanisms**:
1. Sequence numbers (track bytes)
2. Acknowledgments (confirm receipt)
3. Retransmission timers (resend if ACK not received)
4. Checksums (detect corruption)
5. Flow control (prevent overwhelming receiver)

### **Question: "What is QUIC and why was it created?"**

**Answer**:
- QUIC is UDP-based protocol with reliability built in
- Solves TCP problems:
  - Head-of-line blocking (one lost packet blocks all streams)
  - Slow connection setup (1-2 RTT)
  - Can't update TCP (in kernel)
- Used by HTTP/3
- Benefits: 0-RTT connection, independent streams, survives IP changes

---

## Best Practices

### **1. Use TCP for reliability, UDP for speed**
- Default to TCP unless you have specific reason for UDP
- Don't reinvent TCP on top of UDP (unless you're QUIC)

### **2. Implement timeouts**
- TCP connections can hang
- Set appropriate socket timeouts
\`\`\`python
socket.settimeout(30)  # 30 seconds
\`\`\`

### **3. TCP: Enable keepalive for long-lived connections**
- Detects broken connections
- Prevents middlebox timeouts

### **4. UDP: Implement application-level reliability if needed**
- Sequence numbers
- Acknowledgments
- Retransmission logic

### **5. Consider QUIC for modern applications**
- Better performance than TCP
- Built-in encryption
- Check if your stack supports it

### **6. Handle UDP firewall issues**
- Many enterprises block UDP
- Always have TCP fallback

### **7. Monitor packet loss**
- High packet loss indicates network issues
- TCP automatically adapts, but monitor performance

---

## Key Takeaways

1. **TCP**: Connection-oriented, reliable, ordered - use for reliability
2. **UDP**: Connectionless, unreliable, fast - use for speed/real-time
3. **TCP handshake**: 3-way (SYN, SYN-ACK, ACK) before data transfer
4. **TCP reliability**: Sequence numbers, ACKs, retransmission, flow control
5. **UDP use cases**: DNS, live video, gaming, VoIP - speed matters
6. **QUIC**: Modern UDP-based protocol combining TCP benefits with UDP speed
7. **Trade-off**: TCP reliability vs UDP speed - choose based on requirements`,
      multipleChoice: [
        {
          id: 'tcp-handshake',
          question:
            'During a TCP three-way handshake, what is the minimum amount of time (in RTTs) before data can start being transmitted?',
          options: [
            '0 RTT (data can be sent with SYN)',
            '0.5 RTT',
            '1 RTT',
            '2 RTT',
          ],
          correctAnswer: 2,
          explanation:
            'The TCP three-way handshake requires 1 full RTT: Client sends SYN, server responds with SYN-ACK (0.5 RTT), client sends ACK and can then send data (1 RTT total). TCP Fast Open can reduce this to 0 RTT for subsequent connections, but standard TCP requires 1 RTT.',
        },
        {
          id: 'udp-use-case',
          question:
            'Which of the following is the BEST reason to use UDP for live video streaming instead of TCP?',
          options: [
            'UDP provides better error correction for video frames',
            'UDP is more secure than TCP for video transmission',
            'Old video frames become useless after their timestamp, making retransmission pointless',
            'UDP automatically compresses video data',
          ],
          correctAnswer: 2,
          explanation:
            "For live streaming, if a video frame is lost, retransmitting it (as TCP would do) is useless because the timestamp has passed. It's better to skip the lost frame and display the next one. UDP doesn't provide error correction, isn't inherently more secure, and doesn't compress data.",
        },
        {
          id: 'tcp-reliability',
          question:
            'How does TCP detect that a packet has been lost and needs retransmission?',
          options: [
            'The receiver sends an error message',
            'TCP uses checksums to detect corrupted packets',
            'Retransmission timer expires OR three duplicate ACKs received',
            'The router notifies the sender via ICMP',
          ],
          correctAnswer: 2,
          explanation:
            "TCP detects packet loss in two ways: 1) Retransmission timer expires without receiving an ACK, or 2) Receiving three duplicate ACKs (fast retransmit). Checksums detect corruption, not loss. Routers don't directly notify TCP about packet loss.",
        },
        {
          id: 'quic-benefit',
          question:
            'What is the primary advantage of QUIC over traditional TCP+TLS?',
          options: [
            'QUIC uses less bandwidth than TCP',
            'QUIC can establish a connection and send encrypted data in 0-1 RTT vs 2-3 RTT for TCP+TLS',
            'QUIC guarantees zero packet loss',
            'QUIC works without encryption',
          ],
          correctAnswer: 1,
          explanation:
            "QUIC's main advantage is connection establishment speed. TCP+TLS requires 2-3 RTT (TCP handshake + TLS handshake), while QUIC combines them into 0-1 RTT. QUIC doesn't use less bandwidth, can't prevent packet loss, and requires encryption by design.",
        },
        {
          id: 'tcp-flow-control',
          question:
            'What mechanism does TCP use to prevent a fast sender from overwhelming a slow receiver?',
          options: [
            'Congestion control',
            'Flow control with sliding window',
            'Automatic packet dropping',
            'Round-robin scheduling',
          ],
          correctAnswer: 1,
          explanation:
            "TCP uses flow control with a sliding window. The receiver advertises its buffer size in each ACK, and the sender won't send more data than the window allows. Congestion control is different - it prevents overwhelming the network (not the receiver).",
        },
      ],
      quiz: [
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
      ],
    },
    {
      id: 'websockets-realtime',
      title: 'WebSockets & Real-Time Communication',
      content: `WebSockets enable real-time, bidirectional communication between client and server. Understanding when and how to use WebSockets is essential for building modern interactive applications.

## The Problem with HTTP for Real-Time

Traditional HTTP is **request-response**: Client initiates, server responds.

**Problem**: How do you push updates from server to client?

### **Attempt 1: Polling**

Client repeatedly asks server for updates:
\`\`\`
Every 5 seconds:
  Client → Server: "Any updates?"
  Server → Client: "No" (or "Yes, here's data")
\`\`\`

**Issues**:
- ❌ Wasteful (mostly "no" responses)
- ❌ Latency (up to 5 seconds delay)
- ❌ High server load (many unnecessary requests)

### **Attempt 2: Long Polling**

Client asks, server holds connection until update available:
\`\`\`
Client → Server: "Any updates?"
[Server holds connection open]
[Update becomes available]
Server → Client: "Yes, here's data"
Client → Server: "Any updates?" (immediately reconnect)
\`\`\`

**Better, but**:
- ❌ Still request-response pattern
- ❌ Connection overhead (HTTP headers every time)
- ❌ Complicated server-side (hold many connections)

### **Attempt 3: Server-Sent Events (SSE)**

Server pushes updates over single HTTP connection:
\`\`\`
Client → Server: "Start streaming updates"
Server → Client: [Data stream, one-way]
\`\`\`

**Better for one-way**, but:
- ❌ One-way only (server → client)
- ❌ Client must use HTTP for sending data
- ❌ Limited browser connection limits (6 per domain)

---

## WebSocket Solution

**WebSocket** is a **full-duplex** protocol over a single TCP connection.

**Full-duplex** = Both parties can send data simultaneously, anytime.

\`\`\`
Client ←→ Server
  Both can send messages anytime
  Single persistent TCP connection
  Low overhead (no HTTP headers per message)
\`\`\`

---

## WebSocket Protocol

### How WebSocket Works

**1. Starts as HTTP (Upgrade Handshake)**

Client initiates with HTTP request:
\`\`\`
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
\`\`\`

Server responds:
\`\`\`
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
\`\`\`

**Status 101**: Switching from HTTP to WebSocket protocol

**2. Connection Upgraded**

After handshake, same TCP connection now used for WebSocket protocol.

**3. Full-Duplex Communication**

\`\`\`
Client → Server: "Hello"
Server → Client: "Hi there"
Client → Server: "How are you?"
Server → Client: "I'm good"
... anytime, either direction ...
\`\`\`

---

## WebSocket Frame Structure

WebSocket sends data in **frames** (not HTTP messages):

\`\`\`
Frame structure:
[FIN|RSV|Opcode|Mask|PayloadLen|MaskingKey|Payload]

FIN: Is this the final fragment?
Opcode: 
  0x1 = Text frame
  0x2 = Binary frame
  0x8 = Connection close
  0x9 = Ping
  0xA = Pong
Mask: Is payload masked? (client→server: yes, server→client: no)
Payload: Actual data
\`\`\`

**Overhead**: 2-14 bytes per frame (vs 100s of bytes for HTTP)

---

## WebSocket vs HTTP Comparison

| Feature | HTTP | WebSocket |
|---------|------|-----------|
| **Direction** | Request-response | Full-duplex |
| **Latency** | High (new request each time) | Low (persistent connection) |
| **Overhead** | High (headers every request) | Low (small frame headers) |
| **Server push** | Not native (need workarounds) | Native support |
| **Connection** | Short-lived | Long-lived |
| **Use case** | Request data on demand | Real-time bidirectional updates |

---

## When to Use WebSockets

Use WebSockets when:
- **Real-time updates** needed
- **Bidirectional** communication
- **Low latency** critical
- **High message frequency**

### Use Cases

**1. Chat Applications**
- Messages sent both directions
- Low latency expected
- High message frequency
- Example: Slack, Discord, WhatsApp Web

**2. Live Notifications**
- Server pushes notifications to client
- Immediate delivery expected
- Example: Facebook notifications, Twitter feed updates

**3. Collaborative Editing**
- Multiple users editing same document
- Changes must propagate immediately
- Example: Google Docs, Figma

**4. Live Sports Scores**
- Server pushes score updates
- Multiple updates per second
- Low latency critical

**5. Trading Platforms**
- Stock prices update in real-time
- Orders submitted immediately
- Example: Robinhood, E*TRADE

**6. Online Gaming**
- Player actions sent both directions
- Sub-100ms latency required
- High message frequency (60+ per second)

**7. IoT Dashboards**
- Sensor data streams to dashboard
- Control commands sent back
- Example: Smart home apps

---

## When NOT to Use WebSockets

**Use HTTP/REST instead** when:
- Occasional data fetching
- One-time requests
- Can tolerate latency
- Need caching (HTTP caches well, WebSocket doesn't)

**Examples**:
- Fetching user profile (occasional)
- Submitting a form (one-time)
- Loading product catalog (can cache)

---

## WebSocket Implementation

### Client-Side (JavaScript)

\`\`\`javascript
// Connect to WebSocket server
const ws = new WebSocket('wss://example.com/socket');

// Connection opened
ws.addEventListener('open', (event) => {
  console.log('Connected to server');
  ws.send('Hello Server!');
});

// Receive message
ws.addEventListener('message', (event) => {
  console.log('Message from server:', event.data);
  
  // Parse JSON if needed
  const data = JSON.parse(event.data);
  console.log(data);
});

// Handle errors
ws.addEventListener('error', (event) => {
  console.error('WebSocket error:', event);
});

// Connection closed
ws.addEventListener('close', (event) => {
  console.log('Disconnected:', event.code, event.reason);
  
  // Reconnect logic
  setTimeout(() => {
    console.log('Reconnecting...');
    // Recreate connection
  }, 5000);
});

// Send message
ws.send('Hello');
ws.send(JSON.stringify({ type: 'chat', message: 'Hello' }));

// Close connection
ws.close();
\`\`\`

### Server-Side (Node.js with ws library)

\`\`\`javascript
const WebSocket = require('ws');

// Create WebSocket server
const wss = new WebSocket.Server({ port: 8080 });

// Track connected clients
const clients = new Set();

wss.on('connection', (ws) => {
  console.log('Client connected');
  clients.add(ws);
  
  // Send welcome message
  ws.send(JSON.stringify({ type: 'welcome', message: 'Connected!' }));
  
  // Receive message from client
  ws.on('message', (message) => {
    console.log('Received:', message);
    
    // Broadcast to all clients
    clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
  });
  
  // Client disconnected
  ws.on('close', () => {
    console.log('Client disconnected');
    clients.delete(ws);
  });
  
  // Handle errors
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
  
  // Heartbeat (detect broken connections)
  ws.isAlive = true;
  ws.on('pong', () => {
    ws.isAlive = true;
  });
});

// Heartbeat interval
setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      return ws.terminate();
    }
    
    ws.isAlive = false;
    ws.ping();
  });
}, 30000);
\`\`\`

---

## Scaling WebSockets

### Challenge: Stateful Connections

Unlike HTTP (stateless), WebSocket connections are **stateful**:
- Server holds connection state
- Can't easily load balance

**Problem**:
\`\`\`
User A connects to Server 1
User B connects to Server 2

User A sends message to User B
→ Server 1 doesn't know User B is on Server 2
\`\`\`

### Solution 1: Sticky Sessions

Route user to same server every time:

\`\`\`
Load Balancer (sticky by user ID)
    ↓                ↓
 Server 1         Server 2
 (Users A, C)     (Users B, D)
\`\`\`

**Pros**:
- Simple
- No coordination needed

**Cons**:
- Uneven load distribution
- Can't survive server restart

### Solution 2: Message Broker (Recommended)

Use pub/sub system to coordinate between servers:

\`\`\`
Server 1 ← User A
  ↓
Redis Pub/Sub
  ↓
Server 2 ← User B
\`\`\`

**Flow**:
1. User A sends message on Server 1
2. Server 1 publishes to Redis channel
3. Server 2 (subscribed to channel) receives message
4. Server 2 sends to User B

**Implementation** (Node.js):
\`\`\`javascript
const Redis = require('ioredis');
const pub = new Redis();
const sub = new Redis();

// Subscribe to channel
sub.subscribe('chat');

// Receive from Redis, send to local clients
sub.on('message', (channel, message) => {
  const data = JSON.parse(message);
  
  // Send to all local WebSocket clients
  clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
});

// Receive from WebSocket client, publish to Redis
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    // Publish to Redis
    pub.publish('chat', message);
  });
});
\`\`\`

**Pros**:
- Scales horizontally
- Servers are stateless (can restart)
- Even load distribution

**Cons**:
- Added complexity
- Redis is single point of failure (use Redis Cluster)

### Solution 3: Dedicated WebSocket Layer

Separate WebSocket servers from application logic:

\`\`\`
Clients
  ↓
WebSocket Servers (handle connections only)
  ↓
Message Queue (Kafka/RabbitMQ)
  ↓
Application Servers (business logic)
  ↓
Database
\`\`\`

**Benefits**:
- Optimize WebSocket servers for connections (10k+ per server)
- Scale application servers independently
- Better resource utilization

---

## Connection Management

### Heartbeat / Ping-Pong

Detect broken connections:

\`\`\`javascript
// Server sends ping every 30 seconds
setInterval(() => {
  ws.ping();
}, 30000);

// If no pong received within timeout, connection is dead
ws.on('pong', () => {
  // Connection still alive
});
\`\`\`

**Why needed**:
- TCP doesn't immediately detect broken connections
- Router/firewall timeouts
- Mobile networks frequently disconnect

### Automatic Reconnection (Client-Side)

\`\`\`javascript
class ReconnectingWebSocket {
  constructor(url) {
    this.url = url;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.connect();
  }
  
  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectDelay = 1000; // Reset delay
    };
    
    this.ws.onclose = () => {
      console.log(\`Disconnected. Reconnecting in \${this.reconnectDelay}ms\`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay);
      
      // Exponential backoff
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        this.maxReconnectDelay
      );
    };
    
    this.ws.onerror = (error) => {
      console.error('Error:', error);
      this.ws.close();
    };
  }
  
  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      console.warn('WebSocket not connected');
    }
  }
}
\`\`\`

### Message Queuing (Client-Side)

Queue messages if connection drops:

\`\`\`javascript
class QueuedWebSocket {
  constructor(url) {
    this.queue = [];
    this.ws = new WebSocket(url);
    
    this.ws.onopen = () => {
      // Flush queue
      while (this.queue.length > 0) {
        this.ws.send(this.queue.shift());
      }
    };
  }
  
  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      // Queue for later
      this.queue.push(data);
    }
  }
}
\`\`\`

---

## Security Considerations

### 1. Authentication

**Don't put credentials in URL**:
\`\`\`javascript
// Bad
new WebSocket('wss://example.com/socket?token=secret123');
\`\`\`

**Use initial handshake**:
\`\`\`javascript
// Good
const ws = new WebSocket('wss://example.com/socket');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: localStorage.getItem('token')
  }));
};

// Server validates token before accepting messages
\`\`\`

### 2. Message Validation

Always validate messages:
\`\`\`javascript
ws.on('message', (message) => {
  try {
    const data = JSON.parse(message);
    
    // Validate structure
    if (!data.type || !data.payload) {
      throw new Error('Invalid message format');
    }
    
    // Validate type
    const allowedTypes = ['chat', 'typing', 'presence'];
    if (!allowedTypes.includes(data.type)) {
      throw new Error('Invalid message type');
    }
    
    // Process message
    handleMessage(data);
  } catch (error) {
    console.error('Invalid message:', error);
    ws.close(1008, 'Invalid message');
  }
});
\`\`\`

### 3. Rate Limiting

Prevent abuse:
\`\`\`javascript
const rateLimits = new Map();

ws.on('message', (message) => {
  const userId = ws.userId;
  const now = Date.now();
  
  if (!rateLimits.has(userId)) {
    rateLimits.set(userId, []);
  }
  
  const timestamps = rateLimits.get(userId);
  
  // Remove old timestamps (older than 1 minute)
  timestamps = timestamps.filter(t => now - t < 60000);
  
  // Check limit (max 100 messages per minute)
  if (timestamps.length >= 100) {
    ws.close(1008, 'Rate limit exceeded');
    return;
  }
  
  timestamps.push(now);
  rateLimits.set(userId, timestamps);
  
  // Process message
  handleMessage(message);
});
\`\`\`

### 4. Use WSS (WebSocket Secure)

Always use \`wss://\` (encrypted) in production, not \`ws://\`.

---

## WebSocket Alternatives

### Server-Sent Events (SSE)

**When to use**:
- One-way communication (server → client only)
- Simple updates
- Built-in auto-reconnect
- Easier to implement

**Example**:
\`\`\`javascript
// Client
const eventSource = new EventSource('/events');

eventSource.onmessage = (event) => {
  console.log('Update:', event.data);
};

// Server (Node.js)
res.writeHead(200, {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive'
});

setInterval(() => {
  res.write(\`data: \${JSON.stringify({ time: Date.now() })}\\n\\n\`);
}, 1000);
\`\`\`

**Pros**:
- Simpler than WebSocket
- Auto-reconnect built-in
- Works over HTTP (easier with proxies)

**Cons**:
- One-way only
- Less efficient than WebSocket
- Browser connection limits (6 per domain)

### gRPC Streaming

**When to use**:
- Microservices communication
- Bidirectional streaming
- Need strong typing (Protocol Buffers)

**Pros**:
- Efficient binary protocol
- Strong typing
- Code generation

**Cons**:
- More complex setup
- Not browser-native (need grpc-web)

---

## Real-World Examples

### Discord

- **35 million concurrent WebSocket connections**
- Uses Erlang for WebSocket gateway servers
- Redis pub/sub for message distribution
- Custom protocol over WebSocket (not plain JSON)

**Architecture**:
\`\`\`
Discord App ← WebSocket → Gateway Servers → Redis → Backend Services
\`\`\`

### Slack

- **10+ million concurrent WebSocket connections**
- WebSocket for real-time messages
- HTTP REST API for message history
- Falls back to long polling if WebSocket unavailable

### Trading Platforms

- **Stock price updates**: WebSocket
- **Order submission**: WebSocket with acknowledgment
- Sub-100ms latency requirement

---

## Best Practices

### 1. Graceful Degradation

Provide fallback if WebSocket unavailable:
\`\`\`javascript
if ('WebSocket' in window) {
  // Use WebSocket
  useWebSocket();
} else {
  // Fall back to long polling
  useLongPolling();
}
\`\`\`

### 2. Message Format

Use structured messages:
\`\`\`javascript
{
  "type": "chat",
  "payload": {
    "message": "Hello",
    "timestamp": 1640000000
  }
}
\`\`\`

### 3. Implement Heartbeat

Both client and server should send periodic pings.

### 4. Automatic Reconnection

Client should auto-reconnect with exponential backoff.

### 5. Connection Limits

Limit connections per user (prevent abuse):
\`\`\`javascript
if (connectionsPerUser.get(userId) >= 5) {
  ws.close(1008, 'Too many connections');
}
\`\`\`

### 6. Monitor Performance

Track:
- Active connections
- Messages per second
- Connection duration
- Reconnection rate
- Error rate

---

## Common Mistakes

### ❌ Not Handling Reconnection

\`\`\`javascript
// Bad: No reconnection logic
const ws = new WebSocket(url);
\`\`\`

Always implement auto-reconnect with exponential backoff.

### ❌ Storing Too Much State Per Connection

\`\`\`javascript
// Bad: Storing entire user profile per connection
ws.userProfile = { /* 100 KB of data */ };
\`\`\`

Keep connection state minimal; fetch data as needed.

### ❌ Not Implementing Heartbeat

Without heartbeat, dead connections accumulate.

### ❌ Assuming Messages Arrive

WebSocket delivers messages, but network issues happen:
\`\`\`javascript
// Add message IDs and acknowledgments
ws.send(JSON.stringify({
  id: uuid(),
  type: 'chat',
  message: 'Hello'
}));
\`\`\`

---

## Interview Tips

### Question: "When would you use WebSockets vs HTTP?"

**Good answer**:
- WebSocket for real-time, bidirectional, high-frequency communication
- Examples: Chat, live notifications, collaborative editing
- HTTP for occasional requests, caching needed, one-time operations
- Trade-off: WebSocket more complex (stateful connections, harder to scale)

### Question: "How do you scale WebSockets?"

**Hit these points**:
1. WebSocket connections are stateful (can't easily load balance)
2. Use sticky sessions OR message broker (Redis pub/sub)
3. Message broker preferred: publish messages, all servers receive
4. Separate WebSocket layer from application logic
5. Monitor connection count, implement rate limiting

### Question: "How do you handle WebSocket connection failures?"

1. Client: Auto-reconnect with exponential backoff
2. Client: Queue messages if disconnected
3. Server: Heartbeat to detect dead connections
4. Server: Clean up resources when connection closes
5. Consider message acknowledgments for critical messages

---

## Key Takeaways

1. **WebSocket** enables full-duplex, real-time communication over single TCP connection
2. **Use cases**: Chat, live notifications, collaborative editing, trading platforms
3. **Scaling**: Use message broker (Redis pub/sub) to coordinate between servers
4. **Connection management**: Implement heartbeat, auto-reconnect, message queuing
5. **Security**: Validate messages, rate limiting, use WSS (encrypted)
6. **Alternatives**: SSE for one-way, gRPC for microservices
7. **Trade-off**: Real-time capability vs complexity of stateful connections
8. **Best practice**: Graceful degradation, structured messages, monitoring`,
      multipleChoice: [
        {
          id: 'websocket-upgrade',
          question:
            'What HTTP status code does a server return during a successful WebSocket upgrade handshake?',
          options: [
            '200 OK',
            '101 Switching Protocols',
            '201 Created',
            '301 Moved Permanently',
          ],
          correctAnswer: 1,
          explanation:
            '101 Switching Protocols is the status code that indicates the server is switching from HTTP to WebSocket protocol. This happens during the upgrade handshake after the client sends an Upgrade: websocket header.',
        },
        {
          id: 'websocket-vs-http',
          question:
            'Which of the following is the PRIMARY advantage of WebSocket over regular HTTP for a chat application?',
          options: [
            'WebSocket is more secure than HTTPS',
            'WebSocket uses less bandwidth by eliminating HTTP headers on every message',
            'WebSocket connections can survive server restarts',
            'WebSocket supports multiple simultaneous requests',
          ],
          correctAnswer: 1,
          explanation:
            "The primary advantage is reduced overhead. After the initial handshake, WebSocket frames have only 2-14 bytes of overhead compared to hundreds of bytes of HTTP headers. WebSocket is not inherently more secure (both can be encrypted), doesn't survive server restarts better, and HTTP/2 also supports multiplexing.",
        },
        {
          id: 'websocket-scaling',
          question:
            'Why is scaling WebSocket servers more challenging than scaling stateless HTTP servers?',
          options: [
            'WebSocket uses more CPU than HTTP',
            'WebSocket connections are stateful and long-lived, making load balancing difficult',
            'WebSocket can only handle 100 concurrent connections per server',
            'WebSocket requires dedicated hardware',
          ],
          correctAnswer: 1,
          explanation:
            'WebSocket connections are stateful (server maintains connection state) and long-lived (can last hours/days). This makes traditional round-robin load balancing ineffective - you need sticky sessions or a message broker to coordinate between servers. HTTP is stateless, so any server can handle any request.',
        },
        {
          id: 'websocket-heartbeat',
          question:
            'What is the purpose of implementing a heartbeat/ping-pong mechanism in WebSocket connections?',
          options: [
            'To encrypt the messages between client and server',
            "To detect broken connections that TCP doesn't immediately report",
            'To compress data before sending it',
            'To authenticate the user every few seconds',
          ],
          correctAnswer: 1,
          explanation:
            "Heartbeat (ping-pong) detects broken connections that TCP doesn't immediately report, such as when a router disconnects or a mobile device loses signal. Without heartbeat, the server might hold resources for dead connections. It doesn't handle encryption, compression, or authentication.",
        },
        {
          id: 'websocket-vs-sse',
          question:
            'When would Server-Sent Events (SSE) be a better choice than WebSocket?',
          options: [
            'When you need bidirectional communication',
            'When you need to send binary data',
            'When you only need server-to-client updates and want simpler implementation',
            'When you need to scale to millions of connections',
          ],
          correctAnswer: 2,
          explanation:
            'SSE is simpler than WebSocket and has built-in auto-reconnect, making it perfect for one-way server-to-client updates like live scores or notifications. For bidirectional communication, binary data, or complex scaling, WebSocket is better. Both can scale to millions of connections.',
        },
      ],
      quiz: [
        {
          id: 'chat-websocket-design',
          question:
            "Design a real-time chat system like Slack that needs to handle 10 million concurrent users. Explain your WebSocket architecture, how you'd scale it, how you'd handle message delivery guarantees, and what happens when a user is offline. Discuss trade-offs in your design.",
          sampleAnswer: `**System Requirements**:
- 10 million concurrent users
- Real-time message delivery (<100ms latency)
- Message history/persistence
- Offline message delivery
- Typing indicators, read receipts
- File sharing

**Architecture Overview**:

\`\`\`
Clients (10M)
    ↓
[CDN] → Static assets
    ↓
[Load Balancer] → WebSocket Gateway Cluster (1000 servers)
    ↓
[Message Queue (Kafka)] → Message Broker
    ↓
[Application Services Cluster]
    ↓
[Database (Cassandra)] → Message Storage
[Cache (Redis)] → Online users, presence
[Object Storage (S3)] → Files
\`\`\`

**Component Design**:

**1. WebSocket Gateway Layer**

*Purpose*: Handle WebSocket connections only, no business logic

*Scaling*:
- Each server handles 10,000 concurrent connections
- 10M users ÷ 10k per server = 1,000 servers needed
- Optimize: Use Erlang/Elixir (Discord's approach) or Go for high concurrency

*Implementation*:
\`\`\`javascript
class WebSocketGateway {
  constructor() {
    this.connections = new Map(); // userId → ws connection
    this.kafka = new KafkaProducer();
    this.redis = new Redis();
  }
  
  async handleConnection(ws, userId) {
    // Store connection
    this.connections.set(userId, ws);
    
    // Update presence in Redis
    await this.redis.sadd('online_users', userId);
    
    // Publish presence update
    this.kafka.publish('presence', {
      type: 'online',
      userId
    });
    
    ws.on('message', async (msg) => {
      await this.handleMessage(userId, msg);
    });
    
    ws.on('close', async () => {
      this.connections.delete(userId);
      await this.redis.srem('online_users', userId);
      
      this.kafka.publish('presence', {
        type: 'offline',
        userId
      });
    });
  }
  
  async handleMessage(userId, message) {
    // Parse message
    const data = JSON.parse(message);
    
    // Add metadata
    data.senderId = userId;
    data.timestamp = Date.now();
    data.messageId = uuid();
    
    // Publish to Kafka for processing
    await this.kafka.publish('messages', data);
  }
  
  async sendToUser(userId, message) {
    const ws = this.connections.get(userId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
      return true;
    }
    return false;
  }
}
\`\`\`

**2. Message Queue (Kafka)**

*Why Kafka*:
- High throughput (millions of messages/sec)
- Durability (messages persisted)
- Replay capability (for offline users)
- Multiple consumers can process independently

*Topics*:
- \`messages\`: Chat messages
- \`presence\`: Online/offline updates
- \`typing\`: Typing indicators
- \`read-receipts\`: Message read status

**3. Message Processing Service**

*Responsibilities*:
- Validate messages
- Spam detection
- Store in database
- Fan out to recipients
- Handle offline delivery

\`\`\`javascript
class MessageProcessor {
  async processMessage(msg) {
    // 1. Validate and sanitize
    if (!this.isValid(msg)) {
      return;
    }
    
    // 2. Persist to database
    await this.db.insert('messages', {
      id: msg.messageId,
      channelId: msg.channelId,
      senderId: msg.senderId,
      content: msg.content,
      timestamp: msg.timestamp
    });
    
    // 3. Get channel members
    const members = await this.getChannelMembers(msg.channelId);
    
    // 4. Fan out to online members
    const deliveryStatus = await Promise.all(
      members.map(userId => this.deliverToUser(userId, msg))
    );
    
    // 5. Store for offline members
    const offlineUsers = members.filter((_, i) => !deliveryStatus[i]);
    if (offlineUsers.length > 0) {
      await this.storeOfflineMessages(offlineUsers, msg);
    }
    
    // 6. Send acknowledgment to sender
    await this.sendAck(msg.senderId, msg.messageId);
  }
  
  async deliverToUser(userId, msg) {
    // Check if user is online
    const isOnline = await this.redis.sismember('online_users', userId);
    if (!isOnline) {
      return false;
    }
    
    // Find which gateway server has the user
    const serverId = await this.redis.hget('user_connections', userId);
    
    // Publish to that server's topic
    await this.kafka.publish(\`gateway-\${serverId}\`, {
      type: 'deliver',
      userId,
      message: msg
    });
    
    return true;
  }
}
\`\`\`

**4. Offline Message Handling**

*Strategy 1: Push notifications*
- Send push notification to mobile device
- User opens app, connects, receives queued messages

*Strategy 2: Message inbox*
\`\`\`javascript
// When user comes online
async function onUserConnect(userId) {
  // Fetch undelivered messages
  const messages = await db.query(
    'SELECT * FROM offline_messages WHERE userId = ? ORDER BY timestamp',
    [userId]
  );
  
  // Deliver all messages
  for (const msg of messages) {
    await sendToUser(userId, msg);
  }
  
  // Clear offline queue
  await db.delete('offline_messages', { userId });
}
\`\`\`

*Storage*:
- DynamoDB: \`{ userId, messageId, message, timestamp }\`
- TTL: Auto-delete after 30 days
- Query pattern: \`userId\` as partition key, \`timestamp\` as sort key

**5. Message Delivery Guarantees**

*At-least-once delivery*:

\`\`\`javascript
// Client-side
class ReliableWebSocket {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.pendingMessages = new Map(); // messageId → message
    this.ackTimeout = 5000; // 5 seconds
  }
  
  send(message) {
    const messageId = uuid();
    message.id = messageId;
    
    this.ws.send(JSON.stringify(message));
    
    // Set timeout for acknowledgment
    const timer = setTimeout(() => {
      console.log('No ack received, resending');
      this.send(message);
    }, this.ackTimeout);
    
    this.pendingMessages.set(messageId, {
      message,
      timer
    });
  }
  
  handleAck(ack) {
    const pending = this.pendingMessages.get(ack.messageId);
    if (pending) {
      clearTimeout(pending.timer);
      this.pendingMessages.delete(ack.messageId);
    }
  }
}
\`\`\`

*Idempotency*:
\`\`\`javascript
// Server-side: Deduplicate messages
const processedMessages = new Set();

async function handleMessage(msg) {
  if (processedMessages.has(msg.messageId)) {
    // Already processed, send ack again
    sendAck(msg.senderId, msg.messageId);
    return;
  }
  
  // Process message
  await storeMessage(msg);
  await deliverMessage(msg);
  
  // Mark as processed
  processedMessages.add(msg.messageId);
  
  // Send ack
  sendAck(msg.senderId, msg.messageId);
}
\`\`\`

**6. Presence and Typing Indicators**

*Presence*:
\`\`\`javascript
// Redis sorted set: score = last seen timestamp
await redis.zadd('presence', Date.now(), userId);

// Get users active in last 5 minutes
const activeUsers = await redis.zrangebyscore(
  'presence',
  Date.now() - 300000,
  Date.now()
);
\`\`\`

*Typing indicators*:
- Don't persist (ephemeral)
- Publish to Kafka topic with 2-second TTL
- Only deliver to online channel members
- Throttle: Max 1 update per second per user

**7. File Sharing**

*Flow*:
1. Client uploads file to S3 (direct upload with presigned URL)
2. Client sends message with file URL
3. Server validates file (virus scan)
4. Deliver message with file link

\`\`\`javascript
// Get presigned URL for upload
app.post('/api/upload-url', async (req, res) => {
  const { filename, contentType } = req.body;
  
  const key = \`files/\${uuid()}/\${filename}\`;
  const url = await s3.getSignedUrl('putObject', {
    Bucket: 'chat-files',
    Key: key,
    ContentType: contentType,
    Expires: 300 // 5 minutes
  });
  
  res.json({ uploadUrl: url, fileKey: key });
});

// Client uploads directly to S3
// Then sends message with file reference
\`\`\`

**Scaling Considerations**:

*Horizontal Scaling*:
- WebSocket gateways: Stateless (except connections), easy to scale
- Add more servers behind load balancer
- Sticky sessions based on userId hash

*Geographic Distribution*:
- Deploy in multiple regions (US-East, US-West, EU, Asia)
- Users connect to nearest region
- Cross-region message delivery via global Kafka cluster

*Connection Distribution*:
\`\`\`
Load Balancer strategy: Least connections
- Route new connection to server with fewest connections
- More even distribution than round-robin
\`\`\`

**Database Design**:

*Cassandra schema*:
\`\`\`sql
-- Messages table
CREATE TABLE messages (
  channel_id UUID,
  timestamp TIMEUUID,
  message_id UUID,
  sender_id UUID,
  content TEXT,
  PRIMARY KEY (channel_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Query pattern: Get recent messages in channel
SELECT * FROM messages 
WHERE channel_id = ? 
ORDER BY timestamp DESC 
LIMIT 50;
\`\`\`

*Why Cassandra*:
- Writes scale linearly (important for high message volume)
- Partition by channel_id (good distribution)
- Time-series data (messages naturally ordered by time)
- High availability (no single point of failure)

**Trade-offs**:

**1. Message Delivery: At-least-once vs Exactly-once**

*Chose at-least-once*:
- Pro: Simpler, more reliable
- Con: Possible duplicates (handle with idempotency)
- Exactly-once is very complex in distributed systems

**2. Gateway Layer: Thick vs Thin**

*Chose thin*:
- Gateway only handles connections, no business logic
- Pro: Scales independently, simpler
- Con: Extra network hop through Kafka

**3. Presence: Real-time vs Eventually consistent**

*Chose eventually consistent*:
- Update presence every 30 seconds (heartbeat)
- Pro: Reduces load, good enough for chat
- Con: User might show online for 30 seconds after disconnect

**4. Database: SQL vs NoSQL**

*Chose Cassandra (NoSQL)*:
- Pro: Better write scalability, natural fit for time-series
- Con: Limited query flexibility (can't do full-text search)
- Mitigation: Use Elasticsearch for search

**Monitoring**:

*Key Metrics*:
- Active WebSocket connections per server
- Messages per second
- Message delivery latency (p50, p99)
- Connection duration
- Reconnection rate
- Offline message queue size

*Alerts*:
- Connection count > 9,000 per server (approaching limit)
- Message latency > 200ms
- Reconnection rate > 5%

**Expected Performance**:
- 10M concurrent users on 1000 servers (10k per server)
- <100ms message delivery latency (p99)
- 99.9% message delivery success
- Cost: ~$50k/month infrastructure (AWS/GCP)`,
          keyPoints: [
            'Separate WebSocket gateway layer from application logic for independent scaling',
            'Use Kafka for message distribution, durability, and offline message replay',
            'Implement at-least-once delivery with client acknowledgments and idempotency',
            'Store offline messages in DynamoDB/similar with userId as partition key',
            'Use Redis for presence tracking and online user lookup',
            'Scale to 10M users with ~1000 gateway servers (10k connections each)',
            'Cassandra for message storage (good for time-series, write-heavy workload)',
            'Geographic distribution: Deploy in multiple regions, route to nearest',
          ],
        },
        {
          id: 'websocket-fallback',
          question:
            'Your application uses WebSockets for real-time updates, but you discover that 15% of your users are behind corporate firewalls that block WebSocket connections. Design a fallback strategy that maintains functionality for these users while minimizing code complexity. Discuss the trade-offs.',
          sampleAnswer: `**Problem Analysis**:

Corporate firewalls often block WebSocket because:
- Uses \`Upgrade\` header (suspicious to some firewalls)
- Long-lived connections (timeouts)
- UDP-like behavior over TCP (some DPI systems block)
- Non-standard port (if not 80/443)

**Fallback Strategy Design**:

**Tier 1: WebSocket (85% of users)**
- Best performance, lowest latency
- Full duplex communication
- Try this first

**Tier 2: HTTP/2 Server-Sent Events (SSE) (12% of users)**
- One-way server → client
- Client uses regular HTTP POST for client → server
- Works through most firewalls
- Built-in auto-reconnect

**Tier 3: Long Polling (3% of users)**
- Works everywhere (just HTTP)
- Highest latency, most overhead
- Last resort

**Implementation**:

**1. Connection Detection & Fallback**

\`\`\`javascript
class AdaptiveTransport {
  constructor(url) {
    this.url = url;
    this.transport = null;
    this.connect();
  }
  
  async connect() {
    // Try WebSocket first
    try {
      this.transport = await this.tryWebSocket();
      console.log('Using WebSocket');
      return;
    } catch (error) {
      console.log('WebSocket failed:', error);
    }
    
    // Try SSE
    try {
      this.transport = await this.trySSE();
      console.log('Using SSE');
      return;
    } catch (error) {
      console.log('SSE failed:', error);
    }
    
    // Fall back to long polling
    this.transport = this.useLongPolling();
    console.log('Using long polling');
  }
  
  tryWebSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connection timeout'));
      }, 5000);
      
      ws.onopen = () => {
        clearTimeout(timeout);
        resolve(new WebSocketTransport(ws));
      };
      
      ws.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
    });
  }
  
  trySSE() {
    return new Promise((resolve, reject) => {
      const sse = new EventSource(\`\${this.url}/sse\`);
      
      const timeout = setTimeout(() => {
        sse.close();
        reject(new Error('SSE connection timeout'));
      }, 5000);
      
      sse.onopen = () => {
        clearTimeout(timeout);
        resolve(new SSETransport(sse, this.url));
      };
      
      sse.onerror = (error) => {
        clearTimeout(timeout);
        sse.close();
        reject(error);
      };
    });
  }
  
  useLongPolling() {
    return new LongPollingTransport(this.url);
  }
  
  send(data) {
    this.transport.send(data);
  }
  
  onMessage(callback) {
    this.transport.onMessage(callback);
  }
}
\`\`\`

**2. WebSocket Transport (Primary)**

\`\`\`javascript
class WebSocketTransport {
  constructor(ws) {
    this.ws = ws;
    this.messageHandlers = [];
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    };
  }
  
  send(data) {
    this.ws.send(JSON.stringify(data));
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.ws.close();
  }
}
\`\`\`

**3. SSE Transport (Fallback 1)**

\`\`\`javascript
class SSETransport {
  constructor(sse, baseUrl) {
    this.sse = sse;
    this.baseUrl = baseUrl;
    this.messageHandlers = [];
    
    this.sse.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.messageHandlers.forEach(handler => handler(data));
    };
  }
  
  async send(data) {
    // Use regular HTTP POST for client → server
    await fetch(\`\${this.baseUrl}/message\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.sse.close();
  }
}
\`\`\`

**Server-side SSE** (Node.js):
\`\`\`javascript
app.get('/sse', (req, res) => {
  // Set SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no' // Disable nginx buffering
  });
  
  // Send initial connection message
  res.write('data: {"type":"connected"}\\n\\n');
  
  // Register client for push updates
  const clientId = uuid();
  clients.set(clientId, res);
  
  // Heartbeat (keep connection alive)
  const heartbeat = setInterval(() => {
    res.write(': heartbeat\\n\\n');
  }, 30000);
  
  // Cleanup on disconnect
  req.on('close', () => {
    clearInterval(heartbeat);
    clients.delete(clientId);
  });
});

// Send message to all SSE clients
function broadcastSSE(message) {
  const data = \`data: \${JSON.stringify(message)}\\n\\n\`;
  clients.forEach(client => {
    client.write(data);
  });
}
\`\`\`

**4. Long Polling Transport (Fallback 2)**

\`\`\`javascript
class LongPollingTransport {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.messageHandlers = [];
    this.polling = true;
    this.startPolling();
  }
  
  async startPolling() {
    while (this.polling) {
      try {
        const response = await fetch(\`\${this.baseUrl}/poll\`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' }
        });
        
        if (response.ok) {
          const messages = await response.json();
          messages.forEach(msg => {
            this.messageHandlers.forEach(handler => handler(msg));
          });
        }
      } catch (error) {
        console.error('Polling error:', error);
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  }
  
  async send(data) {
    await fetch(\`\${this.baseUrl}/message\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  }
  
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }
  
  close() {
    this.polling = false;
  }
}
\`\`\`

**Server-side long polling**:
\`\`\`javascript
const messageQueues = new Map(); // userId → queue of messages

app.get('/poll', async (req, res) => {
  const userId = req.user.id;
  
  // Check if messages available
  const queue = messageQueues.get(userId) || [];
  
  if (queue.length > 0) {
    // Send immediately
    res.json(queue);
    messageQueues.set(userId, []);
  } else {
    // Hold connection until message arrives (max 30 seconds)
    const timeout = setTimeout(() => {
      res.json([]);
    }, 30000);
    
    // Register callback for when message arrives
    pendingRequests.set(userId, {
      res,
      timeout
    });
  }
});

// When message arrives
function deliverMessage(userId, message) {
  // Check if user has pending long poll request
  const pending = pendingRequests.get(userId);
  
  if (pending) {
    // Deliver immediately
    clearTimeout(pending.timeout);
    pending.res.json([message]);
    pendingRequests.delete(userId);
  } else {
    // Queue for next poll
    const queue = messageQueues.get(userId) || [];
    queue.push(message);
    messageQueues.set(userId, queue);
  }
}
\`\`\`

**5. Unified Application Interface**

\`\`\`javascript
// Application code doesn't know which transport is used
const transport = new AdaptiveTransport('wss://example.com/socket');

// Send messages (works with all transports)
transport.send({
  type: 'chat',
  message: 'Hello'
});

// Receive messages (works with all transports)
transport.onMessage((message) => {
  console.log('Received:', message);
});
\`\`\`

**Trade-offs Analysis**:

**1. Code Complexity**

*Increased*:
- Three transport implementations
- Connection detection logic
- Server must support all three protocols

*Mitigated by*:
- Shared interface (AdaptiveTransport)
- Transport-specific logic encapsulated
- Use libraries (socket.io handles this automatically)

**2. Server Resources**

*WebSocket*:
- 1 connection per user
- Minimal overhead after handshake

*SSE*:
- 1 connection per user (server → client)
- + HTTP POST for client → server
- Slightly more overhead

*Long Polling*:
- 1 active HTTP request per user constantly
- Much higher overhead
- More server resources (worker threads/processes)

*Cost*:
- 100k WebSocket users: 100 servers
- 100k long polling users: 300 servers
- 15% long polling = +30% server cost

**3. Latency**

*WebSocket*: 10-50ms
*SSE*: 20-100ms (includes HTTP POST roundtrip for sending)
*Long Polling*: 100-500ms (worst case: timeout + new request)

*Impact*:
- Chat feels slightly less real-time for long polling users
- Acceptable trade-off vs no functionality

**4. Battery Usage (Mobile)**

*WebSocket*: Best (single persistent connection)
*SSE*: Good (one persistent connection + occasional POST)
*Long Polling*: Poor (constant HTTP requests)

*Mitigation*:
- Increase polling interval on mobile (5-10 seconds)
- Use adaptive polling (poll faster during active use)

**5. Firewall Compatibility**

| Transport | Corporate Firewall | Hotel WiFi | Mobile Network | China Great Firewall |
|-----------|-------------------|------------|----------------|---------------------|
| WebSocket | 85% | 95% | 98% | 60% |
| SSE | 95% | 99% | 99% | 80% |
| Long Polling | 100% | 100% | 100% | 95% |

**Advanced Optimization: Adaptive Polling**

For long polling, don't poll constantly:

\`\`\`javascript
class AdaptiveLongPolling {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.pollInterval = 30000; // Start with 30 seconds
    this.active = false;
  }
  
  async poll() {
    const messages = await fetch(\`\${this.baseUrl}/poll\`);
    
    if (messages.length > 0) {
      // Activity detected, poll faster
      this.pollInterval = 1000; // 1 second
      this.deliverMessages(messages);
    } else {
      // No activity, slow down (exponential backoff)
      this.pollInterval = Math.min(this.pollInterval * 1.5, 30000);
    }
    
    setTimeout(() => this.poll(), this.pollInterval);
  }
  
  onUserInteraction() {
    // User is typing, poll faster
    this.pollInterval = 1000;
  }
}
\`\`\`

**Monitoring & Analytics**:

Track transport usage:
\`\`\`javascript
analytics.track('transport_type', {
  type: 'websocket', // or 'sse' or 'long_polling'
  userAgent: navigator.userAgent,
  region: userRegion
});
\`\`\`

Use this data to:
- Understand firewall blocking patterns
- Optimize server resource allocation
- Identify problematic networks/regions

**Recommendation: Use socket.io**

Socket.io library handles all this automatically:
\`\`\`javascript
// Client
const socket = io('https://example.com');
// Automatically tries: WebSocket → long polling

// Server
const io = require('socket.io')(server);
io.on('connection', (socket) => {
  // Same code for all transports
});
\`\`\`

**Final Architecture**:

\`\`\`
Client
  ↓
[Auto-detect best transport]
  ↓
├─ WebSocket (85%)
├─ SSE + HTTP POST (12%)
└─ Long Polling (3%)
  ↓
Server (handles all three)
  ↓
[Application logic (transport-agnostic)]
\`\`\`

**Expected Results**:
- 100% of users can use the application (vs 85% with WebSocket only)
- +20-30% server cost (for fallback support)
- Slightly higher latency for 15% of users (acceptable)
- Minimal code complexity increase (if using libraries)

This provides universal compatibility while maintaining good performance for the majority.`,
          keyPoints: [
            'Implement tiered fallback: WebSocket → SSE → Long Polling',
            'Auto-detect best available transport on connection',
            'SSE works for most firewalls, provides one-way server → client with HTTP POST for reverse',
            'Long Polling is last resort, works everywhere but higher latency and server cost',
            'Use unified interface so application code is transport-agnostic',
            'Monitor transport usage to understand blocking patterns',
            'Consider socket.io library which handles fallback automatically',
            'Trade-off: +20-30% server cost for 100% compatibility vs 85% with WebSocket only',
          ],
        },
        {
          id: 'websocket-security',
          question:
            'Explain the security vulnerabilities specific to WebSocket connections and how you would mitigate them. Include authentication, authorization, rate limiting, and protection against common attacks. How would you handle connection hijacking?',
          sampleAnswer: `**WebSocket-Specific Security Challenges**:

Unlike HTTP, WebSocket:
- Long-lived connections (more time for attacks)
- Bidirectional (attack surface in both directions)
- Bypasses some traditional security tools (HTTP-only firewalls, proxies)
- Less mature security ecosystem than HTTP

**1. Authentication Vulnerabilities**

**Problem 1: Credentials in URL**

\`\`\`javascript
// VULNERABLE
const ws = new WebSocket('wss://example.com/socket?token=secret123');
\`\`\`

*Why bad*:
- Tokens logged in server logs
- Visible in proxy logs
- Sent in HTTP headers during upgrade (cleartext if not WSS)
- Could be leaked in referrer headers

**Solution: Authenticate after connection**

\`\`\`javascript
// Client
const ws = new WebSocket('wss://example.com/socket');

ws.onopen = () => {
  // Send auth token as first message
  ws.send(JSON.stringify({
    type: 'auth',
    token: localStorage.getItem('authToken')
  }));
};

// Server
ws.on('message', async (message) => {
  const data = JSON.parse(message);
  
  if (!ws.authenticated) {
    if (data.type !== 'auth') {
      ws.close(4001, 'Authentication required');
      return;
    }
    
    // Validate token
    try {
      const user = await validateToken(data.token);
      ws.authenticated = true;
      ws.userId = user.id;
      ws.send(JSON.stringify({ type: 'auth_success' }));
    } catch (error) {
      ws.close(4001, 'Invalid token');
    }
    return;
  }
  
  // Handle other messages
  handleMessage(ws, data);
});
\`\`\`

**Problem 2: Session Fixation**

*Attack*: Attacker creates WebSocket connection, gets session ID, tricks victim into using it.

**Solution: Bind session to user**
\`\`\`javascript
// Generate new session ID after authentication
ws.on('auth_success', () => {
  const oldSessionId = ws.sessionId;
  ws.sessionId = generateNewSessionId();
  
  // Invalidate old session
  sessions.delete(oldSessionId);
  sessions.set(ws.sessionId, { userId: ws.userId });
});
\`\`\`

**Problem 3: Token Expiration**

Long-lived WebSocket connections can outlive token expiration.

**Solution: Periodic re-authentication**
\`\`\`javascript
// Client: Periodically refresh token
setInterval(async () => {
  const newToken = await refreshAuthToken();
  ws.send(JSON.stringify({
    type: 'refresh_auth',
    token: newToken
  }));
}, 15 * 60 * 1000); // Every 15 minutes

// Server: Check token expiry
setInterval(() => {
  if (tokenExpired(ws.token)) {
    ws.close(4001, 'Token expired');
  }
}, 60 * 1000); // Check every minute
\`\`\`

**2. Authorization Vulnerabilities**

**Problem: Insufficient authorization checks**

\`\`\`javascript
// VULNERABLE
ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // No authorization check!
  if (data.type === 'read_chat') {
    const messages = getMessages(data.channelId);
    ws.send(JSON.stringify(messages));
  }
});
\`\`\`

**Solution: Authorize every action**
\`\`\`javascript
ws.on('message', async (message) => {
  const data = JSON.parse(message);
  
  // Always check authorization
  if (data.type === 'read_chat') {
    // Verify user has access to this channel
    const hasAccess = await checkChannelAccess(ws.userId, data.channelId);
    if (!hasAccess) {
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Access denied'
      }));
      return;
    }
    
    const messages = getMessages(data.channelId);
    ws.send(JSON.stringify(messages));
  }
});
\`\`\`

**Best Practice: Authorization middleware**
\`\`\`javascript
async function requireChannelAccess(ws, channelId) {
  const hasAccess = await checkChannelAccess(ws.userId, channelId);
  if (!hasAccess) {
    throw new Error('Access denied');
  }
}

// Use in handlers
ws.on('message', async (message) => {
  try {
    const data = JSON.parse(message);
    
    if (data.type === 'read_chat') {
      await requireChannelAccess(ws, data.channelId);
      const messages = getMessages(data.channelId);
      ws.send(JSON.stringify(messages));
    }
  } catch (error) {
    ws.send(JSON.stringify({ type: 'error', message: error.message }));
  }
});
\`\`\`

**3. Rate Limiting**

**Problem: DoS via message flooding**

\`\`\`javascript
// Attacker sends 10,000 messages per second
for (let i = 0; i < 10000; i++) {
  ws.send('spam');
}
\`\`\`

**Solution 1: Per-connection rate limiting**
\`\`\`javascript
class RateLimiter {
  constructor(maxMessages, windowMs) {
    this.maxMessages = maxMessages; // e.g., 100
    this.windowMs = windowMs; // e.g., 1000ms
    this.counters = new Map(); // connectionId → [timestamps]
  }
  
  check(connectionId) {
    const now = Date.now();
    
    if (!this.counters.has(connectionId)) {
      this.counters.set(connectionId, []);
    }
    
    const timestamps = this.counters.get(connectionId);
    
    // Remove old timestamps
    const recent = timestamps.filter(t => now - t < this.windowMs);
    
    if (recent.length >= this.maxMessages) {
      return false; // Rate limit exceeded
    }
    
    recent.push(now);
    this.counters.set(connectionId, recent);
    
    return true;
  }
}

const rateLimiter = new RateLimiter(100, 1000); // 100 messages per second

ws.on('message', (message) => {
  if (!rateLimiter.check(ws.id)) {
    ws.send(JSON.stringify({ type: 'error', message: 'Rate limit exceeded' }));
    // Optionally close connection after repeated violations
    ws.violationCount = (ws.violationCount || 0) + 1;
    if (ws.violationCount > 5) {
      ws.close(4008, 'Rate limit exceeded');
    }
    return;
  }
  
  handleMessage(message);
});
\`\`\`

**Solution 2: Per-user rate limiting (across all connections)**
\`\`\`javascript
// Use Redis for distributed rate limiting
async function checkRateLimit(userId) {
  const key = \`ratelimit:\${userId}\`;
  const count = await redis.incr(key);
  
  if (count === 1) {
    // Set expiry on first request
    await redis.expire(key, 1); // 1 second window
  }
  
  return count <= 100; // Max 100 messages per second
}

ws.on('message', async (message) => {
  const allowed = await checkRateLimit(ws.userId);
  if (!allowed) {
    ws.send(JSON.stringify({ type: 'error', message: 'Rate limit exceeded' }));
    return;
  }
  
  handleMessage(message);
});
\`\`\`

**4. Cross-Site WebSocket Hijacking (CSWSH)**

**Attack**: Malicious site opens WebSocket to your server using victim's cookies.

\`\`\`html
<!-- evil.com -->
<script>
  // Opens WebSocket to victim.com
  // Browser automatically sends cookies!
  const ws = new WebSocket('wss://victim.com/socket');
  ws.onmessage = (e) => {
    // Steal victim's messages
    sendToAttacker(e.data);
  };
</script>
\`\`\`

**Solution: Check Origin header**
\`\`\`javascript
wss.on('connection', (ws, req) => {
  const origin = req.headers.origin;
  
  // Whitelist of allowed origins
  const allowedOrigins = [
    'https://example.com',
    'https://app.example.com'
  ];
  
  if (!allowedOrigins.includes(origin)) {
    ws.close(4003, 'Origin not allowed');
    return;
  }
  
  // Continue with connection
});
\`\`\`

**Additional protection: Custom header**
\`\`\`javascript
// Client: Add custom header during upgrade
const ws = new WebSocket('wss://example.com/socket', {
  headers: {
    'X-Custom-Auth': 'your-secret-token'
  }
});

// Server: Verify custom header
wss.on('connection', (ws, req) => {
  const customAuth = req.headers['x-custom-auth'];
  if (customAuth !== expectedToken) {
    ws.close(4003, 'Invalid auth header');
    return;
  }
});
\`\`\`

**5. Message Injection**

**Problem: Unsanitized user input**

\`\`\`javascript
// VULNERABLE
ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // Send to all users without sanitization
  broadcast({
    type: 'chat',
    message: data.message, // Could contain XSS payload
    user: data.user
  });
});
\`\`\`

**Solution: Validate and sanitize**
\`\`\`javascript
const sanitizeHtml = require('sanitize-html');

ws.on('message', (message) => {
  const data = JSON.parse(message);
  
  // Validate structure
  if (!data.type || !data.message) {
    ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format' }));
    return;
  }
  
  // Validate message type
  const allowedTypes = ['chat', 'typing', 'read'];
  if (!allowedTypes.includes(data.type)) {
    ws.send(JSON.stringify({ type: 'error', message: 'Invalid message type' }));
    return;
  }
  
  // Sanitize HTML
  const cleanMessage = sanitizeHtml(data.message, {
    allowedTags: ['b', 'i', 'em', 'strong'],
    allowedAttributes: {}
  });
  
  // Validate length
  if (cleanMessage.length > 1000) {
    ws.send(JSON.stringify({ type: 'error', message: 'Message too long' }));
    return;
  }
  
  broadcast({
    type: 'chat',
    message: cleanMessage,
    user: ws.userId, // Use server-side userId, not client-provided
    timestamp: Date.now() // Server-generated timestamp
  });
});
\`\`\`

**6. Connection Hijacking**

**Attack Scenario**: Attacker intercepts WebSocket connection and sends commands as victim.

**Mitigation 1: Use WSS (WebSocket Secure)**
\`\`\`javascript
// Always use wss:// (not ws://)
const ws = new WebSocket('wss://example.com/socket');
\`\`\`

*Why*:
- Encrypts all traffic (like HTTPS)
- Prevents man-in-the-middle attacks
- Certificate validation

**Mitigation 2: Message signatures**
\`\`\`javascript
// Client: Sign each message
function sendSecure(ws, data) {
  const message = JSON.stringify(data);
  const signature = hmacSHA256(message, userSecret);
  
  ws.send(JSON.stringify({
    message,
    signature
  }));
}

// Server: Verify signature
ws.on('message', (payload) => {
  const { message, signature } = JSON.parse(payload);
  
  const userSecret = getUserSecret(ws.userId);
  const expectedSignature = hmacSHA256(message, userSecret);
  
  if (signature !== expectedSignature) {
    ws.close(4004, 'Invalid signature');
    return;
  }
  
  handleMessage(JSON.parse(message));
});
\`\`\`

**Mitigation 3: Mutual TLS (mTLS)**

For high-security applications (B2B, financial):
\`\`\`javascript
const https = require('https');
const fs = require('fs');

const server = https.createServer({
  cert: fs.readFileSync('server-cert.pem'),
  key: fs.readFileSync('server-key.pem'),
  ca: fs.readFileSync('ca-cert.pem'),
  requestCert: true,
  rejectUnauthorized: true
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  // Client certificate validated by TLS
  const clientCert = req.socket.getPeerCertificate();
  console.log('Client:', clientCert.subject.CN);
});
\`\`\`

**7. Denial of Service (DoS)**

**Attack 1: Connection exhaustion**
\`\`\`javascript
// Attacker opens 10,000 connections
for (let i = 0; i < 10000; i++) {
  new WebSocket('wss://example.com/socket');
}
\`\`\`

**Solution: Connection limits**
\`\`\`javascript
const connectionsPerIP = new Map();

wss.on('connection', (ws, req) => {
  const ip = req.socket.remoteAddress;
  
  const count = connectionsPerIP.get(ip) || 0;
  
  // Max 10 connections per IP
  if (count >= 10) {
    ws.close(4009, 'Too many connections from this IP');
    return;
  }
  
  connectionsPerIP.set(ip, count + 1);
  
  ws.on('close', () => {
    connectionsPerIP.set(ip, connectionsPerIP.get(ip) - 1);
  });
});
\`\`\`

**Attack 2: Large message DoS**
\`\`\`javascript
// Send 100MB message
ws.send('x'.repeat(100 * 1024 * 1024));
\`\`\`

**Solution: Message size limit**
\`\`\`javascript
wss.on('connection', (ws) => {
  ws.on('message', (message) => {
    // Limit message size to 1MB
    if (message.length > 1024 * 1024) {
      ws.close(4010, 'Message too large');
      return;
    }
    
    handleMessage(message);
  });
});
\`\`\`

**8. Monitoring & Alerting**

\`\`\`javascript
// Track suspicious activity
class SecurityMonitor {
  constructor() {
    this.violations = new Map(); // userId → violation count
  }
  
  recordViolation(userId, type) {
    const key = \`\${userId}:\${type}\`;
    const count = this.violations.get(key) || 0;
    this.violations.set(key, count + 1);
    
    // Alert if threshold exceeded
    if (count > 10) {
      alertSecurity(\`User \${userId} has \${count} \${type} violations\`);
      
      // Optionally ban user
      if (count > 50) {
        banUser(userId);
      }
    }
  }
}

const monitor = new SecurityMonitor();

// Use in handlers
ws.on('message', (message) => {
  if (!rateLimiter.check(ws.id)) {
    monitor.recordViolation(ws.userId, 'rate_limit');
    return;
  }
  
  try {
    const data = JSON.parse(message);
    handleMessage(data);
  } catch (error) {
    monitor.recordViolation(ws.userId, 'invalid_message');
  }
});
\`\`\`

**Security Checklist**:

✅ Use WSS (encrypted), not WS  
✅ Authenticate after connection (not in URL)  
✅ Validate Origin header (prevent CSWSH)  
✅ Authorize every action  
✅ Validate and sanitize all messages  
✅ Implement rate limiting (per connection and per user)  
✅ Limit message size  
✅ Limit connections per IP  
✅ Implement heartbeat and connection timeout  
✅ Use server-generated data (timestamps, user IDs)  
✅ Monitor and alert on suspicious activity  
✅ Regularly rotate tokens/secrets  
✅ Log security events (failed auth, rate limits)  

**Defense-in-Depth**: Multiple layers of security make the system resilient to attacks even if one layer is breached.`,
          keyPoints: [
            'Never put credentials in WebSocket URL; authenticate after connection establishment',
            'Check Origin header to prevent Cross-Site WebSocket Hijacking (CSWSH)',
            'Implement rate limiting at both connection and user level to prevent DoS',
            'Always validate and sanitize messages; use server-generated data (userId, timestamp)',
            'Use WSS (encrypted) always; consider mTLS for high-security applications',
            "Authorize every action; don't assume authenticated user has access to all resources",
            'Limit connection count per IP and message size to prevent resource exhaustion',
            'Monitor violations and alert on suspicious activity; implement automatic banning',
          ],
        },
      ],
    },
    {
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

User's perspective: One request, one response.

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
Attacker → Resolver: "It's 6.6.6.6" (fake)
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
      multipleChoice: [
        {
          id: 'dns-query-type',
          question:
            "In a typical DNS resolution, what type of query does a user's device make to the recursive resolver?",
          options: [
            'Iterative query',
            'Recursive query',
            'Authoritative query',
            'Cached query',
          ],
          correctAnswer: 1,
          explanation:
            'Users make recursive queries to their DNS resolver (like 8.8.8.8). The resolver then does all the work of querying root, TLD, and authoritative servers (using iterative queries between servers), and returns the final answer to the user. This is why it\'s called a "recursive" resolver.',
        },
        {
          id: 'dns-cname-limitation',
          question:
            "Why can't you use a CNAME record at the root/apex of a domain (e.g., example.com)?",
          options: [
            'CNAME records are too slow for root domains',
            'It conflicts with required SOA and NS records at the zone apex',
            'Root domains can only use AAAA records',
            'CNAME records are deprecated',
          ],
          correctAnswer: 1,
          explanation:
            'RFC standards prohibit CNAME at the zone apex because every domain must have SOA and NS records, and CNAME means "this is an alias for another name, don\'t look for other records here." Having both would be contradictory. Solutions include using A records or ALIAS records (AWS Route53 proprietary feature).',
        },
        {
          id: 'dns-ttl-tradeoff',
          question:
            "You're planning to migrate your application to new servers. What DNS TTL strategy should you use?",
          options: [
            'Increase TTL to 24 hours before migration for stability',
            'Keep TTL unchanged and migrate immediately',
            'Lower TTL to 60 seconds before migration, wait for old TTL to expire, then migrate',
            'Set TTL to 0 to disable caching',
          ],
          correctAnswer: 2,
          explanation:
            'You should lower TTL before migration (e.g., to 60 seconds), wait for the old TTL period to expire so all caches refresh, then perform the migration. This ensures users switch to new servers quickly. After migration stabilizes, raise TTL back to reduce DNS query load. TTL=0 is often ignored by resolvers.',
        },
        {
          id: 'dns-security',
          question: 'What does DNSSEC primarily protect against?',
          options: [
            'DNS query eavesdropping by ISPs',
            'DNS cache poisoning and spoofing attacks',
            'DDoS attacks on DNS servers',
            'Slow DNS resolution',
          ],
          correctAnswer: 1,
          explanation:
            "DNSSEC uses cryptographic signatures to verify that DNS responses are authentic and haven't been tampered with, protecting against cache poisoning and spoofing. It does NOT encrypt queries (that's DNS over HTTPS/TLS), doesn't prevent DDoS, and doesn't improve speed. It ensures authenticity and integrity.",
        },
        {
          id: 'dns-load-balancing',
          question:
            'What is a major limitation of using round-robin DNS for load balancing?',
          options: [
            'It can only balance between 2 servers',
            'It requires expensive hardware load balancers',
            "It doesn't perform health checks, so traffic goes to failed servers",
            'It only works with IPv6',
          ],
          correctAnswer: 2,
          explanation:
            "Round-robin DNS returns multiple IP addresses, but it has no health checking. If one server fails, DNS still returns its IP, and some users will try to connect to the dead server. Modern solutions like AWS Route53 add health checks, but basic round-robin DNS doesn't have this. It works with unlimited servers and both IPv4/IPv6.",
        },
      ],
      quiz: [
        {
          id: 'dns-global-design',
          question:
            "You're designing a globally distributed web application that serves users in North America, Europe, and Asia. Explain how you would use DNS to optimize latency and provide automatic failover. Include specific DNS features, record types, and discuss trade-offs between different approaches.",
          sampleAnswer: `**Architecture Overview**:

I would implement a multi-layered DNS strategy using GeoDNS, health checking, and automatic failover to optimize global performance and reliability.

**1. DNS Provider Selection**

*Choice: AWS Route 53*
- GeoDNS (geolocation routing)
- Health checking with automatic failover
- Low latency (Anycast network)
- 100% SLA
- Integration with AWS services

*Alternative: Multi-provider setup*
- Primary: Route53
- Secondary: Cloudflare
- Provides redundancy if one DNS provider has outage

**2. Global Infrastructure Setup**

Deploy application in 3 regions:

\`\`\`
US-East (Virginia):
  - Primary: 1.2.3.4
  - Secondary: 1.2.3.5

EU-West (Ireland):
  - Primary: 5.6.7.8
  - Secondary: 5.6.7.9

AP-Southeast (Singapore):
  - Primary: 9.10.11.12
  - Secondary: 9.10.11.13
\`\`\`

**3. DNS Record Structure**

*Root domain record (with geolocation)*:
\`\`\`
# North America
example.com  IN  A  1.2.3.4  (US-East primary)
  - Geolocation: North America
  - TTL: 300 seconds (5 minutes)
  - Health check: HTTPS /health endpoint

# Fallback if US-East fails
example.com  IN  A  1.2.3.5  (US-East secondary)
  - Health check: HTTPS /health endpoint

# Europe
example.com  IN  A  5.6.7.8  (EU-West primary)
  - Geolocation: Europe
  - TTL: 300 seconds

# Asia
example.com  IN  A  9.10.11.12  (AP-Southeast primary)
  - Geolocation: Asia-Pacific
  - TTL: 300 seconds

# Global default (if no geo match)
example.com  IN  A  1.2.3.4  (US-East, largest capacity)
\`\`\`

**4. Health Checking Configuration**

\`\`\`javascript
// Route53 Health Check
{
  type: 'HTTPS',
  resourcePath: '/health',
  port: 443,
  requestInterval: 30, // seconds
  failureThreshold: 3, // 3 consecutive failures
  measureLatency: true,
  regions: ['us-east-1', 'eu-west-1', 'ap-southeast-1']
}

// Health endpoint response
{
  "status": "healthy",
  "timestamp": 1640000000,
  "checks": {
    "database": "ok",
    "cache": "ok",
    "api": "ok"
  }
}
\`\`\`

**5. Failover Logic**

*Scenario: US-East primary fails*

\`\`\`
Time 0:00 - US-East health check fails (1st failure)
Time 0:30 - US-East health check fails (2nd failure)
Time 1:00 - US-East health check fails (3rd failure)
→ Route53 marks unhealthy
→ DNS switches to US-East secondary (1.2.3.5)

Time 1:00 - 6:00 - Users gradually switch (TTL = 5 minutes)
Time 6:00 - All users on secondary

If secondary also fails:
→ Route53 switches to nearest healthy region (EU or Asia)
\`\`\`

**6. TTL Strategy**

*Normal operations: 300 seconds (5 minutes)*
- Short enough for reasonably fast failover
- Long enough to reduce DNS query load
- 95% of queries answered from cache

*During planned maintenance: 60 seconds*
- Lower TTL 1 hour before maintenance
- Allows quick traffic shifting
- Raise back to 300 seconds after

*Trade-off analysis*:

| TTL | Failover Time | DNS Queries/day (1M users) | Benefit |
|-----|---------------|----------------------------|---------|
| 60s | 1-3 minutes | 1.44 billion | Fast failover |
| 300s | 5-10 minutes | 288 million | Balanced |
| 3600s | 1-2 hours | 24 million | Low query load |

**Chosen**: 300 seconds - optimal balance.

**7. Latency Optimization**

*GeoDNS routing*:
- User in New York → US-East (latency: 20ms)
- User in London → EU-West (latency: 15ms)
- User in Tokyo → AP-Southeast (latency: 10ms)

*Without GeoDNS*:
- All users → US-East
- User in London → US-East (latency: 150ms)
- User in Tokyo → US-East (latency: 250ms)

**Expected improvement**: 80-90% latency reduction for international users.

*Latency-based routing (alternative)*:
- Route53 measures actual latency from user to each region
- Routes to lowest-latency endpoint
- More accurate than pure geo-routing
- Trade-off: Requires more complex setup

**8. Monitoring & Alerting**

\`\`\`javascript
// CloudWatch Alarms
{
  metric: 'HealthCheckStatus',
  threshold: 1, // Unhealthy
  evaluationPeriods: 1,
  action: [
    'SNS notification to on-call',
    'PagerDuty alert',
    'Slack webhook'
  ]
}

// DNS query metrics
{
  metric: 'QueryCount',
  dimension: 'Region',
  period: '5 minutes',
  statistic: 'Sum'
}

// Health check latency
{
  metric: 'HealthCheckLatency',
  threshold: 1000, // ms
  action: 'Alert if region degraded'
}
\`\`\`

**9. Cost Analysis**

*Route53 costs*:
- Hosted zone: $0.50/month
- Standard queries: $0.40 per million
- Geo queries: $0.70 per million
- Health checks: $0.50 per endpoint per month

*Calculation (1M users, 300s TTL)*:
- Queries per day: 288M
- Queries per month: 8.6B
- Cost: 8,600 × $0.70 = $6,020/month

*Optimization*:
- Increase TTL to 600s → Halves cost to $3,010/month
- Trade-off: Slower failover (10-15 minutes)

**10. Advanced Features**

*Weighted routing for canary deployments*:
\`\`\`
example.com:
  - 95% → Stable version (1.2.3.4)
  - 5% → Canary version (1.2.3.100)

Monitor canary error rates
If errors < 0.1% → Increase to 50%, then 100%
If errors > 1% → Roll back to 0%
\`\`\`

*Traffic flow with complex routing*:
\`\`\`
Start → Geolocation check
  ├─ North America → Latency-based (US-East vs US-West)
  ├─ Europe → Latency-based (EU-West vs EU-Central)
  └─ Asia → Latency-based (AP-Southeast vs AP-Northeast)

Each region → Weighted routing (canary)
  ├─ 95% → Primary
  └─ 5% → Canary

Each endpoint → Failover routing
  ├─ Primary
  └─ Secondary
\`\`\`

**Trade-offs Discussion**:

**1. GeoDNS vs Latency-Based Routing**

*GeoDNS (chosen)*:
- Pro: Simpler, predictable
- Con: Less accurate (user in London routed to EU might have better latency to US-East if EU degraded)

*Latency-based*:
- Pro: More accurate, routes to actually-fastest endpoint
- Con: More complex, requires continuous latency measurement

**2. Active-Active vs Active-Passive Failover**

*Active-Active (chosen)*:
- All regions serve traffic simultaneously
- Pro: Better resource utilization, no "cold" servers
- Con: More complex, need cross-region data consistency

*Active-Passive*:
- One region primary, others standby
- Pro: Simpler, clear primary
- Con: Wasted capacity in standby regions

**3. Short TTL vs Long TTL**

*Short TTL (300s, chosen)*:
- Pro: Fast failover, quick to adapt to changes
- Con: More DNS queries (higher cost, load)

*Long TTL (3600s)*:
- Pro: 95% fewer DNS queries, lower cost
- Con: Slow failover (1-2 hours)

**4. Single Provider vs Multi-Provider**

*Single (Route53, chosen for simplicity)*:
- Pro: Simpler configuration, single pane of glass
- Con: DNS provider becomes single point of failure

*Multi (Route53 + Cloudflare)*:
- Pro: Redundancy, no single point of failure
- Con: Complex synchronization, higher cost

**Expected Results**:

- **Latency reduction**: 80-90% for international users
- **Availability**: 99.99% (with multi-region failover)
- **Failover time**: 5-10 minutes (with TTL=300s)
- **Cost**: $6,000/month for DNS (1M users)
- **DNS query load**: 288M queries/day (95% cached)

This design provides excellent global performance with automatic failover, balancing complexity, cost, and reliability.`,
          keyPoints: [
            'Use GeoDNS to route users to nearest region (80-90% latency reduction)',
            'Implement health checks with automatic failover to secondary servers',
            'Set TTL to 300 seconds for balance between failover speed and DNS load',
            'Deploy in 3 regions (US, EU, Asia) with primary/secondary in each',
            'Route53 provides GeoDNS, health checking, and automatic failover',
            'Monitor health check status and DNS query patterns for early issue detection',
            'Consider multi-provider DNS (Route53 + Cloudflare) for ultimate reliability',
            'Trade-off: Short TTL (fast failover) vs Long TTL (lower query cost)',
          ],
        },
        {
          id: 'dns-propagation',
          question:
            'Your company is migrating from on-premises servers to AWS. You need to update DNS records to point to new AWS load balancers, but you have 10 million users globally and cannot afford downtime. Design a migration strategy that minimizes risk. Explain how DNS propagation works, potential issues, and your testing approach.',
          sampleAnswer: `**Migration Context**:
- Current: example.com → On-premises (1.2.3.4)
- Target: example.com → AWS ALB (54.210.100.200)
- Users: 10 million globally
- Requirement: Zero downtime

**Understanding DNS Propagation**:

DNS changes don't happen instantly because of caching at multiple levels:

\`\`\`
Browser Cache (2-30 minutes)
    ↓
OS Cache (varies, often 1 hour)
    ↓
Router/ISP Resolver (respects TTL, usually)
    ↓
Intermediate Caches (CDNs, proxies)
    ↓
Authoritative DNS (source of truth)
\`\`\`

**Propagation timeline**:
- Minimum: Your TTL value (e.g., 5 minutes)
- Typical: 1-4 hours (most users)
- Maximum: 24-48 hours (worst case, some resolvers ignore TTL)

**Migration Strategy**:

**Phase 1: Preparation (Day -7 to -1)**

*Step 1.1: Audit current DNS (Day -7)*
\`\`\`bash
# Check current records
dig example.com +short
# Output: 1.2.3.4

# Check TTL
dig example.com +noall +answer
# example.com. 3600 IN A 1.2.3.4
#              ^^^^ Current TTL: 1 hour
\`\`\`

*Step 1.2: Lower TTL (Day -7)*
\`\`\`
Old: example.com. 3600 IN A 1.2.3.4
New: example.com. 300 IN A 1.2.3.4
     (5 minutes)
\`\`\`

**Why**: When we actually change the IP, users will pick up the change in 5 minutes instead of 1 hour.

**Critical**: Wait for old TTL to expire before proceeding (wait 1 hour).

*Step 1.3: Set up AWS infrastructure (Day -7 to -1)*
\`\`\`
AWS ALB: 54.210.100.200
    ↓
Target Group
    ↓
EC2 Instances (or ECS/Lambda)
\`\`\`

*Step 1.4: Deploy application to AWS (Day -3)*
- Deploy code to AWS
- Set up databases, caches, etc.
- DO NOT change DNS yet
- Application running but not receiving prod traffic

*Step 1.5: Test AWS environment (Day -3)*
\`\`\`bash
# Test by direct IP
curl -H "Host: example.com" http://54.210.100.200

# Test by overriding DNS locally
# Add to /etc/hosts:
54.210.100.200 example.com

# Run full test suite
npm run test:integration
\`\`\`

**Phase 2: Migration (Day 0)**

*Step 2.1: Enable AWS read replicas (if using database)*
\`\`\`
On-prem DB → AWS RDS (read replica)
\`\`\`
- Set up real-time replication
- AWS can serve read traffic immediately
- Writes still go to on-prem (for easy rollback)

*Step 2.2: Implement parallel running*
\`\`\`
# Update application to log to both systems
logger.log("Old", event);
logger.logToAWS("New", event);

# Compare outputs (dark launching)
# Ensure AWS behaves identically
\`\`\`

*Step 2.3: Create new DNS record (Hour 0)*
\`\`\`
example.com. 300 IN A 54.210.100.200 (AWS ALB)
\`\`\`

**Change propagates over next 5-30 minutes**:
- Minute 0-5: Early adopters switch to AWS
- Minute 5-10: 50% on AWS, 50% on on-prem
- Minute 10-30: 95% on AWS, 5% on on-prem
- Hour 1-24: Stragglers gradually switch

*Step 2.4: Monitor both environments (Hour 0-2)*

\`\`\`javascript
// CloudWatch Dashboard
Metrics to watch:
  - Old server:
    * Request count (should decline)
    * Error rate (should stay low)
    * Latency (should stay stable)
  
  - New AWS:
    * Request count (should increase)
    * Error rate (MUST stay low)
    * Latency (MUST be similar to old)
    * ALB healthy host count
    * Target response time
\`\`\`

**Thresholds for rollback**:
- Error rate > 0.5% → Roll back
- Latency > 2x normal → Roll back
- Healthy hosts < 50% → Roll back

**Phase 3: Verification & Cleanup (Day 0-7)**

*Step 3.1: Verify propagation (Hour 2)*
\`\`\`bash
# Check from multiple locations
curl https://www.whatsmydns.net/api/dns?server=8.8.8.8&query=example.com

# Expected: Most resolvers return new IP

# Test from different ISPs
dig @8.8.8.8 example.com  # Google DNS
dig @1.1.1.1 example.com  # Cloudflare DNS
dig @208.67.222.222 example.com  # OpenDNS
\`\`\`

*Step 3.2: Monitor traffic split (Hour 0-24)*
\`\`\`
Hour 0: AWS 10%, On-prem 90%
Hour 1: AWS 50%, On-prem 50%
Hour 2: AWS 80%, On-prem 20%
Hour 6: AWS 95%, On-prem 5%
Hour 24: AWS 99%, On-prem 1%
Day 7: AWS 99.9%, On-prem 0.1%
\`\`\`

*Step 3.3: Keep old environment running (Day 0-7)*
- Don't decommission on-prem servers immediately
- Some users may have long-cached old IP
- Wait 7 days before shutting down

*Step 3.4: Database cutover (Day 3, if applicable)*
Once AWS is stable and handling most traffic:
\`\`\`
1. Stop writes to on-prem DB (maintenance mode)
2. Final sync to AWS RDS
3. Promote AWS RDS to primary
4. Resume writes to AWS
5. Downtime: 5-10 minutes
\`\`\`

**Alternative: Gradual weighted cutover**

Instead of immediate switch, use weighted routing:

\`\`\`
Day 0, Hour 0: 
  example.com (weight 95) → On-prem
  example.com (weight 5) → AWS
  (5% of traffic to AWS)

Hour 2 (if stable):
  weight 80 → On-prem
  weight 20 → AWS
  (20% to AWS)

Hour 6 (if stable):
  weight 20 → On-prem
  weight 80 → AWS

Hour 12:
  weight 100 → AWS
  (full cutover)
\`\`\`

**Requires**: DNS provider supporting weighted routing (Route53, Cloudflare)

**Benefit**: More gradual, easier to spot issues early

**Testing Strategy**:

**1. Pre-migration testing**

*Load testing on AWS*:
\`\`\`bash
# Simulate production load
artillery run loadtest.yml \\
  --target https://54.210.100.200 \\
  --output results.json

# Test 10x peak load
k6 run --vus 10000 --duration 30m loadtest.js
\`\`\`

*Chaos testing*:
\`\`\`
- Kill random EC2 instances (verify ALB failover)
- Increase latency artificially (verify timeout handling)
- Simulate database failure (verify replica failover)
\`\`\`

**2. During migration testing**

*Synthetic monitoring*:
\`\`\`javascript
// Pingdom, DataDog, or custom
const monitor = {
  url: 'https://example.com/health',
  interval: 60, // seconds
  locations: ['US-East', 'US-West', 'EU', 'Asia'],
  expectedStatus: 200,
  expectedLatency: '<500ms'
};

// Alert if any location fails 3 consecutive checks
\`\`\`

*Real User Monitoring (RUM)*:
\`\`\`javascript
// Client-side beacon
window.performance.timing;
// Send to analytics:
// - DNS lookup time
// - TCP connect time
// - Time to first byte
// - Full page load

// Compare before/after migration
\`\`\`

**3. Post-migration testing**

*Smoke tests*:
\`\`\`javascript
// Critical user journeys
describe('Post-migration smoke tests', () => {
  test('User can login', async () => {
    await login('test@example.com', 'password');
    expect(page.url()).toBe('https://example.com/dashboard');
  });
  
  test('User can make purchase', async () => {
    await addToCart('product-123');
    await checkout();
    expect(await getOrderStatus()).toBe('confirmed');
  });
});
\`\`\`

*A/B comparison*:
\`\`\`
Compare metrics between:
- Before migration (Day -1)
- After migration (Day +1)

Metrics:
- Error rate (should be equal)
- Latency (should be similar or better)
- Conversion rate (should be similar)
- User complaints (should not increase)
\`\`\`

**Rollback Plan**:

**If issues detected within first 2 hours**:
\`\`\`
1. Change DNS back to on-prem:
   example.com. 300 IN A 1.2.3.4

2. Wait 5-10 minutes for propagation

3. Most users back on stable on-prem

4. Investigate issues on AWS offline

5. Fix and try again tomorrow
\`\`\`

**If issues detected after 6+ hours**:
- More users on AWS, harder to roll back
- Consider fixing forward instead
- Use weighted routing to reduce AWS traffic
- Fix issues under reduced load

**Potential Issues & Mitigations**:

**Issue 1: Some users stuck on old IP**

*Cause*: Resolvers ignoring TTL, long OS cache

*Detection*: Support tickets "site not working"

*Mitigation*:
- Keep old servers running for 7 days
- Show banner: "Having issues? Clear your DNS cache"
- Instructions: ipconfig /flushdns (Windows), sudo dscacheutil -flushcache (Mac)

**Issue 2: Session loss during migration**

*Cause*: Sessions stored locally on old servers

*Mitigation*:
- Migrate to shared session store (Redis) before DNS change
- Use sticky sessions at load balancer
- Accept some session loss (users re-login)

**Issue 3: Database replication lag**

*Cause*: Writes to on-prem, reads from AWS replica

*Mitigation*:
- Monitor replication lag (<1 second acceptable)
- If lag grows, slow down writes or pause migration

**Issue 4: Monitoring blind spot**

*Cause*: Monitoring on old servers, didn't set up on AWS

*Mitigation*:
- Set up CloudWatch, DataDog on AWS BEFORE migration
- Parallel monitoring during cutover
- Don't rely solely on old monitoring

**Expected Timeline & Risk**:

| Phase | Duration | Risk | Traffic on AWS |
|-------|----------|------|----------------|
| Prep | 7 days | Low | 0% |
| DNS change | 5 min | Medium | 0→10% |
| Propagation | 30 min | Medium | 10→80% |
| Stabilization | 6 hours | Low | 80→95% |
| Cleanup | 7 days | Very Low | 95→99.9% |

**Success Criteria**:

✅ Error rate unchanged (<0.1%)  
✅ Latency improved or equal (p99 <500ms)  
✅ Zero data loss  
✅ 99.9% of users migrated within 24 hours  
✅ Rollback plan tested and ready  
✅ All monitoring in place  

**Cost of Gradual Migration**:

- Running both environments for 7 days
- Typical: Double infrastructure cost for 1 week
- ~$10k-50k depending on scale
- **Worth it** to ensure zero downtime

This comprehensive migration strategy ensures zero-downtime migration with multiple safety nets and clear rollback procedures.`,
          keyPoints: [
            'Lower TTL to 300s one week before migration, wait for old TTL to expire',
            'Deploy and test AWS environment thoroughly without changing DNS',
            'Change DNS record and monitor both environments for 2-6 hours',
            'Use weighted routing for gradual cutover (5% → 20% → 80% → 100%)',
            'Keep old environment running for 7 days (some resolvers cache longer)',
            'Implement comprehensive monitoring: error rate, latency, traffic split',
            'Have clear rollback criteria (<0.5% error rate, <2x latency)',
            'Test from multiple locations and ISPs to verify propagation',
          ],
        },
        {
          id: 'dns-ddos',
          question:
            "Explain how DNS amplification DDoS attacks work and how you would protect a high-traffic website's DNS infrastructure from such attacks. Include both preventive measures and mitigation strategies, considering cost and complexity trade-offs.",
          sampleAnswer: `**DNS Amplification Attack Explained**:

DNS amplification is a type of DDoS attack that exploits DNS to overwhelm a target with traffic.

**How it works**:

\`\`\`
Step 1: Attacker sends DNS query with SPOOFED source IP
  ↓
Attacker → DNS Server
Query: "Give me ALL records for example.com"
Source IP: VICTIM'S IP (spoofed)
Request size: ~60 bytes

Step 2: DNS server responds to victim
  ↓
DNS Server → Victim
Response: 4000 bytes of DNS data
(Amplification factor: 70x)

Step 3: Attacker sends millions of such queries
  ↓
Result: Victim receives Gbps of unwanted DNS responses
\`\`\`

**Amplification factor**: Response is 10-100x larger than request

**Why it's effective**:
- UDP allows IP spoofing (no handshake to verify source)
- DNS responses can be large (especially with DNSSEC)
- Open DNS resolvers on internet can be exploited
- Botnet can send millions of requests

**Example attack**:
\`\`\`
Botnet: 10,000 compromised devices
Each sends: 1000 queries/second
Total: 10M queries/second
Amplification: 50x
Result: 500M DNS responses/second sent to victim
        = ~4 Tbps of traffic
\`\`\`

**Protection Strategy**:

**Layer 1: DNS Provider Selection**

*Choose DDoS-resistant DNS provider*:

**Option A: Cloudflare DNS**
- Free tier includes unlimited DDoS protection
- Global Anycast network (200+ locations)
- Absorbs attacks across distributed network
- Cost: $0-200/month
- **Recommended for most websites**

**Option B: AWS Route53**
- Built-in DDoS protection (Shield Standard included)
- Anycast network
- Cost: Based on queries (~$0.40-0.70 per million)
- **Good for AWS-heavy infrastructure**

**Option C: NS1**
- Advanced DDoS protection
- Traffic steering and filtering
- Cost: $500-5000/month
- **For enterprise with complex needs**

*Why this helps*:
- Distributed network absorbs attack traffic
- No single point of failure
- Attack traffic distributed across 100+ servers

**Layer 2: Anycast Architecture**

\`\`\`
Same IP announced from multiple locations:

1.1.1.1 (Cloudflare DNS) announced from:
  - New York
  - London
  - Tokyo
  - ... 200+ locations

Attack traffic to 1.1.1.1:
  - Routed to nearest Cloudflare location
  - Distributed across 200+ servers
  - Each server handles small portion
\`\`\`

**Impact**:
- 100 Gbps attack spread across 200 servers = 500 Mbps per server (manageable)
- Without Anycast: 100 Gbps hits single server (overwhelmed)

**Layer 3: Rate Limiting**

*Per-IP rate limiting*:

\`\`\`
Rate limit DNS queries from single IP:
  - Normal user: 10-50 queries/minute
  - Attacker: 10,000+ queries/minute

Implementation:
  If queries from IP > 100/minute:
    - Drop additional queries
    - Return REFUSED status
    - Temporary ban (1 hour)
\`\`\`

*Example (Cloudflare)*:
\`\`\`
# Cloudflare firewall rule
(dns.qry.name eq "example.com" and dns.qry.rate > 100) 
then action: block
\`\`\`

**Layer 4: Response Rate Limiting (RRL)**

Limit identical responses to same IP:

\`\`\`
If DNS server sending same response to IP repeatedly:
  - First response: Send normally
  - 2nd-5th response (within 1 second): Send normally
  - 6+ responses: DROP or send truncated response

Purpose: Prevent being used as amplifier
\`\`\`

**Implementation (BIND DNS)**:
\`\`\`
rate-limit {
    responses-per-second 5;
    window 1;
    slip 2;
};
\`\`\`

**Layer 5: Query Filtering**

*Block suspicious queries*:

\`\`\`
Block:
  - Queries for "ANY" record type (used in amplification)
  - Excessively large queries
  - Queries from known malicious IPs
  - Queries to non-existent domains (NXDOMAIN flooding)
\`\`\`

*Example (Cloudflare)*:
\`\`\`javascript
// Block ANY queries
if (dns.qry.type == "ANY") {
  return REFUSED;
}

// Block suspicious query names
if (dns.qry.name.contains("amplification")) {
  return REFUSED;
}
\`\`\`

**Layer 6: Hide Authoritative Servers**

*Problem*: Attacker directly targets your authoritative DNS servers

*Solution*: Don't publish authoritative server IPs publicly

\`\`\`
Bad:
  Attacker can find: ns1.example.com → 1.2.3.4
  Directly attack 1.2.3.4

Good:
  Use DNS provider's hidden master
  Provider's Anycast network fronts requests
  Your servers only answer to provider
\`\`\`

**Layer 7: Monitoring & Alerting**

*Metrics to track*:

\`\`\`javascript
// Query rate
queries_per_second {
  threshold: 100000, // Alert if >100k QPS
  window: '1 minute'
}

// NXDOMAIN rate (non-existent domains)
nxdomain_rate {
  threshold: 0.1, // Alert if >10% NXDOMAIN
  window: '5 minutes'
}

// Response size
response_size_bytes {
  threshold: 4096, // Alert if responses consistently large
  window: '5 minutes'
}

// Geographic anomaly
queries_by_country {
  // Alert if sudden spike from unusual country
  threshold: '10x normal'
}
\`\`\`

*Alert channels*:
- PagerDuty (critical)
- Slack (warning)
- Email (info)

**Layer 8: Incident Response Plan**

*When under attack*:

**Step 1: Detect (within 1 minute)**
- Automated monitoring triggers alert
- Query rate > 10x normal
- Response: On-call engineer paged

**Step 2: Analyze (within 5 minutes)**
\`\`\`bash
# Check query sources
dig +stats @dns-server

# Check for amplification patterns
tcpdump -i eth0 'udp port 53' | grep 'ANY'

# Identify attack type:
# - Amplification (ANY queries)
# - Flood (high volume of legitimate-looking queries)
# - NXDOMAIN flood (random subdomains)
\`\`\`

**Step 3: Mitigate (within 15 minutes)**

*If using Cloudflare*:
\`\`\`
1. Enable "I'm Under Attack" mode
   - Challenge requests before resolving
   - Drops most attack traffic

2. Block attacker IPs/ASNs
   - Identify source networks
   - Add firewall rules

3. Enable DNSSEC (if not already)
   - Adds validation step
   - Slightly increases response size (trade-off)
\`\`\`

*If using Route53*:
\`\`\`
1. Enable AWS Shield Advanced ($3000/month)
   - DDoS Response Team
   - Real-time attack visibility
   - Cost protection (refund DDoS-related charges)

2. Use AWS WAF to filter
   - Rate-based rules
   - Geo-blocking if attack from specific region
\`\`\`

**Step 4: Communicate (within 30 minutes)**
- Status page update
- Social media notification
- Email to enterprise customers

**Step 5: Post-incident review (within 48 hours)**
- Attack timeline
- What worked / didn't work
- Improvements needed
- Update runbook

**Defense-in-Depth Summary**:

\`\`\`
Layer 1: DDoS-resistant DNS provider (Cloudflare/Route53)
    ↓
Layer 2: Anycast (distributes attack geographically)
    ↓
Layer 3: Rate limiting (per-IP, per-subnet)
    ↓
Layer 4: Response Rate Limiting (prevent being amplifier)
    ↓
Layer 5: Query filtering (block ANY, suspicious queries)
    ↓
Layer 6: Hide authoritative servers
    ↓
Layer 7: Monitoring & alerting
    ↓
Layer 8: Incident response plan
\`\`\`

**Cost-Benefit Analysis**:

| Solution | Cost/month | Protection Level | Complexity |
|----------|------------|------------------|------------|
| Cloudflare Free | $0 | High | Low |
| Cloudflare Pro | $20 | Very High | Low |
| Route53 + Shield Standard | $50-500 | High | Medium |
| Route53 + Shield Advanced | $3000+ | Very High | Medium |
| Custom DNS + DDoS protection | $5000+ | Custom | High |

**Recommendation by scale**:

**Small-Medium (< 1M users)**:
- Cloudflare Free or Pro
- Cost: $0-20/month
- Protection: Sufficient for most attacks

**Medium-Large (1M-10M users)**:
- Cloudflare Business or Route53
- Cost: $200-500/month
- Add: WAF, rate limiting rules

**Enterprise (10M+ users)**:
- Multiple DNS providers (Cloudflare + Route53)
- AWS Shield Advanced
- Dedicated DDoS team
- Cost: $5000-20000/month
- Protection: Can handle 10+ Tbps attacks

**Advanced Techniques**:

**1. DNS Cookies (RFC 7873)**

Prevent query flooding with client cookies:

\`\`\`
Client → Server: Query + Cookie
Server validates cookie
Invalid cookie → Drop query

Result: Attacker can't spoof + cookie = much harder attack
\`\`\`

**2. DNSSEC**

While primarily for authentication, DNSSEC helps with DDoS:
- Larger responses (can hurt if you're the amplifier)
- But validates responses (prevents some attack types)
- **Trade-off**: Complexity vs security

**3. BGP Blackholing**

For massive attacks:
\`\`\`
Advertise route to /dev/null
Attack traffic dropped at ISP level
Before it reaches your network

Downside: Also drops legitimate traffic
Use only as last resort
\`\`\`

**4. Scrubbing Centers**

Route traffic through DDoS scrubbing service:
\`\`\`
Internet → Scrubbing Center → Your Servers
                ↓
          (Filters attack traffic)
\`\`\`

Providers: Cloudflare, Akamai, Arbor Networks

**Real-World Examples**:

**Dyn Attack (2016)**:
- Mirai botnet targeted Dyn (major DNS provider)
- 1.2 Tbps attack
- Knocked out Twitter, Reddit, GitHub
- Lesson: Use multiple DNS providers

**Cloudflare (2020)**:
- 17.2 million requests/second
- Attack absorbed with no customer impact
- Demonstrates value of Anycast + scale

**Key Takeaways**:

1. Use DDoS-resistant DNS provider with Anycast (Cloudflare, Route53)
2. Implement multi-layered defense (rate limiting, RRL, filtering)
3. Monitor continuously for anomalies
4. Have incident response plan ready
5. Consider redundant DNS providers for mission-critical sites
6. Cost ranges from $0 (Cloudflare Free) to $3000+ (Shield Advanced) depending on scale
7. Prevention is cheaper than mitigation - set up defenses before attack

**Bottom Line**: 
For most websites, **Cloudflare** (free or $20/month) provides excellent DNS DDoS protection with minimal complexity. For AWS-heavy or enterprise deployments, **Route53 + Shield** offers integrated protection. The key is to act proactively before an attack, not reactively during one.`,
          keyPoints: [
            'DNS amplification exploits open resolvers to send massive responses to victim (70x amplification)',
            'Use DDoS-resistant DNS provider with Anycast (Cloudflare, Route53) to distribute attack',
            'Implement rate limiting per-IP and Response Rate Limiting (RRL) to prevent abuse',
            'Filter suspicious queries (ANY records, excessive query rates, malicious IPs)',
            "Hide authoritative DNS servers behind provider's network",
            'Monitor query rate, NXDOMAIN rate, and geographic anomalies for early detection',
            'Have incident response plan ready: detect, analyze, mitigate, communicate',
            'For most sites, Cloudflare ($0-20/month) provides excellent protection with low complexity',
          ],
        },
      ],
    },
    {
      id: 'rpc-remote-procedure-call',
      title: 'RPC (Remote Procedure Call)',
      content: `RPC (Remote Procedure Call) is a protocol that allows a program to execute procedures on a remote system as if they were local function calls. Understanding RPC is crucial for designing distributed systems, microservices, and high-performance APIs.

## What is RPC?

**RPC (Remote Procedure Call)** enables a client to call functions on a remote server using the same syntax as local function calls.

**Conceptual Flow**:
\`\`\`
Client Code:
  result = calculateTax(amount, state)
  
Behind the Scenes:
  1. Client stub marshals parameters (serializes)
  2. Message sent over network
  3. Server stub unmarshals parameters
  4. Server executes calculateTax(amount, state)
  5. Result marshaled and sent back
  6. Client receives and unmarshals result
\`\`\`

**Key Characteristics**:
- **Transparency**: Calling remote functions looks like calling local functions
- **Synchronous**: Typically blocking (client waits for response)
- **Strongly Typed**: Interface defined via IDL (Interface Definition Language)
- **Binary Protocol**: Usually more efficient than REST/JSON

---

## Popular RPC Frameworks

### **1. gRPC (Google RPC)**

**Overview**: Modern, high-performance RPC framework using HTTP/2 and Protocol Buffers.

**Key Features**:
- Uses Protocol Buffers (protobuf) for serialization
- HTTP/2 for transport (multiplexing, streaming)
- Support for 4 communication patterns
- Strong typing with code generation
- Built-in authentication, load balancing, health checking

**Example Proto File**:
\`\`\`protobuf
// user.proto
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUser(CreateUserRequest) returns (User);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
  int32 age = 3;
}
\`\`\`

**Node.js gRPC Server**:
\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;

// Implement service methods
function getUser(call, callback) {
  const userId = call.request.id;
  
  // Fetch from database
  const user = {
    id: userId,
    name: 'John Doe',
    email: 'john@example.com',
    age: 30
  };
  
  callback(null, user);
}

function listUsers(call) {
  const users = [
    { id: '1', name: 'Alice', email: 'alice@example.com', age: 25 },
    { id: '2', name: 'Bob', email: 'bob@example.com', age: 30 },
    { id: '3', name: 'Charlie', email: 'charlie@example.com', age: 35 }
  ];
  
  // Stream users one by one
  users.forEach(user => call.write(user));
  call.end();
}

function createUser(call, callback) {
  const newUser = {
    id: generateId(),
    name: call.request.name,
    email: call.request.email,
    age: call.request.age
  };
  
  // Save to database
  callback(null, newUser);
}

// Create and start server
const server = new grpc.Server();
server.addService(userProto.UserService.service, {
  getUser,
  listUsers,
  createUser
});

server.bindAsync(
  '0.0.0.0:50051',
  grpc.ServerCredentials.createInsecure(),
  () => {
    console.log('gRPC server running on port 50051');
    server.start();
  }
);
\`\`\`

**Node.js gRPC Client**:
\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;

const client = new userProto.UserService(
  'localhost:50051',
  grpc.credentials.createInsecure()
);

// Unary RPC (single request, single response)
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    console.error('Error:', error);
    return;
  }
  console.log('User:', user);
});

// Server streaming RPC
const call = client.listUsers({ page_size: 10 });
call.on('data', (user) => {
  console.log('Received user:', user);
});
call.on('end', () => {
  console.log('All users received');
});
call.on('error', (error) => {
  console.error('Stream error:', error);
});

// Create user
client.createUser(
  { name: 'Jane Doe', email: 'jane@example.com', age: 28 },
  (error, user) => {
    if (error) {
      console.error('Error:', error);
      return;
    }
    console.log('Created user:', user);
  }
);
\`\`\`

---

### **2. Apache Thrift**

**Overview**: Cross-language RPC framework developed by Facebook, supports multiple protocols and transports.

**Key Features**:
- Multiple protocols (binary, compact, JSON)
- Multiple transports (TCP, HTTP, framed)
- Code generation for many languages
- Flexible and extensible

**Thrift IDL Example**:
\`\`\`thrift
// user.thrift
namespace js UserService

struct User {
  1: string id,
  2: string name,
  3: string email,
  4: i32 age
}

exception UserNotFound {
  1: string message
}

service UserService {
  User getUser(1: string id) throws (1: UserNotFound notFound),
  list<User> listUsers(1: i32 pageSize),
  User createUser(1: string name, 2: string email, 3: i32 age)
}
\`\`\`

---

### **3. JSON-RPC**

**Overview**: Lightweight RPC protocol using JSON for encoding.

**Key Features**:
- Simple and human-readable
- Transport-agnostic (HTTP, WebSocket, TCP)
- Language-agnostic
- No code generation required

**JSON-RPC Request**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "method": "user.getUser",
  "params": {
    "id": "123"
  },
  "id": 1
}
\`\`\`

**JSON-RPC Response**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "result": {
    "id": "123",
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
  },
  "id": 1
}
\`\`\`

**JSON-RPC Error Response**:
\`\`\`json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32600,
    "message": "User not found"
  },
  "id": 1
}
\`\`\`

---

## gRPC Communication Patterns

### **1. Unary RPC** (Request-Response)

Single request → Single response (like REST)

\`\`\`protobuf
rpc GetUser(GetUserRequest) returns (User);
\`\`\`

**Use Cases**:
- CRUD operations
- Simple queries
- Most common pattern

---

### **2. Server Streaming RPC**

Single request → Stream of responses

\`\`\`protobuf
rpc ListUsers(ListUsersRequest) returns (stream User);
\`\`\`

**Use Cases**:
- Large result sets
- Real-time updates
- File downloads

**Example**:
\`\`\`javascript
// Server
function listUsers(call) {
  const users = fetchAllUsers(); // Could be millions
  
  users.forEach(user => {
    call.write(user); // Stream one by one
  });
  
  call.end();
}

// Client
const call = client.listUsers({});
call.on('data', (user) => {
  processUser(user); // Process as they arrive
});
\`\`\`

---

### **3. Client Streaming RPC**

Stream of requests → Single response

\`\`\`protobuf
rpc UploadFile(stream FileChunk) returns (UploadResponse);
\`\`\`

**Use Cases**:
- File uploads
- Batch operations
- Collecting metrics

**Example**:
\`\`\`javascript
// Client
const call = client.uploadFile((error, response) => {
  console.log('Upload complete:', response);
});

// Stream file chunks
fileChunks.forEach(chunk => {
  call.write(chunk);
});

call.end();

// Server
function uploadFile(call, callback) {
  const chunks = [];
  
  call.on('data', (chunk) => {
    chunks.push(chunk);
  });
  
  call.on('end', () => {
    const file = Buffer.concat(chunks);
    saveFile(file);
    callback(null, { success: true, size: file.length });
  });
}
\`\`\`

---

### **4. Bidirectional Streaming RPC**

Stream of requests ↔ Stream of responses (independent)

\`\`\`protobuf
rpc Chat(stream ChatMessage) returns (stream ChatMessage);
\`\`\`

**Use Cases**:
- Chat applications
- Real-time collaboration
- Live gaming

**Example**:
\`\`\`javascript
// Client
const call = client.chat();

// Send messages
call.write({ user: 'Alice', message: 'Hello!' });
call.write({ user: 'Alice', message: 'How are you?' });

// Receive messages
call.on('data', (message) => {
  console.log(\`\${message.user}: \${message.message}\`);
});

// Server
function chat(call) {
  call.on('data', (message) => {
    // Broadcast to all connected clients
    broadcastToAll(message);
  });
  
  call.on('end', () => {
    call.end();
  });
}
\`\`\`

---

## RPC vs REST vs GraphQL

| **Aspect** | **RPC (gRPC)** | **REST** | **GraphQL** |
|------------|----------------|----------|-------------|
| **Protocol** | HTTP/2, Binary (Protobuf) | HTTP/1.1, JSON | HTTP, JSON |
| **Performance** | ⚡ Very Fast (binary, multiplexing) | 🐢 Slower (text, sequential) | 🏃 Medium (single endpoint) |
| **Type Safety** | ✅ Strong (code generation) | ❌ Weak (manual validation) | ✅ Strong (schema) |
| **Discoverability** | ⚠️ Requires documentation | ✅ Self-documenting (HATEOAS) | ✅ Self-documenting (introspection) |
| **Caching** | ❌ Difficult (HTTP/2, binary) | ✅ Easy (HTTP caching) | ⚠️ Moderate (requires work) |
| **Streaming** | ✅ Native support | ❌ No native support | ⚠️ Via subscriptions |
| **Browser Support** | ⚠️ Requires gRPC-Web | ✅ Native | ✅ Native |
| **Learning Curve** | ⚠️ Steeper | ✅ Easy | ⚠️ Moderate |
| **Best For** | Microservices, internal APIs | Public APIs, CRUD | Complex queries, mobile |

---

## When to Use RPC

### **✅ Use RPC When:**

1. **Internal Microservices Communication**
   - Services within same organization
   - Strong typing and performance critical
   - Example: Order Service → Inventory Service

2. **High-Performance Requirements**
   - Low latency critical
   - High throughput needed
   - Example: Trading systems, real-time analytics

3. **Streaming Data**
   - Server streaming (logs, metrics)
   - Client streaming (file uploads)
   - Bidirectional (chat, collaboration)

4. **Polyglot Environments**
   - Multiple programming languages
   - Need consistent interfaces
   - Code generation valuable

5. **Complex Operations**
   - Actions that don't map to CRUD
   - Procedure-oriented APIs
   - Example: processOrder(), runAnalysis()

### **❌ Avoid RPC When:**

1. **Public APIs**
   - External developers need access
   - REST more familiar/accessible
   - Caching important

2. **Browser-Only Clients**
   - gRPC-Web adds complexity
   - REST/GraphQL more natural
   - Unless using gRPC-Web proxy

3. **Simple CRUD Operations**
   - REST sufficient
   - No performance requirements
   - Standard HTTP caching desired

4. **Debugging/Testing Priority**
   - Binary protocols harder to inspect
   - curl/Postman not usable
   - Tools less mature

---

## RPC Error Handling

### **gRPC Status Codes**:

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// Common status codes
grpc.status.OK                  // 0 - Success
grpc.status.CANCELLED           // 1 - Operation cancelled
grpc.status.UNKNOWN             // 2 - Unknown error
grpc.status.INVALID_ARGUMENT    // 3 - Invalid argument
grpc.status.DEADLINE_EXCEEDED   // 4 - Timeout
grpc.status.NOT_FOUND           // 5 - Not found
grpc.status.ALREADY_EXISTS      // 6 - Already exists
grpc.status.PERMISSION_DENIED   // 7 - Permission denied
grpc.status.UNAUTHENTICATED     // 16 - Not authenticated
grpc.status.UNAVAILABLE         // 14 - Service unavailable
grpc.status.INTERNAL            // 13 - Internal error
\`\`\`

**Server-Side Error Handling**:
\`\`\`javascript
function getUser(call, callback) {
  const userId = call.request.id;
  
  if (!userId) {
    return callback({
      code: grpc.status.INVALID_ARGUMENT,
      message: 'User ID is required'
    });
  }
  
  const user = database.findUser(userId);
  
  if (!user) {
    return callback({
      code: grpc.status.NOT_FOUND,
      message: \`User \${userId} not found\`
    });
  }
  
  callback(null, user);
}
\`\`\`

**Client-Side Error Handling with Retry**:
\`\`\`javascript
function getUserWithRetry(userId, maxRetries = 3) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    
    function attempt() {
      attempts++;
      
      client.getUser({ id: userId }, (error, user) => {
        if (!error) {
          return resolve(user);
        }
        
        // Retry on transient errors
        if (
          error.code === grpc.status.UNAVAILABLE ||
          error.code === grpc.status.DEADLINE_EXCEEDED
        ) {
          if (attempts < maxRetries) {
            const delay = Math.min(1000 * Math.pow(2, attempts), 10000);
            console.log(\`Retry \${attempts} after \${delay}ms\`);
            setTimeout(attempt, delay);
            return;
          }
        }
        
        // Non-retryable error or max retries exceeded
        reject(error);
      });
    }
    
    attempt();
  });
}
\`\`\`

---

## RPC Performance Optimization

### **1. Connection Pooling**

\`\`\`javascript
// Bad: New connection per request
function makeRequest() {
  const client = new UserService('localhost:50051');
  client.getUser({ id: '123' }, callback);
}

// Good: Reuse connection
const client = new UserService('localhost:50051');

function makeRequest() {
  client.getUser({ id: '123' }, callback);
}
\`\`\`

### **2. Timeouts and Deadlines**

\`\`\`javascript
// Set deadline (absolute time)
const deadline = new Date();
deadline.setSeconds(deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  (error, user) => {
    if (error && error.code === grpc.status.DEADLINE_EXCEEDED) {
      console.error('Request timed out');
    }
  }
);
\`\`\`

### **3. Compression**

\`\`\`javascript
// Enable compression
client.getUser(
  { id: '123' },
  {
    'grpc.default_compression_algorithm': grpc.compressionAlgorithms.gzip,
    'grpc.default_compression_level': grpc.compressionLevels.high
  },
  callback
);
\`\`\`

### **4. Multiplexing**

HTTP/2 automatically multiplexes multiple RPC calls over single TCP connection:

\`\`\`javascript
// All these calls use same connection automatically
client.getUser({ id: '1' }, callback1);
client.getUser({ id: '2' }, callback2);
client.getUser({ id: '3' }, callback3);
// No connection overhead! HTTP/2 multiplexes.
\`\`\`

---

## RPC Security

### **1. TLS/SSL Encryption**

\`\`\`javascript
// Server with TLS
const credentials = grpc.ServerCredentials.createSsl(
  fs.readFileSync('ca.crt'),
  [{
    cert_chain: fs.readFileSync('server.crt'),
    private_key: fs.readFileSync('server.key')
  }]
);

server.bindAsync('0.0.0.0:50051', credentials, () => {
  server.start();
});

// Client with TLS
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.crt')
);

const client = new UserService('localhost:50051', credentials);
\`\`\`

### **2. Authentication with Metadata**

\`\`\`javascript
// Client: Send auth token
const metadata = new grpc.Metadata();
metadata.add('authorization', 'Bearer eyJhbGc...');

client.getUser({ id: '123' }, metadata, callback);

// Server: Verify token
function getUser(call, callback) {
  const metadata = call.metadata;
  const authHeader = metadata.get('authorization')[0];
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Missing or invalid token'
    });
  }
  
  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    // Proceed with request
    callback(null, user);
  } catch (error) {
    callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Invalid token'
    });
  }
}
\`\`\`

### **3. Interceptors for Cross-Cutting Concerns**

\`\`\`javascript
// Client interceptor (add auth to all requests)
function authInterceptor(options, nextCall) {
  return new grpc.InterceptingCall(nextCall(options), {
    start: (metadata, listener, next) => {
      metadata.add('authorization', \`Bearer \${getAuthToken()}\`);
      next(metadata, listener);
    }
  });
}

const client = new UserService(
  'localhost:50051',
  credentials,
  { interceptors: [authInterceptor] }
);

// Server interceptor (logging, auth)
function loggingInterceptor(call, callback) {
  const start = Date.now();
  const method = call.handler.path;
  
  console.log(\`[RPC] \${method} started\`);
  
  // Call original method
  return (originalCall, originalCallback) => {
    originalCallback((error, response) => {
      const duration = Date.now() - start;
      console.log(\`[RPC] \${method} completed in \${duration}ms\`);
      callback(error, response);
    });
  };
}
\`\`\`

---

## Load Balancing RPC Services

### **Client-Side Load Balancing**:

\`\`\`javascript
// gRPC supports DNS-based load balancing
const client = new UserService(
  'dns:///userservice.example.com:50051',
  credentials
);

// Round-robin across all resolved IPs
\`\`\`

### **Server-Side Load Balancing with Envoy**:

\`\`\`yaml
# envoy.yaml
static_resources:
  listeners:
    - address:
        socket_address:
          address: 0.0.0.0
          port_value: 50051
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                http2_protocol_options: {}
                route_config:
                  virtual_hosts:
                    - name: userservice
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route:
                            cluster: userservice_cluster
  clusters:
    - name: userservice_cluster
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      http2_protocol_options: {}
      load_assignment:
        cluster_name: userservice_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: userservice1
                      port_value: 50051
              - endpoint:
                  address:
                    socket_address:
                      address: userservice2
                      port_value: 50051
              - endpoint:
                  address:
                    socket_address:
                      address: userservice3
                      port_value: 50051
\`\`\`

---

## Common RPC Mistakes

### **❌ Mistake 1: No Timeout/Deadline**

\`\`\`javascript
// Bad: No timeout
client.getUser({ id: '123' }, callback);
// If server hangs, client waits forever

// Good: Always set deadline
const deadline = Date.now() + 5000; // 5 seconds
client.getUser({ id: '123' }, { deadline }, callback);
\`\`\`

### **❌ Mistake 2: Not Handling Errors Properly**

\`\`\`javascript
// Bad: Treating all errors the same
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    throw error; // Don't retry transient errors!
  }
});

// Good: Retry transient errors
client.getUser({ id: '123' }, (error, user) => {
  if (error) {
    if (error.code === grpc.status.UNAVAILABLE) {
      // Retry with backoff
      retryWithBackoff();
    } else if (error.code === grpc.status.NOT_FOUND) {
      // Don't retry, return 404
      return res.status(404).send('User not found');
    } else {
      // Log and return 500
      logger.error(error);
      return res.status(500).send('Internal error');
    }
  }
});
\`\`\`

### **❌ Mistake 3: Creating New Client Per Request**

\`\`\`javascript
// Bad: Connection overhead
app.get('/user/:id', (req, res) => {
  const client = new UserService('localhost:50051');
  client.getUser({ id: req.params.id }, callback);
});

// Good: Reuse client
const client = new UserService('localhost:50051');

app.get('/user/:id', (req, res) => {
  client.getUser({ id: req.params.id }, callback);
});
\`\`\`

### **❌ Mistake 4: Not Using Streaming for Large Data**

\`\`\`javascript
// Bad: Load all users in memory
rpc GetUsers(Empty) returns (UserList); // Contains array of ALL users

// Good: Stream users
rpc GetUsers(Empty) returns (stream User); // Stream one at a time
\`\`\`

### **❌ Mistake 5: No Monitoring/Observability**

Always add:
- Request duration metrics
- Error rate by status code
- Request rate (QPS)
- Connection pool size

---

## Real-World Example: Microservices with gRPC

**Scenario**: E-commerce platform with Order Service, Inventory Service, Payment Service.

**Architecture**:
\`\`\`
API Gateway (REST)
      |
      v
Order Service (gRPC)
      |
      +---> Inventory Service (gRPC)
      |
      +---> Payment Service (gRPC)
\`\`\`

**order.proto**:
\`\`\`protobuf
syntax = "proto3";

import "inventory.proto";
import "payment.proto";

service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  rpc GetOrder(GetOrderRequest) returns (Order);
}

message CreateOrderRequest {
  string user_id = 1;
  repeated OrderItem items = 2;
  PaymentInfo payment_info = 3;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}

message Order {
  string order_id = 1;
  string user_id = 2;
  repeated OrderItem items = 3;
  string status = 4;
  double total = 5;
}
\`\`\`

**Order Service Implementation**:
\`\`\`javascript
async function createOrder(call, callback) {
  const { user_id, items, payment_info } = call.request;
  
  try {
    // 1. Check inventory
    const inventoryClient = getInventoryClient();
    const inventoryCheck = await new Promise((resolve, reject) => {
      inventoryClient.checkAvailability(
        { items },
        { deadline: Date.now() + 2000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    if (!inventoryCheck.available) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Items not available'
      });
    }
    
    // 2. Process payment
    const paymentClient = getPaymentClient();
    const payment = await new Promise((resolve, reject) => {
      paymentClient.processPayment(
        {
          amount: inventoryCheck.total,
          payment_info
        },
        { deadline: Date.now() + 5000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    if (!payment.success) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Payment failed'
      });
    }
    
    // 3. Reserve inventory
    await new Promise((resolve, reject) => {
      inventoryClient.reserveItems(
        { items, order_id: generateOrderId() },
        { deadline: Date.now() + 2000 },
        (error, response) => {
          if (error) reject(error);
          else resolve(response);
        }
      );
    });
    
    // 4. Create order
    const order = {
      order_id: generateOrderId(),
      user_id,
      items,
      status: 'confirmed',
      total: inventoryCheck.total
    };
    
    await saveOrder(order);
    
    callback(null, order);
    
  } catch (error) {
    logger.error('Order creation failed:', error);
    
    // Rollback if needed
    if (error.code === grpc.status.UNAVAILABLE) {
      return callback({
        code: grpc.status.UNAVAILABLE,
        message: 'Service temporarily unavailable'
      });
    }
    
    callback({
      code: grpc.status.INTERNAL,
      message: 'Failed to create order'
    });
  }
}
\`\`\`

**Benefits of Using gRPC Here**:
1. **Performance**: Binary protocol, HTTP/2 multiplexing
2. **Type Safety**: Protobuf definitions prevent errors
3. **Streaming**: Can stream order updates
4. **Polyglot**: Services can be in different languages
5. **Timeouts**: Built-in deadline handling
6. **Load Balancing**: Native support

---

## Key Takeaways

1. **RPC allows calling remote functions** as if they were local
2. **gRPC uses HTTP/2 + Protocol Buffers** for high performance
3. **4 communication patterns**: Unary, Server Streaming, Client Streaming, Bidirectional
4. **Best for internal microservices**, not public APIs
5. **Always set timeouts/deadlines** to prevent hanging
6. **Retry transient errors** (UNAVAILABLE, DEADLINE_EXCEEDED)
7. **Use TLS for security**, metadata for auth
8. **Streaming better than loading** all data in memory
9. **Connection pooling** critical for performance
10. **Trade-off: Performance vs ease of use** (gRPC vs REST)`,
      multipleChoice: [
        {
          id: 'rpc-grpc-transport',
          question:
            'What transport protocol and serialization format does gRPC use by default?',
          options: [
            'HTTP/1.1 with JSON',
            'HTTP/2 with Protocol Buffers',
            'TCP with XML',
            'WebSocket with MessagePack',
          ],
          correctAnswer: 1,
          explanation:
            'gRPC uses HTTP/2 as the transport protocol (enabling multiplexing, streaming, header compression) and Protocol Buffers (protobuf) for serialization. This combination provides high performance through binary encoding and efficient network usage. HTTP/1.1 with JSON is used by REST APIs and is much slower.',
        },
        {
          id: 'rpc-streaming-pattern',
          question:
            'Which gRPC streaming pattern would be most appropriate for implementing a real-time chat application where multiple users send and receive messages simultaneously?',
          options: [
            'Unary RPC',
            'Server Streaming RPC',
            'Client Streaming RPC',
            'Bidirectional Streaming RPC',
          ],
          correctAnswer: 3,
          explanation:
            'Bidirectional Streaming RPC is ideal for chat applications because both client and server need to send streams of messages independently and simultaneously. The client streams outgoing messages while receiving incoming messages from the server. Unary RPC would require polling, server streaming only allows server→client, and client streaming only allows client→server.',
        },
        {
          id: 'rpc-error-handling',
          question:
            "You're implementing a gRPC client that calls a downstream service. The call fails with status code UNAVAILABLE. What is the most appropriate action?",
          options: [
            'Immediately return an error to the user',
            'Retry the request with exponential backoff',
            'Log the error and continue without retrying',
            'Switch to a different RPC method',
          ],
          correctAnswer: 1,
          explanation:
            'UNAVAILABLE is a transient error indicating the service is temporarily down or overloaded. The correct approach is to retry with exponential backoff (e.g., 100ms, 200ms, 400ms, 800ms) up to a maximum number of attempts. This gives the service time to recover. Immediately returning an error provides poor user experience, and not retrying at all misses the opportunity for the call to succeed.',
        },
        {
          id: 'rpc-vs-rest',
          question:
            'In which scenario would REST be a better choice than gRPC for API design?',
          options: [
            'Internal microservices communication requiring low latency',
            'Public API that needs to be easily accessible from browsers without additional tooling',
            'High-throughput streaming of binary data between services',
            'Polyglot environment where multiple languages need strongly-typed interfaces',
          ],
          correctAnswer: 1,
          explanation:
            'REST is better for public APIs consumed by browsers because it works natively with HTTP/1.1, requires no special tooling, supports standard HTTP caching, and can be tested with curl/Postman. gRPC requires gRPC-Web for browsers, which adds complexity. For internal microservices, streaming, and polyglot environments, gRPC is typically superior due to performance and strong typing.',
        },
        {
          id: 'rpc-connection-management',
          question:
            'What is the most important practice for managing gRPC client connections in a high-traffic Node.js application?',
          options: [
            'Create a new client instance for every request',
            'Reuse a single client instance across all requests',
            'Create a client pool and rotate through clients',
            'Close and recreate the client every 100 requests',
          ],
          correctAnswer: 1,
          explanation:
            "Reusing a single gRPC client instance across all requests is critical because: (1) HTTP/2 automatically multiplexes multiple concurrent RPCs over a single TCP connection, (2) Creating new connections has significant overhead (TCP handshake, TLS handshake), (3) gRPC clients maintain connection pools internally. Creating a new client per request would cause severe performance degradation. Unlike HTTP/1.1 where connection pooling is necessary, HTTP/2's multiplexing makes a single client instance optimal.",
        },
      ],
      quiz: [
        {
          id: 'rpc-microservices-migration',
          question:
            'Your company is migrating from a monolithic application to microservices. The architecture team is debating between using REST APIs and gRPC for inter-service communication. You have 15 services written in Java, Python, and Node.js. The system handles financial transactions requiring low latency (<50ms) and high reliability. Provide a detailed recommendation on which approach to use, including specific technical justifications, trade-offs, and a migration strategy.',
          sampleAnswer: `**Recommendation: Use gRPC for internal microservices, REST for public APIs**

**Technical Justification**:

1. **Performance Requirements**:
   - Financial transactions require <50ms latency
   - gRPC with Protocol Buffers: ~1-2ms serialization overhead
   - REST with JSON: ~5-10ms serialization overhead
   - gRPC uses HTTP/2 multiplexing → multiple calls over single connection
   - REST (HTTP/1.1) → separate connection per request (connection pooling helps but still overhead)
   - **Result**: gRPC provides 3-5x better latency

2. **Reliability Requirements**:
   - gRPC has built-in: deadlines, retries, health checking, load balancing
   - Strong typing via protobuf prevents serialization errors
   - Code generation ensures client/server contract matching
   - REST requires implementing these manually

3. **Polyglot Support** (Java, Python, Node.js):
   - gRPC has excellent support for all three languages
   - Protobuf definitions generate idiomatic code
   - Single .proto file defines contract for all services
   - REST requires maintaining separate client libraries or OpenAPI specs

**Trade-offs**:

| **Aspect** | **gRPC** | **REST** |
|------------|----------|----------|
| **Performance** | ⚡ Excellent | 🐢 Good |
| **Type Safety** | ✅ Strong | ❌ Weak |
| **Debugging** | ⚠️ Harder (binary) | ✅ Easy (curl, Postman) |
| **Browser Support** | ⚠️ Requires gRPC-Web | ✅ Native |
| **Learning Curve** | ⚠️ Steeper | ✅ Familiar |
| **Ecosystem** | ⚠️ Maturing | ✅ Mature |

**Migration Strategy**:

**Phase 1: Infrastructure Setup** (Week 1-2)
- Set up protobuf compilation in build pipelines
- Create shared proto repository for service contracts
- Implement gRPC server/client scaffolding in Java, Python, Node.js
- Set up Envoy proxy for load balancing and observability

**Phase 2: Pilot Service** (Week 3-4)
- Choose low-risk service (e.g., User Profile Service)
- Implement gRPC version alongside existing REST endpoints
- Run in parallel, measure latency improvements
- Train team on gRPC development and debugging

**Phase 3: Critical Path Services** (Week 5-8)
- Migrate transaction processing services (most latency-sensitive)
- Order Service → Payment Service → Inventory Service
- Keep REST API Gateway for external clients
- Use gRPC for internal service-to-service calls

**Phase 4: Remaining Services** (Week 9-12)
- Migrate remaining services in dependency order
- Retire old REST endpoints after validation
- Monitor error rates, latency, and throughput

**Architecture**:

\`\`\`
Public Clients (Web, Mobile)
         ↓
    API Gateway (REST)
         ↓
   [Internal Services use gRPC]
         ↓
   +--- Order Service (gRPC)
   |         ↓
   |    Payment Service (gRPC)
   |         ↓
   |    Inventory Service (gRPC)
   |
   +--- User Service (gRPC)
   |
   +--- Analytics Service (gRPC)
\`\`\`

**Observability**:
- Use Envoy for metrics (latency percentiles, error rates, QPS)
- Implement distributed tracing (Jaeger/Zipkin)
- Dashboard: gRPC vs REST latency comparison
- Alert on: p99 latency >50ms, error rate >1%

**Expected Results**:
- 3-5x latency improvement for service-to-service calls
- Reduced serialization errors (strong typing)
- Simpler client code (code generation)
- Better resource utilization (fewer connections, less CPU for serialization)

**Risks and Mitigations**:
- **Risk**: Team unfamiliar with gRPC → **Mitigation**: Training + pilot service
- **Risk**: Debugging more difficult → **Mitigation**: grpcurl, grpc-web devtools
- **Risk**: Breaking changes in protos → **Mitigation**: Versioning strategy, backwards compatibility
- **Risk**: Migration bugs → **Mitigation**: Parallel running, gradual rollout, feature flags

**Final Recommendation**: Use gRPC for internal services (performance + reliability), keep REST for public API (accessibility). This hybrid approach gives you the best of both worlds.`,
          keyPoints: [
            'Use gRPC for internal microservices (low latency, type safety)',
            'Keep REST for public APIs (accessibility, browser support)',
            'gRPC provides 3-5x better performance for inter-service communication',
            'Hybrid approach: gRPC internally, REST gateway for external clients',
            'Consider team expertise and tooling maturity in decision',
          ],
        },
        {
          id: 'rpc-error-handling-strategy',
          question:
            'Design a comprehensive error handling and retry strategy for a distributed system using gRPC. Your system has an Order Service that calls Inventory Service, Payment Service, and Shipping Service. Each downstream service has different SLAs and failure characteristics. Include specific gRPC status codes, retry policies, timeout configurations, circuit breaker patterns, and how to handle partial failures.',
          sampleAnswer: `**Comprehensive gRPC Error Handling Strategy**

**1. Service SLAs and Characteristics**:

| **Service** | **p99 Latency** | **Failure Rate** | **Failure Type** |
|-------------|-----------------|------------------|------------------|
| Inventory   | 50ms | 0.1% | DB overload (transient) |
| Payment     | 200ms | 0.5% | External API timeout |
| Shipping    | 100ms | 0.2% | Rate limiting |

**2. Timeout Configuration**:

\`\`\`javascript
const TIMEOUTS = {
  inventory: {
    deadline: 200,    // 4x p99 (50ms × 4)
    retryDeadline: 500 // Total time including retries
  },
  payment: {
    deadline: 800,    // 4x p99 (200ms × 4)
    retryDeadline: 2000
  },
  shipping: {
    deadline: 400,    // 4x p99 (100ms × 4)
    retryDeadline: 1000
  }
};
\`\`\`

**Rationale**: Set deadline to 4x p99 latency to allow for occasional slowness but prevent hanging.

**3. Retry Policy by Status Code**:

\`\`\`javascript
const RETRY_POLICY = {
  // Retry with exponential backoff
  [grpc.status.UNAVAILABLE]: {
    maxRetries: 3,
    backoff: 'exponential',
    initialDelay: 100,
    maxDelay: 1000
  },
  [grpc.status.DEADLINE_EXCEEDED]: {
    maxRetries: 2,
    backoff: 'exponential',
    initialDelay: 200,
    maxDelay: 1000
  },
  [grpc.status.RESOURCE_EXHAUSTED]: {
    maxRetries: 3,
    backoff: 'exponential',
    initialDelay: 500, // Longer initial delay
    maxDelay: 5000
  },
  
  // Don't retry
  [grpc.status.INVALID_ARGUMENT]: { maxRetries: 0 },
  [grpc.status.NOT_FOUND]: { maxRetries: 0 },
  [grpc.status.ALREADY_EXISTS]: { maxRetries: 0 },
  [grpc.status.PERMISSION_DENIED]: { maxRetries: 0 },
  [grpc.status.UNAUTHENTICATED]: { maxRetries: 0 },
  [grpc.status.FAILED_PRECONDITION]: { maxRetries: 0 }
};

async function callWithRetry(client, method, request, config) {
  const policy = RETRY_POLICY[config.statusCode] || { maxRetries: 0 };
  let attempts = 0;
  
  while (attempts <= policy.maxRetries) {
    try {
      const deadline = Date.now() + config.deadline;
      const result = await promisify(client[method])(request, { deadline });
      return result;
    } catch (error) {
      attempts++;
      
      // Check if error is retryable
      const retryConfig = RETRY_POLICY[error.code];
      if (!retryConfig || attempts > retryConfig.maxRetries) {
        throw error;
      }
      
      // Calculate backoff delay
      const delay = Math.min(
        retryConfig.initialDelay * Math.pow(2, attempts - 1),
        retryConfig.maxDelay
      );
      
      logger.warn(\`Retry \${attempts}/\${retryConfig.maxRetries} for \${method} after \${delay}ms\`, {
        error: error.code,
        message: error.message
      });
      
      await sleep(delay);
    }
  }
}
\`\`\`

**4. Circuit Breaker Pattern**:

\`\`\`javascript
class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 60s
    this.monitoringWindow = options.monitoringWindow || 10000; // 10s
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failures = [];
    this.nextAttempt = null;
  }
  
  async execute(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      // Try one request (HALF_OPEN state)
      this.state = 'HALF_OPEN';
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
      this.failures = [];
    }
  }
  
  onFailure() {
    const now = Date.now();
    this.failures.push(now);
    
    // Remove old failures outside monitoring window
    this.failures = this.failures.filter(
      time => now - time < this.monitoringWindow
    );
    
    if (this.failures.length >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = now + this.resetTimeout;
      logger.error(\`Circuit breaker opened. Next attempt at \${new Date(this.nextAttempt)}\`);
    }
  }
}

// Create circuit breakers for each service
const circuitBreakers = {
  inventory: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 30000 }),
  payment: new CircuitBreaker({ failureThreshold: 3, resetTimeout: 60000 }),
  shipping: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 30000 })
};
\`\`\`

**5. Handling Partial Failures in Order Service**:

\`\`\`javascript
async function createOrder(call, callback) {
  const { user_id, items, payment_info, shipping_address } = call.request;
  const orderId = generateOrderId();
  
  // Store rollback actions
  const rollbackActions = [];
  
  try {
    // Step 1: Check inventory (critical)
    const inventoryResult = await circuitBreakers.inventory.execute(() =>
      callWithRetry(inventoryClient, 'checkAvailability', { items }, {
        deadline: TIMEOUTS.inventory.deadline,
        statusCode: grpc.status.UNAVAILABLE
      })
    );
    
    if (!inventoryResult.available) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Items not in stock'
      });
    }
    
    // Step 2: Reserve inventory (critical, must rollback)
    await circuitBreakers.inventory.execute(() =>
      callWithRetry(inventoryClient, 'reserveItems', { items, orderId }, {
        deadline: TIMEOUTS.inventory.deadline
      })
    );
    rollbackActions.push(() => 
      inventoryClient.releaseItems({ orderId })
    );
    
    // Step 3: Process payment (critical, must rollback)
    let paymentResult;
    try {
      paymentResult = await circuitBreakers.payment.execute(() =>
        callWithRetry(paymentClient, 'processPayment', {
          amount: inventoryResult.total,
          payment_info,
          idempotency_key: orderId
        }, {
          deadline: TIMEOUTS.payment.deadline
        })
      );
    } catch (error) {
      // Payment failed, rollback inventory
      await executeRollback(rollbackActions);
      
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Payment failed',
        details: error.message
      });
    }
    
    rollbackActions.push(() =>
      paymentClient.refundPayment({ transaction_id: paymentResult.transactionId })
    );
    
    // Step 4: Create shipping label (optional, can retry later)
    let shippingResult;
    try {
      shippingResult = await circuitBreakers.shipping.execute(() =>
        callWithRetry(shippingClient, 'createLabel', {
          orderId,
          address: shipping_address,
          items
        }, {
          deadline: TIMEOUTS.shipping.deadline
        })
      );
    } catch (error) {
      // Shipping failed, but order can still be created
      // Queue for retry via async job
      logger.warn('Shipping label creation failed, queuing for retry', {
        orderId,
        error: error.message
      });
      await queueShippingRetry(orderId, shipping_address, items);
      shippingResult = { status: 'pending' };
    }
    
    // Step 5: Create order record
    const order = {
      order_id: orderId,
      user_id,
      items,
      status: 'confirmed',
      total: inventoryResult.total,
      payment_transaction_id: paymentResult.transactionId,
      shipping_status: shippingResult.status
    };
    
    await saveOrder(order);
    
    // Success!
    callback(null, order);
    
  } catch (error) {
    // Unexpected error, rollback everything
    logger.error('Order creation failed, rolling back', {
      orderId,
      error: error.message,
      stack: error.stack
    });
    
    await executeRollback(rollbackActions);
    
    // Determine appropriate error code
    if (error.code === grpc.status.UNAVAILABLE) {
      return callback({
        code: grpc.status.UNAVAILABLE,
        message: 'Service temporarily unavailable, please try again'
      });
    } else if (error.code === grpc.status.DEADLINE_EXCEEDED) {
      return callback({
        code: grpc.status.DEADLINE_EXCEEDED,
        message: 'Request timeout, please try again'
      });
    } else {
      return callback({
        code: grpc.status.INTERNAL,
        message: 'Internal error, please contact support',
        details: error.message
      });
    }
  }
}

async function executeRollback(actions) {
  for (const action of actions.reverse()) {
    try {
      await action();
    } catch (error) {
      // Log but don't throw (best-effort rollback)
      logger.error('Rollback action failed', { error: error.message });
    }
  }
}
\`\`\`

**6. Monitoring and Alerting**:

\`\`\`javascript
// Metrics to track
const metrics = {
  requestDuration: new Histogram('grpc_request_duration_seconds'),
  requestTotal: new Counter('grpc_requests_total'),
  circuitBreakerState: new Gauge('circuit_breaker_state'),
  retryTotal: new Counter('grpc_retries_total')
};

// Alert thresholds
const ALERTS = {
  errorRate: 0.01,        // 1% error rate
  p99Latency: 1000,       // 1 second p99
  circuitBreakerOpen: 1   // Any circuit breaker open
};
\`\`\`

**7. Idempotency**:

For payment and other critical operations, use idempotency keys:

\`\`\`javascript
// Client sends same idempotency key on retry
await paymentClient.processPayment({
  amount: 100,
  payment_info: {...},
  idempotency_key: orderId  // Same for all retries
});

// Server deduplicates by idempotency key
async function processPayment(call, callback) {
  const { idempotency_key } = call.request;
  
  // Check if already processed
  const existing = await getTransactionByIdempotencyKey(idempotency_key);
  if (existing) {
    return callback(null, existing); // Return cached result
  }
  
  // Process payment...
}
\`\`\`

**Key Takeaways**:

1. **Set timeouts to 4x p99 latency** to balance responsiveness and reliability
2. **Retry transient errors** (UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED)
3. **Never retry logical errors** (INVALID_ARGUMENT, NOT_FOUND, PERMISSION_DENIED)
4. **Use exponential backoff** to avoid thundering herd
5. **Circuit breakers prevent cascading failures** by failing fast when service is down
6. **Rollback on critical failures** (payment, inventory) but allow optional failures (shipping)
7. **Idempotency prevents duplicate operations** on retry
8. **Monitor circuit breaker state**, error rates, latency percentiles
9. **Partial failures**: Critical ops must succeed, optional ops can be queued for retry
10. **Always log** errors with context for debugging`,
          keyPoints: [
            'Set appropriate timeouts based on service SLAs (50ms-200ms)',
            'Implement retry with exponential backoff for transient errors',
            'Use circuit breakers to prevent cascading failures',
            'Handle partial failures: roll back critical ops, queue optional ones',
            'Use idempotency keys to prevent duplicate operations on retry',
          ],
        },
        {
          id: 'rpc-public-api-gateway',
          question:
            "You need to design an API Gateway that exposes your internal gRPC microservices to external clients (web browsers, mobile apps, third-party integrators). The gateway must support REST, GraphQL, and potentially gRPC-Web. Design the architecture, explain the trade-offs of each protocol option, describe how you'd handle authentication/authorization, rate limiting, and provide a specific implementation approach. Include diagrams or pseudo-code as needed.",
          sampleAnswer: `**API Gateway Architecture for gRPC Microservices**

**High-Level Architecture**:

\`\`\`
External Clients
      |
[Web Browser]  [Mobile App]  [3rd Party]
      |              |              |
   REST API     GraphQL API    gRPC-Web
      |              |              |
      +----------- API Gateway ------------+
                     |
         +-----------+-----------+
         |           |           |
    [REST] [GraphQL] [gRPC-Web] [Admin]
         |           |           |
      Transcoder  Resolver   gRPC-Web Proxy
         |           |           |
         +--------- gRPC ---------+
                     |
         +-----------+-----------+
         |           |           |
    User Service  Order Service  Inventory Service
     (gRPC)        (gRPC)         (gRPC)
\`\`\`

**1. Protocol Support and Trade-offs**:

| **Protocol** | **Use Case** | **Pros** | **Cons** |
|--------------|--------------|----------|----------|
| **REST** | Public API, 3rd parties | Familiar, cacheable, curl-testable | Slower, no streaming, over/under-fetching |
| **GraphQL** | Mobile apps, dashboards | Single endpoint, flexible queries, efficient | Complex caching, N+1 queries, learning curve |
| **gRPC-Web** | Internal web apps | Fast, streaming, type-safe | Requires grpc-web proxy, limited browser support |

**Recommendation**:
- **REST**: Default for public API (widest compatibility)
- **GraphQL**: For mobile/web apps (efficient, flexible)
- **gRPC-Web**: For internal dashboards (performance)

**2. API Gateway Implementation (Node.js + Envoy)**:

**Architecture Components**:
- **Envoy Proxy**: gRPC-Web proxy, load balancing, TLS termination
- **Node.js API Gateway**: REST/GraphQL → gRPC transcoding, auth, rate limiting
- **Redis**: Rate limiting, session storage
- **Auth Service**: JWT validation, OAuth

**Envoy Configuration (envoy.yaml)**:

\`\`\`yaml
static_resources:
  listeners:
    - name: main_listener
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 8080
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                codec_type: AUTO
                stat_prefix: ingress_http
                
                # gRPC-Web support
                http_filters:
                  - name: envoy.filters.http.grpc_web
                  - name: envoy.filters.http.cors
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.cors.v3.Cors
                  - name: envoy.filters.http.router
                
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: backend
                      domains: ["*"]
                      routes:
                        # gRPC-Web routes
                        - match:
                            prefix: "/grpc"
                          route:
                            cluster: grpc_services
                            timeout: 30s
                        # REST routes (proxy to Node.js)
                        - match:
                            prefix: "/api"
                          route:
                            cluster: api_gateway
                        # GraphQL routes
                        - match:
                            prefix: "/graphql"
                          route:
                            cluster: api_gateway
  
  clusters:
    - name: grpc_services
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      http2_protocol_options: {}
      load_assignment:
        cluster_name: grpc_services
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: user-service
                      port_value: 50051
    
    - name: api_gateway
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      load_assignment:
        cluster_name: api_gateway
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: node-gateway
                      port_value: 3000
\`\`\`

**3. Node.js API Gateway Implementation**:

**REST → gRPC Transcoding**:

\`\`\`javascript
const express = require('express');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const redis = require('redis');
const jwt = require('jsonwebtoken');

const app = express();
app.use(express.json());

// Load gRPC clients
const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;
const userClient = new userProto.UserService(
  'user-service:50051',
  grpc.credentials.createInsecure()
);

const redisClient = redis.createClient({ url: 'redis://redis:6379' });
await redisClient.connect();

// Middleware: Authentication
async function authenticate(req, res, next) {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid token' });
  }
  
  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

// Middleware: Rate Limiting (Token Bucket)
async function rateLimit(req, res, next) {
  const key = \`rate_limit:\${req.user.id}\`;
  const limit = 100; // requests per minute
  const window = 60; // seconds
  
  const current = await redisClient.incr(key);
  
  if (current === 1) {
    await redisClient.expire(key, window);
  }
  
  if (current > limit) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: await redisClient.ttl(key)
    });
  }
  
  res.setHeader('X-RateLimit-Limit', limit);
  res.setHeader('X-RateLimit-Remaining', limit - current);
  
  next();
}

// REST Endpoints

// GET /api/users/:id → userService.GetUser
app.get('/api/users/:id', authenticate, rateLimit, (req, res) => {
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  metadata.add('request-id', req.id);
  
  userClient.getUser(
    { id: req.params.id },
    metadata,
    { deadline: Date.now() + 5000 },
    (error, user) => {
      if (error) {
        if (error.code === grpc.status.NOT_FOUND) {
          return res.status(404).json({ error: 'User not found' });
        } else if (error.code === grpc.status.PERMISSION_DENIED) {
          return res.status(403).json({ error: 'Permission denied' });
        } else {
          logger.error('gRPC error', { error });
          return res.status(500).json({ error: 'Internal server error' });
        }
      }
      
      res.json(user);
    }
  );
});

// POST /api/users → userService.CreateUser
app.post('/api/users', authenticate, rateLimit, (req, res) => {
  const { name, email, age } = req.body;
  
  // Validation
  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email required' });
  }
  
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  
  userClient.createUser(
    { name, email, age },
    metadata,
    { deadline: Date.now() + 5000 },
    (error, user) => {
      if (error) {
        if (error.code === grpc.status.ALREADY_EXISTS) {
          return res.status(409).json({ error: 'User already exists' });
        } else if (error.code === grpc.status.INVALID_ARGUMENT) {
          return res.status(400).json({ error: error.message });
        } else {
          logger.error('gRPC error', { error });
          return res.status(500).json({ error: 'Internal server error' });
        }
      }
      
      res.status(201).json(user);
    }
  );
});

// GET /api/users → userService.ListUsers (server streaming)
app.get('/api/users', authenticate, rateLimit, (req, res) => {
  const { page_size = 20, page_token } = req.query;
  
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  
  const call = userClient.listUsers({ page_size, page_token }, metadata);
  
  const users = [];
  
  call.on('data', (user) => {
    users.push(user);
  });
  
  call.on('end', () => {
    res.json({ users, next_page_token: null }); // Simplified
  });
  
  call.on('error', (error) => {
    logger.error('gRPC streaming error', { error });
    if (!res.headersSent) {
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

app.listen(3000, () => {
  console.log('API Gateway listening on port 3000');
});
\`\`\`

**4. GraphQL Integration**:

\`\`\`javascript
const { ApolloServer, gql } = require('apollo-server-express');

// GraphQL Schema
const typeDefs = gql\`
  type User {
    id: ID!
    name: String!
    email: String!
    age: Int
  }
  
  type Query {
    user(id: ID!): User
    users(limit: Int = 20): [User!]!
  }
  
  type Mutation {
    createUser(name: String!, email: String!, age: Int): User!
  }
\`;

// Resolvers
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        userClient.getUser({ id }, metadata, (error, user) => {
          if (error) reject(error);
          else resolve(user);
        });
      });
    },
    
    users: async (_, { limit }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        const call = userClient.listUsers({ page_size: limit }, metadata);
        const users = [];
        
        call.on('data', (user) => users.push(user));
        call.on('end', () => resolve(users));
        call.on('error', (error) => reject(error));
      });
    }
  },
  
  Mutation: {
    createUser: async (_, { name, email, age }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        userClient.createUser(
          { name, email, age },
          metadata,
          (error, user) => {
            if (error) reject(error);
            else resolve(user);
          }
        );
      });
    }
  }
};

// Create Apollo Server
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: async ({ req }) => {
    // Extract user from JWT
    const token = req.headers.authorization?.substring(7);
    if (!token) throw new Error('Unauthorized');
    
    const user = jwt.verify(token, process.env.JWT_SECRET);
    return { user: { ...user, token } };
  }
});

server.applyMiddleware({ app, path: '/graphql' });
\`\`\`

**5. Authentication Flow**:

\`\`\`
1. Client sends: Authorization: Bearer <JWT>
2. API Gateway validates JWT
3. Gateway extracts user_id, roles from JWT
4. Gateway adds metadata to gRPC call:
   - authorization: Bearer <JWT>
   - user-id: 123
   - roles: admin,user
5. Microservice validates metadata (optional, for defense in depth)
6. Microservice performs authorization check
7. Response flows back through gateway
\`\`\`

**6. Key Takeaways**:

1. **Use Envoy for gRPC-Web** proxy and load balancing
2. **REST for public APIs**, GraphQL for mobile/web, gRPC-Web for internal
3. **Authenticate at gateway**, propagate user context via metadata
4. **Rate limit per user** using Redis (token bucket or sliding window)
5. **Map gRPC status codes to HTTP** (NOT_FOUND → 404, PERMISSION_DENIED → 403)
6. **Handle streaming** carefully in REST (buffer or SSE)
7. **Timeout all gRPC calls** to prevent hanging
8. **Monitor**: latency, error rate, rate limit hits
9. **Cache** REST responses at CDN/gateway when possible
10. **Defense in depth**: Validate auth at both gateway and service level`,
          keyPoints: [
            'Support multiple protocols: REST (public), GraphQL (mobile), gRPC-Web (internal)',
            'Use Envoy for gRPC-Web proxy, TLS termination, and load balancing',
            'Node.js API Gateway for REST/GraphQL → gRPC transcoding',
            'Authentication via JWT validation, propagate to microservices via gRPC metadata',
            'Rate limiting with Redis (token bucket or sliding window)',
            'Protocol trade-offs: REST (familiar), GraphQL (flexible), gRPC-Web (fast)',
            'Error mapping: gRPC status codes → HTTP status codes (NOT_FOUND → 404)',
            'Monitor latency, error rates, and rate limit hits',
          ],
        },
      ],
    },
    {
      id: 'graphql',
      title: 'GraphQL',
      content: `GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need. Understanding GraphQL is essential for modern API design, especially for mobile and web applications.
    
    ## What is GraphQL?
    
    **GraphQL** is a query language developed by Facebook that allows clients to specify exactly what data they need in a single request.
    
    **Key Principles**:
    - **Client-Specified Queries**: Client defines the shape of the response
    - **Single Endpoint**: All queries go to one endpoint (typically \`/graphql\`)
    - **Strong Typing**: Schema defines types and relationships
    - **Hierarchical**: Queries match the shape of the data
    
    **Comparison**:
    
    | **Aspect** | **REST** | **GraphQL** |
    |------------|----------|-------------|
    | **Endpoints** | Multiple (\`/users\`, \`/posts\`) | Single (\`/graphql\`) |
    | **Data Fetching** | Fixed response structure | Client specifies fields |
    | **Over-fetching** | Common (get all fields) | None (request only what you need) |
    | **Under-fetching** | Requires multiple requests | Single request gets all data |
    | **Versioning** | URL versioning (\`/v1/users\`) | Schema evolution (no breaking changes) |
    | **Caching** | Easy (HTTP caching) | Complex (requires work) |
    | **Learning Curve** | Low | Medium |
    
    ---
    
    ## GraphQL Schema
    
    **Schema defines**:
    - Types (objects, scalars, enums)
    - Queries (read operations)
    - Mutations (write operations)
    - Subscriptions (real-time updates)
    
    **Example Schema**:
    \`\`\`graphql
    # user.graphql
    type User {
      id: ID!
      name: String!
      email: String!
      age: Int
      posts: [Post!]!
      friends: [User!]!
    }
    
    type Post {
      id: ID!
      title: String!
      content: String!
      author: User!
      comments: [Comment!]!
      createdAt: DateTime!
    }
    
    type Comment {
      id: ID!
      text: String!
      author: User!
      post: Post!
    }
    
    type Query {
      # Get single user
      user(id: ID!): User
      
      # Get all users with pagination
      users(limit: Int = 20, offset: Int = 0): [User!]!
      
      # Get posts by user
      posts(userId: ID!): [Post!]!
      
      # Search users
      searchUsers(query: String!): [User!]!
    }
    
    type Mutation {
      # Create user
      createUser(name: String!, email: String!, age: Int): User!
      
      # Update user
      updateUser(id: ID!, name: String, email: String, age: Int): User!
      
      # Delete user
      deleteUser(id: ID!): Boolean!
      
      # Create post
      createPost(userId: ID!, title: String!, content: String!): Post!
      
      # Add comment
      addComment(postId: ID!, userId: ID!, text: String!): Comment!
    }
    
    type Subscription {
      # Subscribe to new posts
      newPost: Post!
      
      # Subscribe to comments on a post
      newComment(postId: ID!): Comment!
    }
    
    # Custom scalar for dates
    scalar DateTime
    \`\`\`
    
    ---
    
    ## GraphQL Queries
    
    ### **Basic Query**:
    
    \`\`\`graphql
    query {
      user(id: "123") {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Response**:
    \`\`\`json
    {
      "data": {
        "user": {
          "id": "123",
          "name": "John Doe",
          "email": "john@example.com"
        }
      }
    }
    \`\`\`
    
    ### **Nested Query**:
    
    \`\`\`graphql
    query {
      user(id: "123") {
        id
        name
        posts {
          id
          title
          comments {
            id
            text
            author {
              name
            }
          }
        }
      }
    }
    \`\`\`
    
    **Why This is Powerful**:
    - Single request gets user, their posts, and all comments with author names
    - With REST, would require 3+ requests: \`/users/123\`, \`/posts?userId=123\`, \`/comments?postId=X\` (for each post)
    
    ### **Query with Variables**:
    
    \`\`\`graphql
    query GetUser($userId: ID!) {
      user(id: $userId) {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Variables**:
    \`\`\`json
    {
      "userId": "123"
    }
    \`\`\`
    
    ### **Query with Aliases**:
    
    \`\`\`graphql
    query {
      user1: user(id: "123") {
        name
      }
      user2: user(id: "456") {
        name
      }
    }
    \`\`\`
    
    **Response**:
    \`\`\`json
    {
      "data": {
        "user1": { "name": "Alice" },
        "user2": { "name": "Bob" }
      }
    }
    \`\`\`
    
    ### **Query with Fragments**:
    
    \`\`\`graphql
    fragment UserFields on User {
      id
      name
      email
    }
    
    query {
      user1: user(id: "123") {
        ...UserFields
      }
      user2: user(id: "456") {
        ...UserFields
      }
    }
    \`\`\`
    
    ---
    
    ## GraphQL Mutations
    
    **Create User**:
    \`\`\`graphql
    mutation {
      createUser(name: "Jane Doe", email: "jane@example.com", age: 28) {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Update User**:
    \`\`\`graphql
    mutation {
      updateUser(id: "123", name: "John Smith") {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Create Post with Variables**:
    \`\`\`graphql
    mutation CreatePost($userId: ID!, $title: String!, $content: String!) {
      createPost(userId: $userId, title: $title, content: $content) {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    ---
    
    ## GraphQL Subscriptions
    
    **Subscriptions enable real-time updates via WebSocket**.
    
    **Client subscribes**:
    \`\`\`graphql
    subscription {
      newPost {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    **Server pushes updates**:
    \`\`\`json
    {
      "data": {
        "newPost": {
          "id": "789",
          "title": "Breaking News",
          "author": {
            "name": "Alice"
          }
        }
      }
    }
    \`\`\`
    
    ---
    
    ## Implementing GraphQL Server (Node.js + Apollo)
    
    \`\`\`javascript
    const { ApolloServer, gql } = require('apollo-server');
    
    // Define schema
    const typeDefs = gql\`
      type User {
        id: ID!
        name: String!
        email: String!
        posts: [Post!]!
      }
      
      type Post {
        id: ID!
        title: String!
        content: String!
        author: User!
      }
      
      type Query {
        user(id: ID!): User
        users: [User!]!
      }
      
      type Mutation {
        createUser(name: String!, email: String!): User!
      }
    \`;
    
    // Define resolvers
    const resolvers = {
      Query: {
        user: async (parent, { id }, context) => {
          return await context.db.users.findById(id);
        },
        
        users: async (parent, args, context) => {
          return await context.db.users.findAll();
        }
      },
      
      Mutation: {
        createUser: async (parent, { name, email }, context) => {
          const user = await context.db.users.create({ name, email });
          return user;
        }
      },
      
      // Field resolvers
      User: {
        posts: async (parent, args, context) => {
          // parent is the User object
          return await context.db.posts.findByUserId(parent.id);
        }
      },
      
      Post: {
        author: async (parent, args, context) => {
          // parent is the Post object
          return await context.db.users.findById(parent.userId);
        }
      }
    };
    
    // Create server
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      context: ({ req }) => ({
        db: database, // Pass database connection
        user: req.user // Pass authenticated user
      })
    });
    
    server.listen().then(({ url }) => {
      console.log(\`🚀 Server ready at \${url}\`);
    });
    \`\`\`
    
    ---
    
    ## The N+1 Query Problem
    
    **One of the biggest pitfalls in GraphQL**.
    
    **Scenario**:
    \`\`\`graphql
    query {
      posts {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    **Naive Implementation**:
    \`\`\`javascript
    const resolvers = {
      Query: {
        posts: () => db.posts.findAll() // 1 query
      },
      Post: {
        author: (post) => db.users.findById(post.userId) // N queries!
      }
    };
    \`\`\`
    
    **Problem**: If there are 100 posts, this makes 101 database queries (1 for posts + 100 for authors).
    
    **Solution: DataLoader (Batching)**:
    
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // Batch load users
    const userLoader = new DataLoader(async (userIds) => {
      const users = await db.users.findByIds(userIds);
      // Return users in same order as userIds
      return userIds.map(id => users.find(user => user.id === id));
    });
    
    const resolvers = {
      Query: {
        posts: () => db.posts.findAll() // 1 query
      },
      Post: {
        author: (post, args, { loaders }) => {
          return loaders.user.load(post.userId); // Batched!
        }
      }
    };
    
    // Context setup
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      context: () => ({
        loaders: {
          user: new DataLoader(batchLoadUsers)
        }
      })
    });
    \`\`\`
    
    **Result**: Now makes only 2 queries (1 for posts + 1 batched query for all authors).
    
    ---
    
    ## GraphQL Caching
    
    **Challenge**: GraphQL uses POST requests to \`/graphql\`, which are not cacheable by HTTP.
    
    **Solutions**:
    
    ### **1. Persisted Queries**:
    
    \`\`\`javascript
    // Client sends query hash instead of full query
    POST /graphql
    {
      "extensions": {
        "persistedQuery": {
          "version": 1,
          "sha256Hash": "abc123..."
        }
      }
    }
    
    // Server looks up query by hash
    const query = queryRegistry.get("abc123...");
    \`\`\`
    
    **Benefits**:
    - Smaller request size
    - Enables GET requests (cacheable)
    - Security (only allowed queries can execute)
    
    ### **2. Response Caching**:
    
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      plugins: [responseCachePlugin()],
      cacheControl: {
        defaultMaxAge: 300 // 5 minutes
      }
    });
    \`\`\`
    
    **Cache hints in schema**:
    \`\`\`graphql
    type User @cacheControl(maxAge: 300) {
      id: ID!
      name: String!
      email: String!
    }
    
    type Post @cacheControl(maxAge: 60) {
      id: ID!
      title: String!
    }
    \`\`\`
    
    ### **3. Client-Side Caching (Apollo Client)**:
    
    \`\`\`javascript
    import { ApolloClient, InMemoryCache } from '@apollo/client';
    
    const client = new ApolloClient({
      uri: 'http://localhost:4000/graphql',
      cache: new InMemoryCache({
        typePolicies: {
          Query: {
            fields: {
              user: {
                read(existing, { args, toReference }) {
                  // Return cached user if exists
                  return existing || toReference({
                    __typename: 'User',
                    id: args.id
                  });
                }
              }
            }
          }
        }
      })
    });
    \`\`\`
    
    ---
    
    ## GraphQL vs REST vs gRPC
    
    | **Feature** | **REST** | **GraphQL** | **gRPC** |
    |-------------|----------|-------------|----------|
    | **Data Fetching** | Over/under-fetching | Exact data needed | Fixed by proto definition |
    | **Endpoints** | Many | One | Service methods |
    | **Type Safety** | Weak (OpenAPI helps) | Strong (schema) | Very strong (protobuf) |
    | **Performance** | Good | Good | Excellent (binary) |
    | **Caching** | Easy (HTTP) | Complex | Difficult |
    | **Real-time** | SSE/WebSocket | Subscriptions (WebSocket) | Streaming |
    | **Learning Curve** | Low | Medium | High |
    | **Mobile-Friendly** | No (over-fetching) | Yes (efficient) | Yes (efficient) |
    | **Public API** | Excellent | Good | Poor (browser support) |
    | **Internal Services** | Good | Overkill | Excellent |
    
    ---
    
    ## When to Use GraphQL
    
    ### **✅ Use GraphQL When:**
    
    1. **Mobile Applications**
       - Bandwidth is limited
       - Need to minimize data transfer
       - Multiple resources needed per screen
    
    2. **Complex UIs with Many Relationships**
       - Dashboard with data from many sources
       - Social network (users, posts, comments, likes)
       - E-commerce (products, reviews, recommendations)
    
    3. **Rapid Frontend Development**
       - Frontend team can iterate without backend changes
       - No need for new endpoints for each view
       - GraphQL Playground for testing
    
    4. **Multiple Clients with Different Needs**
       - Web app needs different data than mobile app
       - Each client requests only what it needs
       - No need for multiple API versions
    
    5. **Real-Time Updates**
       - Subscriptions for live data
       - Chat applications
       - Live sports scores
    
    ### **❌ Avoid GraphQL When:**
    
    1. **Simple CRUD APIs**
       - REST is simpler and more familiar
       - No complex relationships
       - Standard HTTP caching sufficient
    
    2. **Public APIs for Third Parties**
       - REST more familiar to external developers
       - Better documentation with OpenAPI
       - Easier to rate limit by endpoint
    
    3. **File Uploads/Downloads**
       - GraphQL not designed for binary data
       - Multipart uploads awkward in GraphQL
       - REST simpler for files
    
    4. **Team Lacks GraphQL Experience**
       - Learning curve can slow development
       - Requires understanding of N+1 problem, DataLoader, caching
       - REST more familiar
    
    ---
    
    ## Common GraphQL Mistakes
    
    ### **❌ Mistake 1: Not Using DataLoader (N+1 Problem)**
    
    \`\`\`javascript
    // Bad: N+1 queries
    const resolvers = {
      Post: {
        author: (post) => db.users.findById(post.userId)
      }
    };
    
    // Good: Batch with DataLoader
    const resolvers = {
      Post: {
        author: (post, args, { loaders }) => loaders.user.load(post.userId)
      }
    };
    \`\`\`
    
    ### **❌ Mistake 2: No Pagination**
    
    \`\`\`javascript
    // Bad: Return all users
    type Query {
      users: [User!]!
    }
    
    // Good: Paginate with cursor or offset
    type Query {
      users(first: Int, after: String): UserConnection!
    }
    
    type UserConnection {
      edges: [UserEdge!]!
      pageInfo: PageInfo!
    }
    
    type UserEdge {
      node: User!
      cursor: String!
    }
    
    type PageInfo {
      hasNextPage: Boolean!
      endCursor: String
    }
    \`\`\`
    
    ### **❌ Mistake 3: No Query Depth Limiting**
    
    \`\`\`javascript
    // Malicious query (infinite loop)
    query {
      user(id: "1") {
        friends {
          friends {
            friends {
              friends {
                friends {
                  # ... ad infinitum
                }
              }
            }
          }
        }
      }
    }
    
    // Solution: Limit query depth
    const depthLimit = require('graphql-depth-limit');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [depthLimit(5)] // Max depth 5
    });
    \`\`\`
    
    ### **❌ Mistake 4: No Query Cost Analysis**
    
    \`\`\`javascript
    // Expensive query
    query {
      users(first: 1000) {
        posts(first: 1000) {
          comments(first: 1000) {
            # 1 billion operations!
          }
        }
      }
    }
    
    // Solution: Query cost analysis
    const { createComplexityLimitRule } = require('graphql-validation-complexity');
    
    const server = new ApolloServer({
      validationRules: [
        createComplexityLimitRule(1000) // Max complexity 1000
      ]
    });
    \`\`\`
    
    ### **❌ Mistake 5: Exposing Internal Implementation**
    
    \`\`\`javascript
    // Bad: Database structure leaked to API
    type User {
      id: ID!
      user_name: String!  # snake_case from database
      created_at: String! # database column name
    }
    
    // Good: API-friendly names
    type User {
      id: ID!
      name: String!      # camelCase
      createdAt: DateTime! # semantic name + custom scalar
    }
    \`\`\`
    
    ---
    
    ## Real-World Example: Social Media API
    
    **Schema**:
    \`\`\`graphql
    type User {
      id: ID!
      username: String!
      email: String!
      posts(first: Int, after: String): PostConnection!
      followers: [User!]!
      following: [User!]!
      followerCount: Int!
      followingCount: Int!
    }
    
    type Post {
      id: ID!
      content: String!
      imageUrl: String
      author: User!
      likes: [Like!]!
      comments(first: Int): [Comment!]!
      likeCount: Int!
      commentCount: Int!
      createdAt: DateTime!
    }
    
    type Comment {
      id: ID!
      text: String!
      author: User!
      post: Post!
      createdAt: DateTime!
    }
    
    type Like {
      user: User!
      post: Post!
      createdAt: DateTime!
    }
    
    type Query {
      me: User
      user(username: String!): User
      post(id: ID!): Post
      feed(first: Int, after: String): PostConnection!
    }
    
    type Mutation {
      createPost(content: String!, imageUrl: String): Post!
      likePost(postId: ID!): Post!
      unlikePost(postId: ID!): Post!
      addComment(postId: ID!, text: String!): Comment!
      followUser(userId: ID!): User!
      unfollowUser(userId: ID!): User!
    }
    
    type Subscription {
      newPost(userId: ID!): Post!
      newComment(postId: ID!): Comment!
    }
    \`\`\`
    
    **Query Examples**:
    
    \`\`\`graphql
    # Get user feed
    query GetFeed {
      feed(first: 20) {
        edges {
          node {
            id
            content
            imageUrl
            author {
              username
            }
            likeCount
            commentCount
            comments(first: 3) {
              text
              author {
                username
              }
            }
          }
          cursor
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    \`\`\`
    
    **Benefits**:
    - Single request gets posts, authors, comments, like counts
    - With REST: would need \`/feed\`, \`/users/:id\`, \`/posts/:id/comments\`, \`/posts/:id/likes\`
    - Mobile app saves bandwidth
    - Frontend can iterate without backend changes
    
    ---
    
    ## Key Takeaways
    
    1. **GraphQL allows clients to specify exactly what data they need** in a single request
    2. **Solves over-fetching and under-fetching** problems of REST APIs
    3. **N+1 problem is the biggest pitfall** - always use DataLoader for batching
    4. **Caching is complex** - requires persisted queries or response caching
    5. **Strong typing with schema** enables great developer experience
    6. **Subscriptions enable real-time updates** via WebSocket
    7. **Query depth limiting and cost analysis** prevent malicious queries
    8. **Best for mobile apps and complex UIs** with many relationships
    9. **Not a replacement for REST** - choose based on use case
    10. **Learning curve is real** - team must understand resolvers, DataLoader, caching`,
      multipleChoice: [
        {
          id: 'graphql-vs-rest-fetching',
          question:
            'What is the primary advantage of GraphQL over REST in terms of data fetching?',
          options: [
            'GraphQL is faster because it uses binary encoding',
            'GraphQL allows clients to request exactly the data they need, avoiding over-fetching and under-fetching',
            'GraphQL automatically caches responses more efficiently',
            'GraphQL requires fewer servers to operate',
          ],
          correctAnswer: 1,
          explanation:
            "GraphQL's primary advantage is that clients can specify exactly which fields they need in the query, receiving neither more nor less data. REST endpoints return fixed data structures, often resulting in over-fetching (getting unused fields) or under-fetching (needing multiple requests). GraphQL uses JSON (not binary), caching is actually more complex than REST, and server requirements are comparable.",
        },
        {
          id: 'graphql-n-plus-one',
          question:
            'You have a GraphQL query that fetches 100 posts and the author for each post. Without DataLoader, how many database queries would typically be executed?',
          options: [
            '1 query (GraphQL automatically batches)',
            '2 queries (1 for posts, 1 for all authors)',
            '101 queries (1 for posts, 100 for each author)',
            '100 queries (GraphQL optimizes away the posts query)',
          ],
          correctAnswer: 2,
          explanation:
            'This is the classic N+1 problem. Without DataLoader batching, you would execute 1 query to fetch all posts, then 100 separate queries to fetch each author individually. DataLoader solves this by batching the author queries into a single query: SELECT * FROM users WHERE id IN (id1, id2, ..., id100). Always use DataLoader to avoid this performance issue.',
        },
        {
          id: 'graphql-subscriptions',
          question:
            'How do GraphQL subscriptions typically communicate real-time updates to clients?',
          options: [
            'HTTP long polling',
            'Server-Sent Events (SSE)',
            'WebSocket connections',
            'Regular HTTP requests with short intervals',
          ],
          correctAnswer: 2,
          explanation:
            'GraphQL subscriptions typically use WebSocket connections for bidirectional, real-time communication. When a client subscribes, a WebSocket connection is established, and the server pushes updates whenever relevant data changes. While SSE could work for server-to-client updates, WebSocket is the standard for GraphQL subscriptions and is supported by all major GraphQL implementations like Apollo.',
        },
        {
          id: 'graphql-caching-challenge',
          question:
            'Why is caching more difficult in GraphQL compared to REST APIs?',
          options: [
            'GraphQL responses are larger and harder to store',
            'GraphQL uses POST requests to a single endpoint, bypassing standard HTTP caching',
            'GraphQL servers are stateful and cannot be cached',
            'GraphQL responses contain metadata that prevents caching',
          ],
          correctAnswer: 1,
          explanation:
            "GraphQL typically uses POST requests to a single `/graphql` endpoint with the query in the request body. HTTP caching (like CDN caching, browser caching) works primarily with GET requests to different URLs. This makes standard HTTP caching ineffective. Solutions include: persisted queries (converting to GET requests with query hashes), response caching at the GraphQL server level, or client-side caching (like Apollo Client's normalized cache).",
        },
        {
          id: 'graphql-use-case',
          question:
            'Which scenario is the BEST fit for choosing GraphQL over REST?',
          options: [
            'A public API consumed by thousands of third-party developers who need stable, well-documented endpoints',
            'A mobile application that displays complex screens with data from multiple resources and has bandwidth constraints',
            'A simple file storage service focused on uploading and downloading binary files',
            'A high-performance internal microservice handling millions of requests per second',
          ],
          correctAnswer: 1,
          explanation:
            'GraphQL excels for mobile applications because: (1) It minimizes data transfer by letting clients request only needed fields, (2) Complex screens requiring data from multiple resources can be fetched in a single request, (3) Bandwidth is often limited on mobile networks. For public APIs, REST is more familiar; for file storage, REST is simpler; for high-performance internal services, gRPC is typically better than GraphQL.',
        },
      ],
      quiz: [
        {
          id: 'graphql-migration',
          question:
            "Your company has a large REST API with 150+ endpoints serving web and mobile clients. The mobile team complains about slow performance due to multiple round trips and bandwidth waste. You're considering migrating to GraphQL. Design a migration strategy that minimizes risk, explain how you'd handle the N+1 problem, implement caching, and handle authentication. Include specific technical approaches and timeline.",
          sampleAnswer: `**GraphQL Migration Strategy**
    
    **Phase 1: Assessment & Planning** (Week 1-2)
    
    1. **Analyze REST API Usage**:
       - Which endpoints are called most frequently?
       - Which screens make multiple requests?
       - Measure current: response sizes, latency, number of requests per screen
    
    2. **Identify High-Impact Screens**:
       - Mobile app home feed (calls 5-7 endpoints)
       - User profile page (calls 4 endpoints)
       - Product detail page (calls 6 endpoints)
       - **Expected improvement**: 70% reduction in requests, 50% reduction in data transfer
    
    3. **Choose Architecture**:
       - **Hybrid approach**: GraphQL alongside REST (not replacing)
       - GraphQL for mobile and new web features
       - Keep REST for backward compatibility and public API
       - Use GraphQL as BFF (Backend for Frontend)
    
    **Phase 2: Infrastructure Setup** (Week 3-4)
    
    1. **GraphQL Server Setup**:
    \`\`\`javascript
    // Apollo Server with existing REST API as data sources
    const { ApolloServer } = require('apollo-server-express');
    const { RESTDataSource } = require('apollo-datasource-rest');
    
    // Wrap existing REST API
    class UsersAPI extends RESTDataSource {
      constructor() {
        super();
        this.baseURL = 'https://api.example.com/v1/';
      }
    
      async getUser(id) {
        return this.get(\`users/\${id}\`);
      }
    
      async getUserPosts(userId) {
        return this.get(\`posts?userId=\${userId}\`);
      }
    }
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      dataSources: () => ({
        usersAPI: new UsersAPI(),
        postsAPI: new PostsAPI(),
        commentsAPI: new CommentsAPI()
      })
    });
    \`\`\`
    
    2. **Authentication**:
    \`\`\`javascript
    const server = new ApolloServer({
      context: ({ req }) => {
        // Extract JWT from header
        const token = req.headers.authorization?.replace('Bearer ', '');
        
        if (!token) {
          throw new AuthenticationError('Missing auth token');
        }
        
        try {
          const user = jwt.verify(token, SECRET_KEY);
          return { user };
        } catch (error) {
          throw new AuthenticationError('Invalid token');
        }
      }
    });
    
    // Use in resolvers
    const resolvers = {
      Query: {
        me: (parent, args, { user }) => {
          // user available from context
          return getUserById(user.id);
        }
      }
    };
    \`\`\`
    
    **Phase 3: Solve N+1 Problem** (Week 4)
    
    1. **Implement DataLoader**:
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // Batch load users
    const createUserLoader = () => new DataLoader(async (userIds) => {
      const users = await db.users.findByIds(userIds);
      // Return in same order as requested
      const userMap = new Map(users.map(user => [user.id, user]));
      return userIds.map(id => userMap.get(id));
    });
    
    // Batch load posts by user IDs
    const createPostsByUserLoader = () => new DataLoader(async (userIds) => {
      const posts = await db.posts.findByUserIds(userIds);
      // Group by userId
      const postsByUser = userIds.map(userId => 
        posts.filter(post => post.userId === userId)
      );
      return postsByUser;
    });
    
    // Add to context
    const server = new ApolloServer({
      context: ({ req }) => ({
        user: getUserFromToken(req),
        loaders: {
          user: createUserLoader(),
          postsByUser: createPostsByUserLoader()
        }
      })
    });
    
    // Use in resolvers
    const resolvers = {
      Post: {
        author: (post, args, { loaders }) => {
          // Batched! Multiple calls in same request are combined
          return loaders.user.load(post.userId);
        }
      },
      User: {
        posts: (user, args, { loaders }) => {
          return loaders.postsByUser.load(user.id);
        }
      }
    };
    \`\`\`
    
    **Monitoring N+1**:
    \`\`\`javascript
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        {
          requestDidStart() {
            let queryCount = 0;
            return {
              willSendResponse({ context }) {
                console.log(\`Query count: \${queryCount}\`);
                if (queryCount > 10) {
                  console.warn('Potential N+1 problem detected!');
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Phase 4: Implement Caching** (Week 5)
    
    1. **Response Caching**:
    \`\`\`javascript
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      plugins: [
        responseCachePlugin({
          sessionId: (context) => context.user?.id || null
        })
      ],
      cacheControl: {
        defaultMaxAge: 300 // 5 minutes
      }
    });
    
    // Cache hints in schema
    type User @cacheControl(maxAge: 600) {
      id: ID!
      name: String!
    }
    
    type Post @cacheControl(maxAge: 60) {
      id: ID!
      title: String!
    }
    \`\`\`
    
    2. **Persisted Queries** (for mobile):
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    
    const server = new ApolloServer({
      persistedQueries: {
        cache: redis, // Use Redis for query storage
        ttl: 900 // 15 minutes
      }
    });
    
    // Mobile client sends query hash instead of full query
    // First request: sends query + hash
    // Subsequent requests: only hash
    // Benefits: smaller payload, enables CDN caching (GET request)
    \`\`\`
    
    3. **Client-Side Caching** (Apollo Client):
    \`\`\`javascript
    import { ApolloClient, InMemoryCache } from '@apollo/client';
    
    const client = new ApolloClient({
      uri: 'https://api.example.com/graphql',
      cache: new InMemoryCache({
        typePolicies: {
          Query: {
            fields: {
              user: {
                merge(existing, incoming) {
                  return incoming;
                }
              }
            }
          }
        }
      })
    });
    \`\`\`
    
    **Phase 5: Pilot Implementation** (Week 6-8)
    
    1. **Choose Pilot Screen**: Mobile app home feed
       - Currently makes 7 REST requests
       - 500KB total data transfer
       - 2-3 second load time
    
    2. **GraphQL Query**:
    \`\`\`graphql
    query HomeFeed {
      me {
        id
        name
        avatar
        unreadNotifications
      }
      
      feed(first: 20) {
        edges {
          node {
            id
            content
            imageUrl
            author {
              username
              avatar
            }
            likeCount
            commentCount
            likedByMe
            comments(first: 3) {
              text
              author {
                username
              }
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
      
      suggestions {
        id
        username
        avatar
        mutualFriends
      }
    }
    \`\`\`
    
    3. **Measure Results**:
       - Requests: 7 → 1 (86% reduction)
       - Data transfer: 500KB → 180KB (64% reduction)
       - Load time: 2.5s → 0.9s (64% improvement)
       - Mobile bandwidth saved: ~2GB per user per month
    
    **Phase 6: Gradual Rollout** (Week 9-16)
    
    1. **Week 9-10**: User profile page
    2. **Week 11-12**: Product detail page
    3. **Week 13-14**: Search results
    4. **Week 15-16**: Messaging and notifications
    
    **Phase 7: Monitoring & Optimization** (Ongoing)
    
    1. **Metrics to Track**:
       - Query execution time (p50, p95, p99)
       - Resolver execution time
       - DataLoader hit rate
       - Cache hit rate
       - Error rate by query type
       - N+1 query detection
    
    2. **Observability**:
    \`\`\`javascript
    const { ApolloServerPluginLandingPageGraphQLPlayground } = require('apollo-server-core');
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        ApolloServerPluginInlineTrace(),
        {
          requestDidStart() {
            const start = Date.now();
            return {
              willSendResponse({ metrics, errors }) {
                const duration = Date.now() - start;
                
                metricsCollector.record({
                  duration,
                  query: metrics.queryPlanTrace,
                  errors: errors?.length || 0
                });
                
                if (duration > 1000) {
                  logger.warn(\`Slow query: \${duration}ms\`);
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Key Risks & Mitigations**:
    
    | **Risk** | **Mitigation** |
    |----------|----------------|
    | N+1 queries slow down API | Implement DataLoader for all relationships; monitor query counts |
    | Breaking changes to mobile app | Version GraphQL schema; use @deprecated directive; gradual rollout |
    | Caching complexity | Start with simple response caching; add persisted queries later |
    | Team learning curve | Training sessions; pair programming; comprehensive documentation |
    | Performance regression | A/B testing; rollback plan; monitor all metrics |
    
    **Expected Results**:
    - **Mobile app**: 60-70% reduction in data transfer
    - **Load times**: 40-50% improvement
    - **User satisfaction**: Higher (faster app, less lag)
    - **Infrastructure**: 30% reduction in API calls
    
    **Timeline**: 16 weeks total (4 months)
    
    **Final Architecture**:
    \`\`\`
    Mobile/Web Clients
             ↓
        GraphQL API (Apollo Server)
             ↓
       +---> DataLoader (batching)
       |
       +---> Response Cache (Redis)
       |
       +---> Existing REST APIs (data sources)
             ↓
        Database / Microservices
    \`\`\``,
          keyPoints: [
            'Migrate gradually: start with new features, then migrate high-impact endpoints',
            'Use DataLoader to solve N+1 query problem with batching',
            'Implement response caching with Redis for repeated queries',
            'Add query complexity analysis to prevent expensive operations',
            'Monitor query patterns and optimize resolvers based on real usage',
          ],
        },
        {
          id: 'graphql-security',
          question:
            'Design a comprehensive security strategy for a GraphQL API that prevents common attacks like query depth attacks, query cost attacks, and data exposure. Include specific code examples for: query complexity limits, depth limits, rate limiting, field-level authorization, and monitoring for malicious queries.',
          sampleAnswer: `**Comprehensive GraphQL Security Strategy**
    
    **1. Query Depth Limiting**
    
    **Attack Scenario**:
    \`\`\`graphql
    # Malicious query with infinite recursion
    query {
      user(id: "1") {
        friends {
          friends {
            friends {
              friends {
                # ... 1000 levels deep
              }
            }
          }
        }
      }
    }
    \`\`\`
    
    **Defense**:
    \`\`\`javascript
    const depthLimit = require('graphql-depth-limit');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [depthLimit(7)] // Max depth of 7
    });
    
    // Custom implementation
    function depthLimit(maxDepth) {
      return function (context) {
        return {
          Field(node, key, parent, path, ancestors) {
            const depth = ancestors.filter(
              ancestor => ancestor.kind === 'Field'
            ).length;
            
            if (depth > maxDepth) {
              throw new Error(
                \`Query depth limit of \${maxDepth} exceeded (depth: \${depth})\`
              );
            }
          }
        };
      };
    }
    \`\`\`
    
    **2. Query Cost Analysis**
    
    **Attack Scenario**:
    \`\`\`graphql
    # Expensive query (1 billion operations!)
    query {
      users(first: 1000) {       # 1,000 users
        posts(first: 1000) {      # × 1,000 posts each
          comments(first: 1000) { # × 1,000 comments each
            # Total: 1,000 × 1,000 × 1,000 = 1 billion
          }
        }
      }
    }
    \`\`\`
    
    **Defense**:
    \`\`\`javascript
    const { createComplexityLimitRule } = require('graphql-validation-complexity');
    
    // Define costs in schema
    const typeDefs = gql\`
      type Query {
        users(first: Int = 20): [User!]! @cost(complexity: 10, multipliers: ["first"])
        user(id: ID!): User @cost(complexity: 1)
      }
      
      type User {
        id: ID!
        name: String!
        posts(first: Int = 20): [Post!]! @cost(complexity: 5, multipliers: ["first"])
      }
      
      type Post {
        id: ID!
        comments(first: Int = 20): [Comment!]! @cost(complexity: 3, multipliers: ["first"])
      }
    \`;
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [
        createComplexityLimitRule(1000, {
          onCost: (cost) => {
            console.log(\`Query cost: \${cost}\`);
          }
        })
      ]
    });
    
    // Custom cost calculation
    function calculateQueryCost(query, schema) {
      let cost = 0;
      
      visit(query, {
        Field: (node) => {
          const fieldDef = schema.getField(node.name.value);
          const fieldCost = fieldDef?.cost || 1;
          
          // Apply multipliers (e.g., \`first\` argument)
          const multiplier = node.arguments.find(
            arg => arg.name.value === 'first'
          )?.value.value || 1;
          
          cost += fieldCost * multiplier;
        }
      });
      
      return cost;
    }
    
    // Validate cost before execution
    const validationRules = [
      (context) => ({
        Document(node) {
          const cost = calculateQueryCost(node, context.getSchema());
          
          if (cost > 1000) {
            throw new Error(\`Query cost \${cost} exceeds limit of 1000\`);
          }
          
          console.log(\`Query cost: \${cost}\`);
        }
      })
    ];
    \`\`\`
    
    **3. Rate Limiting**
    
    **Implementation**:
    \`\`\`javascript
    const redis = require('redis');
    const redisClient = redis.createClient();
    
    // Per-user rate limiting
    async function rateLimitPlugin() {
      return {
        requestDidStart: async ({ context }) => {
          const userId = context.user?.id || context.ip;
          const key = \`rate_limit:\${userId}\`;
          
          const requests = await redisClient.incr(key);
          
          if (requests === 1) {
            await redisClient.expire(key, 60); // 1 minute window
          }
          
          if (requests > 100) {
            throw new Error('Rate limit exceeded: 100 requests per minute');
          }
          
          return {
            willSendResponse: async ({ response }) => {
              response.http.headers.set('X-RateLimit-Limit', '100');
              response.http.headers.set('X-RateLimit-Remaining', String(100 - requests));
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [rateLimitPlugin()]
    });
    
    // Per-query-cost rate limiting
    async function costBasedRateLimitPlugin() {
      return {
        requestDidStart: async ({ context, request }) => {
          const userId = context.user?.id;
          const cost = calculateQueryCost(request.query);
          
          const key = \`cost_limit:\${userId}\`;
          const currentCost = await redisClient.incrBy(key, cost);
          
          if (currentCost === cost) {
            await redisClient.expire(key, 60); // 1 minute window
          }
          
          if (currentCost > 10000) {
            throw new Error(\`Query cost budget exceeded: \${currentCost}/10000\`);
          }
        }
      };
    }
    \`\`\`
    
    **4. Field-Level Authorization**
    
    **Schema with permissions**:
    \`\`\`javascript
    const { SchemaDirectiveVisitor } = require('apollo-server');
    const { defaultFieldResolver } = require('graphql');
    
    // Define @auth directive
    const typeDefs = gql\`
      directive @auth(requires: Role = USER) on FIELD_DEFINITION
      
      enum Role {
        USER
        ADMIN
        MODERATOR
      }
      
      type Query {
        me: User @auth(requires: USER)
        users: [User!]! @auth(requires: ADMIN)
        adminStats: Stats @auth(requires: ADMIN)
      }
      
      type User {
        id: ID!
        name: String!
        email: String! @auth(requires: USER)
        privateNotes: String @auth(requires: ADMIN)
      }
    \`;
    
    // Implement directive
    class AuthDirective extends SchemaDirectiveVisitor {
      visitFieldDefinition(field) {
        const { resolve = defaultFieldResolver } = field;
        const { requires } = this.args;
        
        field.resolve = async function (...args) {
          const context = args[2];
          const user = context.user;
          
          if (!user) {
            throw new AuthenticationError('Not authenticated');
          }
          
          if (!user.roles.includes(requires)) {
            throw new ForbiddenError(\`Requires role: \${requires}\`);
          }
          
          return resolve.apply(this, args);
        };
      }
    }
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      schemaDirectives: {
        auth: AuthDirective
      }
    });
    
    // Alternative: Check in resolver
    const resolvers = {
      User: {
        email: (parent, args, { user }) => {
          // Only return email if requesting own profile or admin
          if (user.id === parent.id || user.roles.includes('ADMIN')) {
            return parent.email;
          }
          throw new ForbiddenError('Cannot access email');
        },
        
        privateNotes: (parent, args, { user }) => {
          if (!user.roles.includes('ADMIN')) {
            throw new ForbiddenError('Admin only');
          }
          return parent.privateNotes;
        }
      }
    };
    \`\`\`
    
    **5. Input Validation**
    
    \`\`\`javascript
    const Joi = require('joi');
    
    const resolvers = {
      Mutation: {
        createUser: async (parent, args, context) => {
          // Validate input
          const schema = Joi.object({
            name: Joi.string().min(2).max(50).required(),
            email: Joi.string().email().required(),
            age: Joi.number().min(13).max(120)
          });
          
          const { error, value } = schema.validate(args);
          
          if (error) {
            throw new UserInputError('Invalid input', {
              validationErrors: error.details
            });
          }
          
          // Sanitize HTML in user input
          const sanitizedName = sanitizeHtml(value.name);
          
          return createUser({ ...value, name: sanitizedName });
        }
      }
    };
    \`\`\`
    
    **6. Query Timeout**
    
    \`\`\`javascript
    function timeoutPlugin(maxTimeout = 5000) {
      return {
        requestDidStart: () => {
          const timeout = setTimeout(() => {
            throw new Error(\`Query timeout after \${maxTimeout}ms\`);
          }, maxTimeout);
          
          return {
            willSendResponse: () => {
              clearTimeout(timeout);
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [timeoutPlugin(5000)] // 5 second timeout
    });
    \`\`\`
    
    **7. Monitoring and Alerting**
    
    \`\`\`javascript
    const Sentry = require('@sentry/node');
    
    function monitoringPlugin() {
      return {
        requestDidStart: ({ request }) => {
          const start = Date.now();
          
          return {
            didEncounterErrors: ({ errors }) => {
              errors.forEach(error => {
                // Log to Sentry
                Sentry.captureException(error, {
                  contexts: {
                    graphql: {
                      query: request.query,
                      variables: request.variables
                    }
                  }
                });
                
                // Log suspicious queries
                if (error.message.includes('depth') || 
                    error.message.includes('cost')) {
                  logger.warn('Potentially malicious query detected', {
                    query: request.query,
                    error: error.message,
                    user: request.context.user?.id
                  });
                }
              });
            },
            
            willSendResponse: ({ response }) => {
              const duration = Date.now() - start;
              
              // Alert on slow queries
              if (duration > 2000) {
                logger.warn(\`Slow query: \${duration}ms\`, {
                  query: request.query
                });
              }
              
              // Track metrics
              metrics.histogram('graphql.query.duration', duration, {
                operation: request.operationName
              });
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [monitoringPlugin()]
    });
    \`\`\`
    
    **8. Disable Introspection in Production**
    
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      introspection: process.env.NODE_ENV !== 'production',
      playground: process.env.NODE_ENV !== 'production'
    });
    \`\`\`
    
    **9. Persistent Query Whitelist**
    
    \`\`\`javascript
    // Only allow pre-approved queries in production
    const approvedQueries = new Map([
      ['abc123...', 'query GetUser($id: ID!) { user(id: $id) { id name } }'],
      ['def456...', 'query GetFeed { feed { id title } }']
    ]);
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [
        (context) => ({
          Document(node) {
            if (process.env.NODE_ENV === 'production') {
              const queryHash = context.request.extensions?.persistedQuery?.sha256Hash;
              
              if (!queryHash || !approvedQueries.has(queryHash)) {
                throw new Error('Query not in whitelist');
              }
            }
          }
        })
      ]
    });
    \`\`\`
    
    **Complete Security Middleware**:
    
    \`\`\`javascript
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      
      // Authentication
      context: ({ req }) => ({
        user: getUserFromToken(req),
        ip: req.ip
      }),
      
      // Validation rules
      validationRules: [
        depthLimit(7),                    // Max depth 7
        createComplexityLimitRule(1000), // Max cost 1000
        queryTimeoutRule(5000)            // 5 second timeout
      ],
      
      // Plugins
      plugins: [
        rateLimitPlugin(),
        monitoringPlugin(),
        costBasedRateLimitPlugin()
      ],
      
      // Production settings
      introspection: process.env.NODE_ENV !== 'production',
      playground: process.env.NODE_ENV !== 'production',
      
      // Schema directives
      schemaDirectives: {
        auth: AuthDirective
      }
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Query depth limit** prevents recursive query attacks
    2. **Query cost analysis** prevents expensive queries (complexity × multipliers)
    3. **Rate limiting** by user (requests/minute) and query cost (cost/minute)
    4. **Field-level authorization** with @auth directive or in resolvers
    5. **Input validation** with Joi or similar library
    6. **Query timeout** prevents long-running queries
    7. **Monitoring** logs suspicious queries and slow queries
    8. **Disable introspection** in production to hide schema
    9. **Persistent query whitelist** only allows pre-approved queries
    10. **Defense in depth**: layer multiple security measures
    
    **Security Checklist**:
    - ✅ Query depth limit configured
    - ✅ Query cost analysis implemented
    - ✅ Rate limiting (per-user and per-cost)
    - ✅ Field-level authorization
    - ✅ Input validation and sanitization
    - ✅ Query timeout
    - ✅ Monitoring and alerting
    - ✅ Introspection disabled in production
    - ✅ HTTPS enforced
    - ✅ JWT token validation
    - ✅ CSRF protection (if using cookies)
    - ✅ Audit logging for sensitive operations`,
          keyPoints: [
            'Implement query depth and complexity limits to prevent attacks',
            'Use field-level authorization for sensitive data',
            'Apply rate limiting per user/IP to prevent abuse',
            'Disable introspection in production',
            'Monitor query patterns for malicious behavior',
          ],
        },
        {
          id: 'graphql-performance',
          question:
            "You're experiencing performance issues with your GraphQL API. Some queries take 5-10 seconds to complete, and your database is overwhelmed with queries. Using monitoring tools, you discovered: (1) N+1 queries in multiple resolvers, (2) Inefficient pagination, (3) Missing caching. Design a comprehensive performance optimization strategy including DataLoader implementation, pagination improvements, caching at multiple levels, and monitoring. Provide specific code examples and expected performance improvements.",
          sampleAnswer: `**GraphQL Performance Optimization Strategy**
    
    **Current State** (Performance Issues):
    - Query duration: 5-10 seconds (p95)
    - Database queries per request: 500-1000 (N+1 problem)
    - Database CPU: 90% (overloaded)
    - Cache hit rate: 0% (no caching)
    
    **Target State**:
    - Query duration: <300ms (p95)
    - Database queries per request: <20
    - Database CPU: <40%
    - Cache hit rate: >70%
    
    ---
    
    **1. Solve N+1 Problem with DataLoader**
    
    **Problem**:
    \`\`\`graphql
    query {
      posts(first: 100) {     # 1 query
        id
        title
        author {              # 100 queries!
          name
        }
        comments {            # 100 queries!
          text
          author {            # N queries!
            name
          }
        }
      }
    }
    # Total: 1 + 100 + 100 + N = 200+ queries
    \`\`\`
    
    **Solution**:
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // 1. Create DataLoader for users
    function createUserLoader() {
      return new DataLoader(async (userIds) => {
        console.log(\`Batch loading \${userIds.length} users\`);
        
        // Single query for all user IDs
        const users = await db.users.findByIds(userIds);
        
        // Return in same order as requested
        const userMap = new Map(users.map(u => [u.id, u]));
        return userIds.map(id => userMap.get(id));
      });
    }
    
    // 2. Create DataLoader for comments by post
    function createCommentsByPostLoader() {
      return new DataLoader(async (postIds) => {
        console.log(\`Batch loading comments for \${postIds.length} posts\`);
        
        // Single query for all comments
        const comments = await db.comments.findByPostIds(postIds);
        
        // Group by postId
        const commentsByPost = new Map();
        postIds.forEach(id => commentsByPost.set(id, []));
        comments.forEach(comment => {
          commentsByPost.get(comment.postId).push(comment);
        });
        
        return postIds.map(id => commentsByPost.get(id) || []);
      });
    }
    
    // 3. Add loaders to context
    const server = new ApolloServer({
      context: ({ req }) => ({
        user: getUserFromToken(req),
        loaders: {
          user: createUserLoader(),
          commentsByPost: createCommentsByPostLoader()
        }
      })
    });
    
    // 4. Use in resolvers
    const resolvers = {
      Query: {
        posts: async () => {
          return await db.posts.findAll(); // 1 query
        }
      },
      
      Post: {
        author: async (post, args, { loaders }) => {
          // Batched! All posts in request combined into single query
          return await loaders.user.load(post.userId);
        },
        
        comments: async (post, args, { loaders }) => {
          // Batched! All posts in request combined into single query
          return await loaders.commentsByPost.load(post.id);
        }
      },
      
      Comment: {
        author: async (comment, args, { loaders }) => {
          // Batched!
          return await loaders.user.load(comment.userId);
        }
      }
    };
    \`\`\`
    
    **Result**:
    - **Before**: 1 + 100 + 100 + N = 200+ queries
    - **After**: 4 queries (posts, users, comments, comment authors)
    - **Improvement**: 98% reduction in queries
    
    ---
    
    **2. Advanced DataLoader Patterns**
    
    **Prime Cache After Create**:
    \`\`\`javascript
    const resolvers = {
      Mutation: {
        createUser: async (parent, args, { loaders }) => {
          const user = await db.users.create(args);
          
          // Prime the cache so future loads don't hit DB
          loaders.user.prime(user.id, user);
          
          return user;
        }
      }
    };
    \`\`\`
    
    **Clear Cache After Update**:
    \`\`\`javascript
    const resolvers = {
      Mutation: {
        updateUser: async (parent, { id, ...updates }, { loaders }) => {
          const user = await db.users.update(id, updates);
          
          // Clear old cached value
          loaders.user.clear(id);
          
          // Prime with new value
          loaders.user.prime(id, user);
          
          return user;
        }
      }
    };
    \`\`\`
    
    **Composite Keys**:
    \`\`\`javascript
    // Load posts by multiple criteria
    function createPostLoader() {
      return new DataLoader(async (keys) => {
        // keys = [{ userId: '1', status: 'published' }, ...]
        
        const posts = await db.posts.findByMultiple(keys);
        
        return keys.map(key => 
          posts.filter(p => 
            p.userId === key.userId && p.status === key.status
          )
        );
      }, {
        cacheKeyFn: (key) => \`\${key.userId}:\${key.status}\`
      });
    }
    \`\`\`
    
    ---
    
    **3. Efficient Pagination**
    
    **Problem: Offset-based pagination**:
    \`\`\`graphql
    # Slow for large offsets
    query {
      posts(limit: 20, offset: 10000) {
        # SELECT * FROM posts LIMIT 20 OFFSET 10000
        # Database must scan 10,020 rows!
      }
    }
    \`\`\`
    
    **Solution: Cursor-based pagination**:
    
    **Schema**:
    \`\`\`graphql
    type Query {
      posts(first: Int, after: String): PostConnection!
    }
    
    type PostConnection {
      edges: [PostEdge!]!
      pageInfo: PageInfo!
    }
    
    type PostEdge {
      node: Post!
      cursor: String!
    }
    
    type PageInfo {
      hasNextPage: Boolean!
      endCursor: String
    }
    \`\`\`
    
    **Resolver**:
    \`\`\`javascript
    const resolvers = {
      Query: {
        posts: async (parent, { first = 20, after }, context) => {
          let query = db.posts.query().limit(first + 1);
          
          if (after) {
            // Decode cursor (base64 encoded ID)
            const cursorId = Buffer.from(after, 'base64').toString('utf-8');
            query = query.where('id', '>', cursorId);
          }
          
          const posts = await query.orderBy('created_at', 'desc');
          const hasNextPage = posts.length > first;
          const edges = posts.slice(0, first);
          
          return {
            edges: edges.map(post => ({
              node: post,
              cursor: Buffer.from(post.id).toString('base64')
            })),
            pageInfo: {
              hasNextPage,
              endCursor: edges.length > 0 
                ? Buffer.from(edges[edges.length - 1].id).toString('base64')
                : null
            }
          };
        }
      }
    };
    \`\`\`
    
    **Benefits**:
    - No matter the page, always scans ~20 rows (not 10,020)
    - Consistent performance across all pages
    - Works well with real-time data (no skipped/duplicate items)
    
    ---
    
    **4. Multi-Level Caching**
    
    **Layer 1: DataLoader (Request-Level Cache)**:
    - Already implemented above
    - Deduplicates within single request
    - Duration: Request lifetime (~100ms)
    
    **Layer 2: Redis (Application-Level Cache)**:
    \`\`\`javascript
    const redis = require('redis');
    const redisClient = redis.createClient();
    
    // Cached resolver
    async function getCachedUser(userId) {
      const cacheKey = \`user:\${userId}\`;
      
      // Try cache first
      const cached = await redisClient.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }
      
      // Cache miss, load from DB
      const user = await db.users.findById(userId);
      
      // Store in cache (5 minutes)
      await redisClient.setEx(cacheKey, 300, JSON.stringify(user));
      
      return user;
    }
    
    // Integrate with DataLoader
    function createUserLoader() {
      return new DataLoader(async (userIds) => {
        // Check Redis for each ID
        const pipeline = redisClient.pipeline();
        userIds.forEach(id => pipeline.get(\`user:\${id}\`));
        const cachedResults = await pipeline.exec();
        
        // Separate hits and misses
        const hits = [];
        const misses = [];
        
        userIds.forEach((id, idx) => {
          if (cachedResults[idx][1]) {
            hits[idx] = JSON.parse(cachedResults[idx][1]);
          } else {
            misses.push({ id, idx });
          }
        });
        
        // Load misses from database
        if (misses.length > 0) {
          const missedIds = misses.map(m => m.id);
          const users = await db.users.findByIds(missedIds);
          
          // Store in Redis
          const cachePipeline = redisClient.pipeline();
          users.forEach(user => {
            cachePipeline.setEx(\`user:\${user.id}\`, 300, JSON.stringify(user));
          });
          await cachePipeline.exec();
          
          // Fill results
          misses.forEach((miss, idx) => {
            hits[miss.idx] = users[idx];
          });
        }
        
        return hits;
      });
    }
    \`\`\`
    
    **Layer 3: Apollo Response Cache**:
    \`\`\`javascript
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      plugins: [
        responseCachePlugin({
          sessionId: (context) => context.user?.id || null
        })
      ],
      cacheControl: {
        defaultMaxAge: 60 // 1 minute
      }
    });
    
    // Add cache hints to schema
    const typeDefs = gql\`
      type User @cacheControl(maxAge: 300) {
        id: ID!
        name: String!
      }
      
      type Post @cacheControl(maxAge: 60) {
        id: ID!
        title: String!
      }
      
      type Query {
        user(id: ID!): User @cacheControl(maxAge: 600)
      }
    \`;
    \`\`\`
    
    **Layer 4: CDN (Edge Cache)**:
    \`\`\`javascript
    // Use persisted queries for GET requests
    const { ApolloServer } = require('apollo-server-express');
    
    const server = new ApolloServer({
      persistedQueries: {
        cache: redis,
        ttl: 900
      }
    });
    
    // Client sends GET request with query hash
    // GET /graphql?extensions={"persistedQuery":{"version":1,"sha256Hash":"abc123..."}}
    
    // Response includes cache headers
    app.use('/graphql', (req, res, next) => {
      if (req.method === 'GET') {
        res.set('Cache-Control', 'public, max-age=60');
      }
      next();
    });
    \`\`\`
    
    ---
    
    **5. Database Query Optimization**
    
    **Add Indexes**:
    \`\`\`sql
    -- Index for post author lookups
    CREATE INDEX idx_posts_user_id ON posts(user_id);
    
    -- Index for comments by post
    CREATE INDEX idx_comments_post_id ON comments(post_id);
    
    -- Composite index for pagination
    CREATE INDEX idx_posts_created_at_id ON posts(created_at DESC, id);
    \`\`\`
    
    **Use SELECT instead of SELECT ***:
    \`\`\`javascript
    // Bad
    const users = await db.users.findByIds(userIds);
    // SELECT * FROM users WHERE id IN (...)
    
    // Good: Only select needed fields
    const users = await db.users
      .select('id', 'name', 'email', 'avatar')
      .findByIds(userIds);
    // SELECT id, name, email, avatar FROM users WHERE id IN (...)
    \`\`\`
    
    ---
    
    **6. Monitoring and Profiling**
    
    **Install Apollo Tracing**:
    \`\`\`javascript
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        ApolloServerPluginInlineTrace(),
        {
          requestDidStart() {
            const start = Date.now();
            let queryCount = 0;
            
            // Monkey-patch database query function
            const originalQuery = db.query;
            db.query = function(...args) {
              queryCount++;
              return originalQuery.apply(this, args);
            };
            
            return {
              willSendResponse({ response, context }) {
                const duration = Date.now() - start;
                
                console.log({
                  operation: context.operation?.name?.value,
                  duration: \`\${duration}ms\`,
                  queries: queryCount,
                  cacheHits: context.loaders?.user?.stats()?.cacheHits || 0
                });
                
                // Alert if slow
                if (duration > 1000) {
                  logger.warn('Slow query detected', {
                    operation: context.operation?.name?.value,
                    duration,
                    queries: queryCount
                  });
                }
                
                // Alert if too many queries
                if (queryCount > 50) {
                  logger.warn('Possible N+1 problem', {
                    operation: context.operation?.name?.value,
                    queries: queryCount
                  });
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Dashboard Metrics**:
    \`\`\`javascript
    const metrics = {
      queryDuration: new Histogram('graphql_query_duration'),
      databaseQueries: new Histogram('graphql_database_queries'),
      cacheHitRate: new Gauge('graphql_cache_hit_rate'),
      dataLoaderBatchSize: new Histogram('graphql_dataloader_batch_size')
    };
    \`\`\`
    
    ---
    
    **Expected Performance Improvements**:
    
    | **Metric** | **Before** | **After** | **Improvement** |
    |------------|-----------|---------|----------------|
    | **p50 latency** | 3s | 150ms | 95% faster |
    | **p95 latency** | 8s | 300ms | 96% faster |
    | **p99 latency** | 12s | 500ms | 96% faster |
    | **DB queries/request** | 500-1000 | 5-20 | 98% reduction |
    | **DB CPU** | 90% | 30% | 67% reduction |
    | **Cache hit rate** | 0% | 75% | Massive improvement |
    | **Throughput** | 50 RPS | 500 RPS | 10x improvement |
    
    **Key Takeaways**:
    
    1. **DataLoader solves N+1** - batch and cache database queries
    2. **Cursor-based pagination** - consistent performance across all pages
    3. **Multi-level caching** - Request → Redis → Response → CDN
    4. **Database indexes** - essential for joins and lookups
    5. **Monitoring** - track query count, duration, cache hit rate
    6. **Prime/clear cache** - keep DataLoader in sync with mutations
    7. **SELECT only needed fields** - reduce data transfer
    8. **Query depth/cost limits** - prevent expensive queries
    9. **Persisted queries** - enable CDN caching
    10. **Continuous profiling** - identify and fix regressions`,
          keyPoints: [
            'Use DataLoader to batch and deduplicate queries, solving N+1 problem',
            'Implement cursor-based pagination instead of offset/limit for efficiency',
            'Add caching at multiple levels: CDN (persisted queries), Redis (query results), Application (DataLoader)',
            'Monitor query patterns: complexity, duration, and frequency',
            'Set query depth and cost limits to prevent expensive queries',
            'Expected improvements: 20x fewer DB queries, 15x faster response time, 70% cache hit rate',
          ],
        },
      ],
    },
    {
      id: 'service-discovery',
      title: 'Service Discovery',
      content: `Service Discovery is the process by which services in a distributed system locate and communicate with each other. Understanding service discovery is crucial for building scalable microservices architectures.
    
    ## What is Service Discovery?
    
    In a dynamic microservices environment, services are constantly starting, stopping, and moving. **Service Discovery** enables services to find each other without hardcoding network locations.
    
    **Without Service Discovery**:
    \`\`\`javascript
    // Hardcoded - breaks when service moves
    const orderService = 'http://order-service-1.internal:8080';
    const response = await fetch(\`\${orderService}/orders/123\`);
    \`\`\`
    
    **With Service Discovery**:
    \`\`\`javascript
    // Dynamic - always finds current location
    const orderService = serviceRegistry.get('order-service');
    const response = await fetch(\`\${orderService}/orders/123\`);
    \`\`\`
    
    ---
    
    ## Service Discovery Patterns
    
    ### **1. Client-Side Discovery**
    
    **Clients query service registry and load balance**:
    
    \`\`\`
    Client
      ↓
      1. Query service registry for "order-service"
      ↓
    Service Registry (returns: [ip1, ip2, ip3])
      ↓
      2. Client chooses ip2 (round-robin)
      ↓
    Order Service Instance 2 (ip2)
    \`\`\`
    
    **Pros**:
    - No extra network hop
    - Client controls load balancing
    - Simple architecture
    
    **Cons**:
    - Clients must implement discovery logic
    - Couples clients to registry
    - Language-specific client libraries
    
    **Example: Netflix Eureka**:
    \`\`\`javascript
    const Eureka = require('eureka-js-client').Eureka;
    
    const client = new Eureka({
      instance: {
        app: 'user-service',
        hostName: 'localhost',
        ipAddr: '127.0.0.1',
        port: { '$': 3000, '@enabled': true },
        vipAddress: 'user-service',
        dataCenterInfo: {
          '@class': 'com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo',
          name: 'MyOwn'
        }
      },
      eureka: {
        host: 'eureka-server',
        port: 8761,
        servicePath: '/eureka/apps/'
      }
    });
    
    client.start();
    
    // Get service instances
    function getService(serviceName) {
      const instances = client.getInstancesByAppId(serviceName);
      // Client-side load balancing (round-robin)
      const instance = instances[Math.floor(Math.random() * instances.length)];
      return \`http://\${instance.ipAddr}:\${instance.port['$']}\`;
    }
    
    const orderServiceUrl = getService('ORDER-SERVICE');
    \`\`\`
    
    ### **2. Server-Side Discovery**
    
    **Load balancer queries registry**:
    
    \`\`\`
    Client
      ↓
      Request to load-balancer.internal
      ↓
    Load Balancer
      ↓
      1. Query service registry for "order-service"
      ↓
    Service Registry (returns: [ip1, ip2, ip3])
      ↓
      2. Load balancer chooses ip2
      ↓
    Order Service Instance 2 (ip2)
    \`\`\`
    
    **Pros**:
    - Clients don't need discovery logic
    - Centralized load balancing
    - Language-agnostic
    
    **Cons**:
    - Extra network hop
    - Load balancer is single point of failure
    - More complex infrastructure
    
    **Example: AWS ELB + ECS**:
    - ECS registers tasks with ELB
    - Clients call ELB DNS name
    - ELB routes to healthy instances
    
    ---
    
    ## Service Registry Patterns
    
    ### **1. Self-Registration**
    
    **Services register themselves**:
    
    \`\`\`javascript
    // Service startup
    const serviceInfo = {
      name: 'user-service',
      id: uuidv4(),
      address: process.env.HOST,
      port: process.env.PORT,
      health: '/health'
    };
    
    await consul.agent.service.register(serviceInfo);
    
    // Heartbeat to maintain registration
    setInterval(async () => {
      await consul.agent.check.pass(\`service:\${serviceInfo.id}\`);
    }, 10000);
    
    // Deregister on shutdown
    process.on('SIGTERM', async () => {
      await consul.agent.service.deregister(serviceInfo.id);
      process.exit(0);
    });
    \`\`\`
    
    **Pros**:
    - Simple - service manages its own lifecycle
    - No external registration logic
    
    **Cons**:
    - Services must implement registration logic
    - Language-specific clients needed
    
    ### **2. Third-Party Registration**
    
    **External registrar registers services**:
    
    \`\`\`
    Kubernetes
      ↓
      Watches service deployments
      ↓
    Registrar (sidecar)
      ↓
      Registers/deregisters with service registry
      ↓
    Consul/Eureka/etcd
    \`\`\`
    
    **Example: Kubernetes + Service**:
    \`\`\`yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: user-service
    spec:
      selector:
        app: user-service
      ports:
        - protocol: TCP
          port: 80
          targetPort: 3000
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: user-service
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: user-service
      template:
        metadata:
          labels:
            app: user-service
        spec:
          containers:
            - name: user-service
              image: user-service:1.0.0
              ports:
                - containerPort: 3000
    \`\`\`
    
    **Kubernetes automatically**:
    - Registers pods with Service
    - Updates endpoints as pods start/stop
    - Provides DNS resolution (user-service.default.svc.cluster.local)
    - Load balances across pods
    
    ---
    
    ## Popular Service Discovery Tools
    
    ### **1. Consul (HashiCorp)**
    
    **Features**:
    - Service registry and health checking
    - Key-value store
    - Multi-datacenter support
    - Service mesh capabilities
    
    **Registration**:
    \`\`\`javascript
    const Consul = require('consul');
    const consul = new Consul({ host: 'consul.service.consul', port: 8500 });
    
    // Register service
    await consul.agent.service.register({
      id: 'user-service-1',
      name: 'user-service',
      address: '10.0.1.5',
      port: 3000,
      check: {
        http: 'http://10.0.1.5:3000/health',
        interval: '10s',
        timeout: '5s'
      }
    });
    
    // Query service
    const services = await consul.health.service({
      service: 'user-service',
      passing: true // Only healthy instances
    });
    
    services.forEach(service => {
      console.log(\`\${service.Service.Address}:\${service.Service.Port}\`);
    });
    \`\`\`
    
    **DNS Interface**:
    \`\`\`bash
    # Query via DNS
    dig @127.0.0.1 -p 8600 user-service.service.consul
    
    # Returns:
    user-service.service.consul. 0 IN A 10.0.1.5
    user-service.service.consul. 0 IN A 10.0.1.6
    user-service.service.consul. 0 IN A 10.0.1.7
    \`\`\`
    
    ### **2. etcd**
    
    **Features**:
    - Distributed key-value store
    - Used by Kubernetes
    - Strong consistency (Raft consensus)
    - Watch mechanism for real-time updates
    
    **Example**:
    \`\`\`javascript
    const { Etcd3 } = require('etcd3');
    const client = new Etcd3();
    
    // Register service with TTL lease
    const lease = client.lease(10); // 10 second TTL
    await lease.put('services/user-service/instance-1').value(JSON.stringify({
      address: '10.0.1.5',
      port: 3000
    }));
    
    // Keep-alive to maintain registration
    await lease.keepalive();
    
    // Query services
    const services = await client.getAll()
      .prefix('services/user-service/')
      .strings();
    
    console.log(services);
    \`\`\`
    
    ### **3. ZooKeeper**
    
    **Features**:
    - Distributed coordination service
    - Used by Kafka, Hadoop
    - Hierarchical namespace
    - Watches for changes
    
    **Example**:
    \`\`\`javascript
    const zookeeper = require('node-zookeeper-client');
    const client = zookeeper.createClient('localhost:2181');
    
    client.once('connected', async () => {
      // Create ephemeral node (disappears when client disconnects)
      await client.create(
        '/services/user-service/instance-1',
        Buffer.from(JSON.stringify({ address: '10.0.1.5', port: 3000 })),
        zookeeper.CreateMode.EPHEMERAL
      );
      
      // Watch for changes
      const children = await client.getChildren('/services/user-service', (event) => {
        console.log('Services changed:', event);
      });
    });
    
    client.connect();
    \`\`\`
    
    ### **4. Kubernetes Service Discovery**
    
    **Built-in DNS**:
    \`\`\`javascript
    // Services automatically get DNS names
    const response = await fetch('http://user-service.default.svc.cluster.local/users/123');
    
    // Short form within same namespace
    const response = await fetch('http://user-service/users/123');
    \`\`\`
    
    **Environment Variables**:
    \`\`\`javascript
    // Kubernetes injects service info
    const userServiceHost = process.env.USER_SERVICE_SERVICE_HOST;
    const userServicePort = process.env.USER_SERVICE_SERVICE_PORT;
    \`\`\`
    
    **API Server**:
    \`\`\`javascript
    const k8s = require('@kubernetes/client-node');
    const kc = new k8s.KubeConfig();
    kc.loadFromDefault();
    
    const k8sApi = kc.makeApiClient(k8s.CoreV1Api);
    
    // Get service endpoints
    const endpoints = await k8sApi.readNamespacedEndpoints('user-service', 'default');
    endpoints.body.subsets.forEach(subset => {
      subset.addresses.forEach(address => {
        console.log(\`\${address.ip}:\${subset.ports[0].port}\`);
      });
    });
    \`\`\`
    
    ---
    
    ## Health Checking
    
    **Critical for service discovery** - only route traffic to healthy instances.
    
    ### **Active Health Checks**:
    
    \`\`\`javascript
    // Health check endpoint
    app.get('/health', async (req, res) => {
      try {
        // Check dependencies
        await database.ping();
        await redis.ping();
        
        res.status(200).json({
          status: 'healthy',
          uptime: process.uptime(),
          timestamp: Date.now()
        });
      } catch (error) {
        res.status(503).json({
          status: 'unhealthy',
          error: error.message
        });
      }
    });
    
    // Consul health check
    await consul.agent.service.register({
      name: 'user-service',
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s',    // Check every 10 seconds
        timeout: '5s',      // Timeout after 5 seconds
        deregistercriticalserviceafter: '1m' // Deregister if unhealthy for 1 min
      }
    });
    \`\`\`
    
    ### **Passive Health Checks**:
    
    \`\`\`javascript
    // Track failures, remove unhealthy instances
    class CircuitBreaker {
      constructor(threshold = 5) {
        this.failureCount = new Map();
        this.threshold = threshold;
      }
      
      recordFailure(instanceId) {
        const count = this.failureCount.get(instanceId) || 0;
        this.failureCount.set(instanceId, count + 1);
        
        if (count + 1 >= this.threshold) {
          console.log(\`Instance \${instanceId} marked unhealthy\`);
          this.removeFromPool(instanceId);
        }
      }
      
      recordSuccess(instanceId) {
        this.failureCount.set(instanceId, 0);
      }
    }
    \`\`\`
    
    ---
    
    ## Load Balancing with Service Discovery
    
    ### **Client-Side Load Balancing**:
    
    \`\`\`javascript
    class ServiceClient {
      constructor(serviceName, consul) {
        this.serviceName = serviceName;
        this.consul = consul;
        this.instances = [];
        this.currentIndex = 0;
        
        // Refresh instances periodically
        this.refreshInstances();
        setInterval(() => this.refreshInstances(), 30000);
      }
      
      async refreshInstances() {
        const services = await this.consul.health.service({
          service: this.serviceName,
          passing: true
        });
        
        this.instances = services.map(s => ({
          address: s.Service.Address,
          port: s.Service.Port
        }));
      }
      
      // Round-robin load balancing
      getNext() {
        if (this.instances.length === 0) {
          throw new Error('No healthy instances');
        }
        
        const instance = this.instances[this.currentIndex];
        this.currentIndex = (this.currentIndex + 1) % this.instances.length;
        
        return \`http://\${instance.address}:\${instance.port}\`;
      }
      
      // Random load balancing
      getRandom() {
        const instance = this.instances[Math.floor(Math.random() * this.instances.length)];
        return \`http://\${instance.address}:\${instance.port}\`;
      }
      
      // Weighted load balancing
      getWeighted() {
        // Implement weighted random selection
        // (instances can have different weights based on capacity)
      }
    }
    
    // Usage
    const userService = new ServiceClient('user-service', consul);
    
    async function makeRequest(userId) {
      const url = userService.getNext();
      return await fetch(\`\${url}/users/\${userId}\`);
    }
    \`\`\`
    
    ---
    
    ## Service Mesh (Advanced Service Discovery)
    
    **Service mesh** adds a dedicated infrastructure layer for service-to-service communication.
    
    **Popular Service Meshes**:
    - **Istio** (Google/IBM/Lyft)
    - **Linkerd** (CNCF)
    - **Consul Connect** (HashiCorp)
    
    **Architecture**:
    \`\`\`
    Service A → Envoy Proxy (sidecar) → Envoy Proxy (sidecar) → Service B
                    ↓                              ↓
              Control Plane (Istio/Linkerd)
                    ↓
             Service Registry
    \`\`\`
    
    **Features**:
    - Automatic service discovery
    - Load balancing
    - Circuit breaking
    - Mutual TLS
    - Request tracing
    - Metrics collection
    - Traffic splitting (canary deployments)
    
    **Example: Istio Virtual Service**:
    \`\`\`yaml
    apiVersion: networking.istio.io/v1beta1
    kind: VirtualService
    metadata:
      name: user-service
    spec:
      hosts:
        - user-service
      http:
        - match:
            - headers:
                x-version:
                  exact: "v2"
          route:
            - destination:
                host: user-service
                subset: v2
        - route:
            - destination:
                host: user-service
                subset: v1
              weight: 90
            - destination:
                host: user-service
                subset: v2
              weight: 10
    \`\`\`
    
    ---
    
    ## When to Use Service Discovery
    
    ### **✅ Use Service Discovery When:**
    
    1. **Microservices Architecture**
       - Many services communicating
       - Services scale independently
       - Dynamic environments (containers, cloud)
    
    2. **Auto-Scaling**
       - Instances constantly added/removed
       - Need automatic registration/deregistration
    
    3. **Multi-Region Deployments**
       - Services in different regions
       - Geo-routing based on location
    
    4. **Zero-Downtime Deployments**
       - Rolling updates
       - Blue-green deployments
       - Canary releases
    
    ### **❌ Avoid Service Discovery When:**
    
    1. **Monolithic Application**
       - Single application
       - Static deployment
    
    2. **Small Number of Services**
       - 2-3 services
       - Rarely change
       - Can use environment variables
    
    3. **Simple Environment**
       - Single server
       - No auto-scaling
       - Fixed IP addresses
    
    ---
    
    ## Common Mistakes
    
    ### **❌ Mistake 1: No Health Checks**
    
    \`\`\`javascript
    // Bad: Register without health check
    consul.agent.service.register({
      name: 'user-service',
      port: 3000
    });
    // Service stays registered even if crashed!
    
    // Good: Always include health check
    consul.agent.service.register({
      name: 'user-service',
      port: 3000,
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s'
      }
    });
    \`\`\`
    
    ### **❌ Mistake 2: Caching Service Locations Too Long**
    
    \`\`\`javascript
    // Bad: Cache forever
    const userServiceUrl = await getService('user-service');
    // Service might have moved!
    
    // Good: Refresh periodically
    setInterval(async () => {
      this.cachedServices = await refreshServices();
    }, 30000); // 30 seconds
    \`\`\`
    
    ### **❌ Mistake 3: No Graceful Shutdown**
    
    \`\`\`javascript
    // Bad: Abrupt shutdown
    process.on('SIGTERM', () => {
      process.exit(0); // Requests in flight will fail!
    });
    
    // Good: Deregister, then drain
    process.on('SIGTERM', async () => {
      // 1. Deregister from service discovery
      await consul.agent.service.deregister(serviceId);
      
      // 2. Stop accepting new requests
      server.close();
      
      // 3. Wait for in-flight requests to complete
      await waitForRequestsToDrain();
      
      // 4. Exit
      process.exit(0);
    });
    \`\`\`
    
    ### **❌ Mistake 4: Single Point of Failure**
    
    \`\`\`javascript
    // Bad: Single Consul server
    const consul = new Consul({ host: 'consul-server' });
    
    // Good: Consul cluster with multiple nodes
    const consul = new Consul({
      host: 'consul.service.consul', // DNS round-robin to multiple servers
      promisify: true
    });
    \`\`\`
    
    ---
    
    ## Real-World Example: E-Commerce with Service Discovery
    
    **Architecture**:
    \`\`\`
                  Consul Cluster
                        ↓
        +--------------+---------------+--------------+
        |              |               |              |
    API Gateway   Order Service   User Service   Inventory Service
        |              |               |              |
      (3 instances) (5 instances)  (3 instances)  (2 instances)
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    // Order Service
    const consul = new Consul();
    
    // Register on startup
    await consul.agent.service.register({
      id: \`order-service-\${process.env.INSTANCE_ID}\`,
      name: 'order-service',
      address: process.env.HOST,
      port: parseInt(process.env.PORT),
      tags: ['http', 'v1'],
      check: {
        http: \`http://\${process.env.HOST}:\${process.env.PORT}/health\`,
        interval: '10s',
        timeout: '5s'
      }
    });
    
    // Service client for calling other services
    class MicroserviceClient {
      constructor(serviceName) {
        this.serviceName = serviceName;
        this.consul = new Consul();
      }
      
      async call(path, options = {}) {
        // Get healthy instances
        const services = await this.consul.health.service({
          service: this.serviceName,
          passing: true
        });
        
        if (services.length === 0) {
          throw new Error(\`No healthy instances of \${this.serviceName}\`);
        }
        
        // Round-robin
        const service = services[Math.floor(Math.random() * services.length)];
        const url = \`http://\${service.Service.Address}:\${service.Service.Port}\${path}\`;
        
        // Make request with retry
        let attempts = 0;
        while (attempts < 3) {
          try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(\`HTTP \${response.status}\`);
            return await response.json();
          } catch (error) {
            attempts++;
            if (attempts >= 3) throw error;
            await new Promise(resolve => setTimeout(resolve, 100 * attempts));
          }
        }
      }
    }
    
    // Create order (calls inventory and user services)
    app.post('/orders', async (req, res) => {
      const { userId, items } = req.body;
      
      // Call user service to validate user
      const userClient = new MicroserviceClient('user-service');
      const user = await userClient.call(\`/users/\${userId}\`);
      
      // Call inventory service to check availability
      const inventoryClient = new MicroserviceClient('inventory-service');
      const availability = await inventoryClient.call('/check', {
        method: 'POST',
        body: JSON.stringify({ items })
      });
      
      if (!availability.available) {
        return res.status(400).json({ error: 'Items not available' });
      }
      
      // Create order
      const order = await db.orders.create({ userId, items });
      res.json(order);
    });
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **Service discovery enables dynamic service location** without hardcoded addresses
    2. **Two patterns**: Client-side (client queries registry) vs Server-side (load balancer queries)
    3. **Two registration patterns**: Self-registration vs Third-party registration
    4. **Popular tools**: Consul, etcd, ZooKeeper, Kubernetes DNS
    5. **Health checks are critical** - only route to healthy instances
    6. **Cache service locations** but refresh periodically (30-60 seconds)
    7. **Graceful shutdown**: Deregister → Stop accepting requests → Drain → Exit
    8. **Service mesh** adds advanced features (mTLS, observability, traffic management)
    9. **Load balancing strategies**: Round-robin, random, weighted, least connections
    10. **Essential for microservices** in dynamic, auto-scaling environments`,
      multipleChoice: [
        {
          id: 'service-discovery-pattern',
          question:
            'What is the primary difference between client-side and server-side service discovery?',
          options: [
            'Client-side is faster because it uses UDP instead of TCP',
            'In client-side discovery, the client queries the registry and chooses an instance; in server-side, a load balancer handles discovery',
            'Server-side discovery is more secure because it encrypts traffic',
            'Client-side discovery only works with Consul, server-side works with any registry',
          ],
          correctAnswer: 1,
          explanation:
            'In client-side discovery, the client directly queries the service registry to get a list of available instances and performs load balancing itself (e.g., Netflix Eureka). In server-side discovery, the client sends requests to a load balancer, which queries the registry and routes to an instance (e.g., AWS ELB). Client-side has no extra hop but requires discovery logic in clients; server-side is simpler for clients but adds a network hop.',
        },
        {
          id: 'service-discovery-health-check',
          question:
            'Your microservice registers with Consul but occasionally crashes without deregistering. What Consul feature prevents traffic from being routed to the crashed service?',
          options: [
            'Automatic garbage collection of dead services',
            'Health checks that mark services as unhealthy and optionally deregister them after a critical period',
            'Load balancer retry logic',
            'Circuit breaker pattern in the client',
          ],
          correctAnswer: 1,
          explanation:
            "Consul's health checks periodically probe services (e.g., HTTP GET /health every 10 seconds). If a service stops responding, Consul marks it as unhealthy and stops returning it in service queries. The `deregistercriticalserviceafter` option automatically deregisters services that remain unhealthy for a specified period. While circuit breakers help, they don't prevent initial routing; health checks prevent traffic from ever reaching unhealthy instances.",
        },
        {
          id: 'service-discovery-kubernetes',
          question:
            'In Kubernetes, what mechanism enables automatic service discovery without additional tools like Consul or etcd?',
          options: [
            'Kubernetes automatically installs Consul in every cluster',
            'Kubernetes provides built-in DNS that creates records for Services, allowing pods to resolve service names',
            'Kubernetes uses hardcoded IP addresses in ConfigMaps',
            'Kubernetes requires manual registration via kubectl commands',
          ],
          correctAnswer: 1,
          explanation:
            'Kubernetes has a built-in DNS service (CoreDNS) that automatically creates DNS records for every Service. When you create a Service named "user-service", pods can reach it at "user-service.default.svc.cluster.local" or simply "user-service" within the same namespace. Kubernetes also tracks Service endpoints automatically as pods start/stop, updating the DNS and iptables rules without manual intervention.',
        },
        {
          id: 'service-discovery-graceful-shutdown',
          question:
            'Why is graceful shutdown important when using service discovery, and what is the correct shutdown sequence?',
          options: [
            'Graceful shutdown is not important; service discovery handles crashed services automatically',
            'Deregister from service discovery, stop accepting new requests, wait for in-flight requests to complete, then exit',
            'Stop accepting new requests immediately, then deregister after all requests finish',
            'Send SIGKILL to force immediate termination so health checks detect failure faster',
          ],
          correctAnswer: 1,
          explanation:
            'Graceful shutdown is critical to prevent failed requests. The correct sequence is: (1) Deregister from service discovery (stops new traffic), (2) Stop accepting new connections (server.close()), (3) Wait for in-flight requests to complete (connection draining), (4) Exit. If you exit immediately, requests in flight will fail. If you stop accepting requests before deregistering, new requests might still arrive from clients with cached service locations.',
        },
        {
          id: 'service-discovery-mesh',
          question:
            'What is the primary advantage of using a service mesh (like Istio) over traditional service discovery?',
          options: [
            'Service mesh is faster because it uses binary protocols',
            'Service mesh only requires one server instead of a cluster',
            'Service mesh adds a sidecar proxy that handles discovery, load balancing, encryption, and observability without changing application code',
            'Service mesh eliminates the need for health checks',
          ],
          correctAnswer: 2,
          explanation:
            'A service mesh deploys a sidecar proxy (like Envoy) alongside each service. The proxy handles service discovery, load balancing, retries, circuit breaking, mutual TLS encryption, and distributed tracing—all without modifying application code. Services just talk to localhost:port, and the proxy handles everything. Traditional service discovery requires each service to implement discovery logic, load balancing, and security. Service mesh still uses health checks and typically requires a cluster for high availability.',
        },
      ],
      quiz: [
        {
          id: 'service-discovery-migration',
          question:
            "Your company is migrating from a monolithic application to microservices. You have 20 services that need to communicate with each other, and you're deploying on AWS ECS. Design a service discovery strategy using either Consul or AWS Cloud Map. Explain how services register, how clients discover services, how health checks work, and how you'd handle graceful shutdowns. Include specific implementation details and monitoring strategies.",
          sampleAnswer: `**Service Discovery Strategy for AWS ECS Migration**
    
    **Choice: AWS Cloud Map** (native integration with ECS)
    
    **Why Cloud Map over Consul**:
    - Native AWS integration (no additional infrastructure)
    - Automatic ECS task registration/deregistration
    - Integrated with Route 53 for DNS-based discovery
    - Health checks integrated with ECS
    - Lower operational overhead
    
    **Architecture**:
    \`\`\`
                      AWS Cloud Map
                            ↓
             +-----------------------------+
             |                             |
        ECS Service A               ECS Service B
        (3 tasks)                   (5 tasks)
             |                             |
       ALB/Service Connect        ALB/Service Connect
    \`\`\`
    
    **1. Service Registration**
    
    **CloudFormation Template**:
    \`\`\`yaml
    Resources:
      # Cloud Map namespace
      PrivateNamespace:
        Type: AWS::ServiceDiscovery::PrivateDnsNamespace
        Properties:
          Name: internal.mycompany.local
          Vpc: !Ref VPC
      
      # Service discovery for User Service
      UserServiceDiscovery:
        Type: AWS::ServiceDiscovery::Service
        Properties:
          Name: user-service
          DnsConfig:
            DnsRecords:
              - Type: A
                TTL: 10
            NamespaceId: !Ref PrivateNamespace
          HealthCheckCustomConfig:
            FailureThreshold: 1
      
      # ECS Service with Service Discovery
      UserService:
        Type: AWS::ECS::Service
        Properties:
          ServiceName: user-service
          Cluster: !Ref ECSCluster
          TaskDefinition: !Ref UserServiceTaskDef
          DesiredCount: 3
          LaunchType: FARGATE
          NetworkConfiguration:
            AwsvpcConfiguration:
              Subnets:
                - !Ref PrivateSubnet1
                - !Ref PrivateSubnet2
              SecurityGroups:
                - !Ref UserServiceSecurityGroup
          ServiceRegistries:
            - RegistryArn: !GetAtt UserServiceDiscovery.Arn
              ContainerName: user-service
              ContainerPort: 3000
          HealthCheckGracePeriodSeconds: 60
    \`\`\`
    
    **Automatic Registration**:
    - ECS automatically registers tasks when they start
    - Each task gets DNS entry: \`<task-id>.user-service.internal.mycompany.local\`
    - Service DNS entry: \`user-service.internal.mycompany.local\` (returns all healthy IPs)
    
    **2. Service Discovery in Application Code**
    
    **Option A: DNS-based (Simplest)**:
    \`\`\`javascript
    // services/client.js
    class ServiceClient {
      constructor(serviceName) {
        // Use Cloud Map DNS name
        this.baseUrl = \`http://\${serviceName}.internal.mycompany.local\`;
      }
      
      async call(path, options = {}) {
        const url = \`\${this.baseUrl}\${path}\`;
        
        // Add retry logic
        return this.retry(async () => {
          const response = await fetch(url, {
            ...options,
            timeout: 5000
          });
          
          if (!response.ok) {
            throw new Error(\`HTTP \${response.status}\`);
          }
          
          return response.json();
        });
      }
      
      async retry(fn, maxAttempts = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
          try {
            return await fn();
          } catch (error) {
            lastError = error;
            
            if (attempt < maxAttempts) {
              // Exponential backoff
              const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
              await new Promise(resolve => setTimeout(resolve, delay));
            }
          }
        }
        
        throw lastError;
      }
    }
    
    // Usage in Order Service
    const userClient = new ServiceClient('user-service');
    const inventoryClient = new ServiceClient('inventory-service');
    
    app.post('/orders', async (req, res) => {
      const { userId, items } = req.body;
      
      // Calls resolve via DNS to current healthy instances
      const user = await userClient.call(\`/users/\${userId}\`);
      const availability = await inventoryClient.call('/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items })
      });
      
      // Create order...
    });
    \`\`\`
    
    **Option B: AWS SDK (More Control)**:
    \`\`\`javascript
    const AWS = require('aws-sdk');
    const servicediscovery = new AWS.ServiceDiscovery();
    
    class CloudMapClient {
      constructor(serviceName, namespace) {
        this.serviceName = serviceName;
        this.namespace = namespace;
        this.instances = [];
        this.currentIndex = 0;
        
        // Refresh instances periodically
        this.refreshInstances();
        setInterval(() => this.refreshInstances(), 30000);
      }
      
      async refreshInstances() {
        try {
          const services = await servicediscovery.discoverInstances({
            NamespaceName: this.namespace,
            ServiceName: this.serviceName,
            HealthStatus: 'HEALTHY'
          }).promise();
          
          this.instances = services.Instances.map(inst => ({
            ip: inst.Attributes.AWS_INSTANCE_IPV4,
            port: inst.Attributes.AWS_INSTANCE_PORT
          }));
          
          console.log(\`Refreshed \${this.serviceName}: \${this.instances.length} instances\`);
        } catch (error) {
          console.error(\`Failed to refresh \${this.serviceName}:\`, error);
        }
      }
      
      getNextInstance() {
        if (this.instances.length === 0) {
          throw new Error(\`No healthy instances of \${this.serviceName}\`);
        }
        
        const instance = this.instances[this.currentIndex];
        this.currentIndex = (this.currentIndex + 1) % this.instances.length;
        
        return \`http://\${instance.ip}:\${instance.port}\`;
      }
    }
    
    const userClient = new CloudMapClient('user-service', 'internal.mycompany.local');
    \`\`\`
    
    **3. Health Checks**
    
    **Task Definition**:
    \`\`\`json
    {
      "family": "user-service",
      "containerDefinitions": [
        {
          "name": "user-service",
          "image": "user-service:1.0.0",
          "portMappings": [
            {
              "containerPort": 3000,
              "protocol": "tcp"
            }
          ],
          "healthCheck": {
            "command": [
              "CMD-SHELL",
              "curl -f http://localhost:3000/health || exit 1"
            ],
            "interval": 30,
            "timeout": 5,
            "retries": 3,
            "startPeriod": 60
          }
        }
      ]
    }
    \`\`\`
    
    **Health Check Endpoint**:
    \`\`\`javascript
    // Health check implementation
    app.get('/health', async (req, res) => {
      const checks = {
        database: false,
        redis: false,
        dependencies: false
      };
      
      try {
        // Check database
        await db.query('SELECT 1');
        checks.database = true;
        
        // Check Redis
        await redis.ping();
        checks.redis = true;
        
        // Check critical dependencies
        const userServiceHealth = await fetch('http://dependency-service.internal.mycompany.local/health', {
          timeout: 2000
        });
        checks.dependencies = userServiceHealth.ok;
        
        // All checks passed
        if (checks.database && checks.redis && checks.dependencies) {
          res.status(200).json({
            status: 'healthy',
            checks,
            uptime: process.uptime(),
            timestamp: Date.now()
          });
        } else {
          throw new Error('Some checks failed');
        }
      } catch (error) {
        res.status(503).json({
          status: 'unhealthy',
          checks,
          error: error.message
        });
      }
    });
    \`\`\`
    
    **4. Graceful Shutdown**
    
    \`\`\`javascript
    // Graceful shutdown handler
    let isShuttingDown = false;
    let activeConnections = 0;
    
    // Track active connections
    app.use((req, res, next) => {
      if (isShuttingDown) {
        res.status(503).send('Service is shutting down');
        return;
      }
      
      activeConnections++;
      res.on('finish', () => activeConnections--);
      next();
    });
    
    async function gracefulShutdown(signal) {
      console.log(\`Received \${signal}, starting graceful shutdown...\`);
      isShuttingDown = true;
      
      // 1. Mark health check as unhealthy (stops new traffic from Cloud Map)
      app.get('/health', (req, res) => {
        res.status(503).json({ status: 'shutting down' });
      });
      
      // 2. Wait a bit for ALB to detect unhealthy (deregistration delay)
      console.log('Waiting for deregistration...');
      await new Promise(resolve => setTimeout(resolve, 15000)); // 15 seconds
      
      // 3. Stop accepting new connections
      console.log('Stopping server...');
      server.close(() => {
        console.log('Server closed');
      });
      
      // 4. Wait for active connections to complete
      console.log(\`Draining \${activeConnections} active connections...\`);
      while (activeConnections > 0) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log(\`Remaining connections: \${activeConnections}\`);
      }
      
      // 5. Close database connections
      console.log('Closing database...');
      await db.close();
      await redis.quit();
      
      console.log('Shutdown complete');
      process.exit(0);
    }
    
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    \`\`\`
    
    **5. Monitoring & Observability**
    
    **CloudWatch Metrics**:
    \`\`\`javascript
    const { CloudWatch } = require('aws-sdk');
    const cloudwatch = new CloudWatch();
    
    // Middleware to track metrics
    app.use((req, res, next) => {
      const start = Date.now();
      
      res.on('finish', () => {
        const duration = Date.now() - start;
        
        // Send custom metric
        cloudwatch.putMetricData({
          Namespace: 'UserService',
          MetricData: [
            {
              MetricName: 'RequestDuration',
              Value: duration,
              Unit: 'Milliseconds',
              Dimensions: [
                { Name: 'Endpoint', Value: req.path },
                { Name: 'StatusCode', Value: String(res.statusCode) }
              ]
            },
            {
              MetricName: 'RequestCount',
              Value: 1,
              Unit: 'Count',
              Dimensions: [
                { Name: 'Endpoint', Value: req.path },
                { Name: 'StatusCode', Value: String(res.statusCode) }
              ]
            }
          ]
        }).promise().catch(console.error);
      });
      
      next();
    });
    \`\`\`
    
    **Service Discovery Metrics**:
    \`\`\`javascript
    // Track service discovery health
    setInterval(async () => {
      const services = ['user-service', 'inventory-service', 'payment-service'];
      
      for (const service of services) {
        try {
          const response = await servicediscovery.discoverInstances({
            NamespaceName: 'internal.mycompany.local',
            ServiceName: service,
            HealthStatus: 'HEALTHY'
          }).promise();
          
          await cloudwatch.putMetricData({
            Namespace: 'ServiceDiscovery',
            MetricData: [
              {
                MetricName: 'HealthyInstances',
                Value: response.Instances.length,
                Unit: 'Count',
                Dimensions: [{ Name: 'Service', Value: service }]
              }
            ]
          }).promise();
        } catch (error) {
          console.error(\`Failed to check \${service}:\`, error);
        }
      }
    }, 60000); // Every minute
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    # CloudWatch Alarm
    ServiceHealthyInstancesAlarm:
      Type: AWS::CloudWatch::Alarm
      Properties:
        AlarmName: user-service-low-healthy-instances
        AlarmDescription: Alert when healthy instances drop below 2
        Namespace: ServiceDiscovery
        MetricName: HealthyInstances
        Dimensions:
          - Name: Service
            Value: user-service
        Statistic: Average
        Period: 60
        EvaluationPeriods: 2
        Threshold: 2
        ComparisonOperator: LessThanThreshold
        AlarmActions:
          - !Ref SNSTopic
    \`\`\`
    
    **6. Testing Strategy**
    
    **Integration Test**:
    \`\`\`javascript
    describe('Service Discovery', () => {
      it('should discover user-service instances', async () => {
        const client = new ServiceClient('user-service');
        const response = await client.call('/health');
        expect(response.status).toBe('healthy');
      });
      
      it('should handle service unavailability gracefully', async () => {
        const client = new ServiceClient('nonexistent-service');
        await expect(client.call('/test')).rejects.toThrow();
      });
      
      it('should retry on transient failures', async () => {
        const client = new ServiceClient('flaky-service');
        // Mock: First 2 calls fail, 3rd succeeds
        const response = await client.call('/test');
        expect(response).toBeDefined();
      });
    });
    \`\`\`
    
    **Expected Results**:
    
    | **Metric** | **Target** | **Implementation** |
    |------------|-----------|-------------------|
    | **Service Discovery Latency** | <50ms | DNS caching, Cloud Map |
    | **Health Check Interval** | 30s | ECS health checks |
    | **Deregistration Time** | <30s | Health check + grace period |
    | **Failed Request Rate** | <0.1% | Retry logic, health checks |
    | **Graceful Shutdown Time** | <60s | Connection draining |
    
    **Key Takeaways**:
    
    1. **AWS Cloud Map** integrates natively with ECS for automatic registration
    2. **DNS-based discovery** is simplest (service-name.namespace.local)
    3. **Health checks** prevent routing to unhealthy tasks
    4. **Graceful shutdown** critical: mark unhealthy → stop accepting → drain → exit
    5. **Retry logic** handles transient failures
    6. **Monitor** healthy instance count, discovery latency, failed requests
    7. **TTL** of 10 seconds balances freshness and query load
    8. **Connection draining** prevents failed requests during deployments`,
          keyPoints: [
            'Consul for service discovery: DNS (simple) or HTTP API (advanced)',
            'Health checks critical: HTTP endpoint checking dependencies (DB, Redis, etc.)',
            'Client-side caching with 10-second TTL balances freshness and query load',
            'Connection draining during deployment: health check fails → removed from registry → existing connections finish',
            'Monitor: healthy instance count, discovery latency, failed request rate',
            'TTL trade-off: Lower = fresher data but more queries, Higher = less load but stale data',
          ],
        },
        {
          id: 'service-discovery-consul-cluster',
          question:
            'Design a highly available Consul cluster for service discovery in a production environment with 100+ microservices across 3 data centers. Include cluster topology, quorum requirements, network configuration, backup/recovery strategy, monitoring, and how to handle split-brain scenarios. Provide specific configurations and operational procedures.',
          sampleAnswer: `**Highly Available Consul Cluster Design**
    
    **1. Cluster Topology**
    
    **Architecture: Multi-Datacenter with WAN Federation**
    
    \`\`\`
    Datacenter 1 (us-east-1)          Datacenter 2 (us-west-2)          Datacenter 3 (eu-west-1)
    ├── Consul Server 1 (leader)      ├── Consul Server 4                ├── Consul Server 7
    ├── Consul Server 2               ├── Consul Server 5                ├── Consul Server 8
    ├── Consul Server 3               ├── Consul Server 6                ├── Consul Server 9
    └── Consul Clients (100+)         └── Consul Clients (100+)          └── Consul Clients (100+)
                ↓                                  ↓                                  ↓
        WAN Gossip Protocol ←→ WAN Gossip Protocol ←→ WAN Gossip Protocol
    \`\`\`
    
    **Quorum Requirements**:
    - **Consul uses Raft consensus** - requires majority (N/2 + 1) for writes
    - **3 servers per DC**: Tolerates 1 failure (quorum: 2/3)
    - **5 servers per DC**: Tolerates 2 failures (quorum: 3/5)
    - **Recommended**: 3 or 5 servers per DC (never even numbers!)
    
    **Why 3 DCs**:
    - Tolerates 1 entire DC failure
    - Majority still available: 2 DCs with 6 servers (quorum: 4/9)
    
    **2. Consul Server Configuration**
    
    **server.hcl** (Datacenter 1):
    \`\`\`hcl
    # Basic server config
    datacenter = "us-east-1"
    node_name = "consul-server-1"
    data_dir = "/opt/consul/data"
    log_level = "INFO"
    
    # Server mode
    server = true
    bootstrap_expect = 3
    
    # Network
    bind_addr = "10.0.1.10"      # Private IP
    advertise_addr = "10.0.1.10"
    client_addr = "0.0.0.0"      # Listen on all interfaces
    
    # Join other servers in same DC
    retry_join = ["10.0.1.11", "10.0.1.12"]
    
    # UI
    ui_config {
      enabled = true
    }
    
    # Performance tuning
    performance {
      raft_multiplier = 1  # Default (lower = faster, higher = more stable)
    }
    
    # Encryption
    encrypt = "base64-encoded-32-byte-key"
    encrypt_verify_incoming = true
    encrypt_verify_outgoing = true
    
    # TLS
    tls {
      defaults {
        ca_file = "/etc/consul/ca.pem"
        cert_file = "/etc/consul/server.pem"
        key_file = "/etc/consul/server-key.pem"
        verify_incoming = true
        verify_outgoing = true
      }
    }
    
    # ACLs
    acl {
      enabled = true
      default_policy = "deny"
      enable_token_persistence = true
      tokens {
        initial_management = "bootstrap-token"
        agent = "agent-token"
      }
    }
    
    # Autopilot (automatic operator-friendly management)
    autopilot {
      cleanup_dead_servers = true
      last_contact_threshold = "200ms"
      max_trailing_logs = 250
      server_stabilization_time = "10s"
    }
    
    # Telemetry
    telemetry {
      prometheus_retention_time = "60s"
      disable_hostname = false
    }
    \`\`\`
    
    **3. WAN Federation (Cross-DC)**
    
    **Primary DC (us-east-1) server.hcl**:
    \`\`\`hcl
    primary_datacenter = "us-east-1"
    
    # WAN join addresses
    retry_join_wan = [
      "consul-server-4.us-west-2.example.com",
      "consul-server-5.us-west-2.example.com",
      "consul-server-7.eu-west-1.example.com",
      "consul-server-8.eu-west-1.example.com"
    ]
    
    # WAN gossip encryption
    encrypt_wan = "base64-encoded-32-byte-key"
    \`\`\`
    
    **Secondary DC (us-west-2) server.hcl**:
    \`\`\`hcl
    datacenter = "us-west-2"
    primary_datacenter = "us-east-1"  # Replicates ACLs from primary
    
    retry_join_wan = [
      "consul-server-1.us-east-1.example.com",
      "consul-server-2.us-east-1.example.com",
      "consul-server-7.eu-west-1.example.com",
      "consul-server-8.eu-west-1.example.com"
    ]
    \`\`\`
    
    **4. Consul Client Configuration**
    
    **client.hcl** (on application servers):
    \`\`\`hcl
    datacenter = "us-east-1"
    node_name = "app-server-1"
    data_dir = "/opt/consul/data"
    
    # Client mode
    server = false
    
    # Join servers in same DC
    retry_join = [
      "consul-server-1.internal",
      "consul-server-2.internal",
      "consul-server-3.internal"
    ]
    
    bind_addr = "{{ GetPrivateIP }}"
    
    # Encryption
    encrypt = "base64-encoded-32-byte-key"
    
    # TLS
    tls {
      defaults {
        ca_file = "/etc/consul/ca.pem"
        verify_incoming = false  # Clients don't need incoming verification
        verify_outgoing = true
      }
    }
    
    # ACLs
    acl {
      enabled = true
      default_policy = "deny"
      tokens {
        agent = "agent-token"
        default = "service-token"
      }
    }
    \`\`\`
    
    **5. Service Registration**
    
    **service.hcl** (user-service):
    \`\`\`hcl
    service {
      name = "user-service"
      id = "user-service-1"
      port = 3000
      tags = ["v1", "http"]
      
      # Health check
      check {
        id = "user-service-health"
        name = "HTTP Health Check"
        http = "http://localhost:3000/health"
        interval = "10s"
        timeout = "5s"
        deregister_critical_service_after = "1m"
      }
      
      # Service metadata
      meta {
        version = "1.2.3"
        environment = "production"
      }
      
      # Connect (service mesh)
      connect {
        sidecar_service {}
      }
    }
    \`\`\`
    
    **6. Split-Brain Prevention**
    
    **Problem**: Network partition causes two groups of servers to elect separate leaders.
    
    **Solution 1: Odd Number of Servers**
    \`\`\`
    3 servers: Partition → 2 (quorum) vs 1 (no quorum)
    5 servers: Partition → 3 (quorum) vs 2 (no quorum)
    
    NEVER 4 servers: Partition → 2 vs 2 (both lose quorum!)
    \`\`\`
    
    **Solution 2: Datacenter-Aware Placement**
    - Deploy servers across availability zones
    - Never 2 servers in same AZ (for 3-server cluster)
    
    **Solution 3: Monitoring for Split**
    \`\`\`bash
    # Check for multiple leaders
    curl http://localhost:8500/v1/status/leader
    
    # Should return same leader across all servers
    # If different leaders returned, split-brain detected!
    \`\`\`
    
    **Solution 4: Automatic Recovery**
    \`\`\`hcl
    autopilot {
      cleanup_dead_servers = true
      # Servers unreachable for >72h automatically removed
      # Prevents split-brain from persisting
    }
    \`\`\`
    
    **7. Backup & Recovery**
    
    **Automated Snapshots**:
    \`\`\`bash
    #!/bin/bash
    # /usr/local/bin/consul-backup.sh
    
    # Take snapshot
    consul snapshot save \\
      -token=\${CONSUL_TOKEN} \\
      /backups/consul-snapshot-\$(date +%Y%m%d-%H%M%S).snap
    
    # Upload to S3
    aws s3 cp /backups/consul-snapshot-*.snap \\
      s3://my-consul-backups/\$(date +%Y/%m/%d)/
    
    # Retention: Keep last 30 days
    find /backups -name "consul-snapshot-*.snap" -mtime +30 -delete
    \`\`\`
    
    **Cron**:
    \`\`\`
    # Every 6 hours
    0 */6 * * * /usr/local/bin/consul-backup.sh
    \`\`\`
    
    **Disaster Recovery**:
    \`\`\`bash
    # Restore from snapshot
    consul snapshot restore \\
      -token=\${CONSUL_TOKEN} \\
      /backups/consul-snapshot-20250101-120000.snap
    
    # Verify
    consul catalog services
    consul catalog nodes
    \`\`\`
    
    **Restore Procedure**:
    1. Stop all Consul servers
    2. Clear data directories: \`rm -rf /opt/consul/data/*\`
    3. Start one server with restored snapshot
    4. Wait for it to become leader
    5. Start remaining servers (they'll sync from leader)
    
    **8. Monitoring**
    
    **Key Metrics**:
    \`\`\`yaml
    # Consul exposes Prometheus metrics at /v1/agent/metrics
    
    metrics_to_monitor:
      # Cluster health
      - consul_raft_peers: Number of Raft peers (should equal server count)
      - consul_raft_leader: Is this server the leader? (1 = yes)
      - consul_raft_apply_time: Time to apply Raft log (should be <10ms)
      
      # Performance
      - consul_rpc_request_time: RPC request latency
      - consul_serf_member_flap: Member flapping (joining/leaving rapidly)
      - consul_dns_query_time: DNS query latency
      
      # Service discovery
      - consul_catalog_services_total: Number of registered services
      - consul_health_node_status: Node health status
      - consul_health_service_status: Service health status
    \`\`\`
    
    **Prometheus Config**:
    \`\`\`yaml
    scrape_configs:
      - job_name: 'consul'
        metrics_path: '/v1/agent/metrics'
        params:
          format: ['prometheus']
        static_configs:
          - targets:
              - 'consul-server-1:8500'
              - 'consul-server-2:8500'
              - 'consul-server-3:8500'
    \`\`\`
    
    **Alerting Rules**:
    \`\`\`yaml
    groups:
      - name: consul
        rules:
          # No leader elected
          - alert: ConsulNoLeader
            expr: sum(consul_raft_leader) == 0
            for: 1m
            annotations:
              summary: "Consul cluster has no leader"
          
          # Raft peer mismatch
          - alert: ConsulRaftPeerMismatch
            expr: consul_raft_peers != 3
            for: 5m
            annotations:
              summary: "Expected 3 Raft peers, found {{ \$value }}"
          
          # High RPC latency
          - alert: ConsulHighRPCLatency
            expr: histogram_quantile(0.99, consul_rpc_request_time) > 1000
            for: 5m
            annotations:
              summary: "Consul RPC p99 latency > 1s"
          
          # Service unhealthy
          - alert: ServiceUnhealthy
            expr: consul_health_service_status{status="critical"} > 0
            for: 2m
            annotations:
              summary: "Service {{ \$labels.service }} is unhealthy"
    \`\`\`
    
    **9. Operational Procedures**
    
    **Adding a Server**:
    \`\`\`bash
    # 1. Deploy new server with same config
    # 2. It auto-joins via retry_join
    # 3. Autopilot promotes it to voter after stabilization
    
    # Verify
    consul operator raft list-peers
    \`\`\`
    
    **Removing a Server**:
    \`\`\`bash
    # Graceful removal (allows transfer of leadership)
    consul leave -id=consul-server-3
    
    # Verify
    consul operator raft list-peers
    \`\`\`
    
    **Rolling Restart**:
    \`\`\`bash
    # 1. Restart non-leader servers first
    for server in consul-server-2 consul-server-3; do
      ssh \$server "systemctl restart consul"
      sleep 60  # Wait for it to rejoin
    done
    
    # 2. Transfer leadership
    consul operator raft transfer-leader -id=consul-server-2
    
    # 3. Restart old leader
    ssh consul-server-1 "systemctl restart consul"
    \`\`\`
    
    **10. Security Best Practices**
    
    **Encryption**:
    \`\`\`bash
    # Generate gossip encryption key
    consul keygen  # Base64-encoded 32-byte key
    
    # Generate TLS certificates
    consul tls ca create
    consul tls cert create -server -dc=us-east-1
    consul tls cert create -client
    \`\`\`
    
    **ACL Bootstrap**:
    \`\`\`bash
    # Bootstrap ACLs
    consul acl bootstrap
    
    # Output:
    # AccessorID:   2b778dd9-f5f1-6f29-b4b4-9a5fa948757a
    # SecretID:     6a1253d2-1785-24fd-91c2-f8e78c745511
    
    # Create policies
    consul acl policy create \\
      -name "service-policy" \\
      -rules @service-policy.hcl
    
    # Create tokens
    consul acl token create \\
      -description "User Service Token" \\
      -policy-name "service-policy"
    \`\`\`
    
    **service-policy.hcl**:
    \`\`\`hcl
    service "user-service" {
      policy = "write"
    }
    
    service_prefix "" {
      policy = "read"
    }
    
    node_prefix "" {
      policy = "read"
    }
    \`\`\`
    
    **Key Takeaways**:
    
    1. **3 or 5 servers per DC** (never even numbers) - tolerates N/2 failures
    2. **Odd number of DCs** (3 recommended) - tolerates 1 DC failure
    3. **WAN federation** connects DCs for global service discovery
    4. **Gossip encryption + TLS** for security
    5. **ACLs with deny-default** prevent unauthorized access
    6. **Autopilot** for automatic dead server cleanup
    7. **Automated snapshots** every 6 hours to S3
    8. **Monitor Raft peers, leader, and latency**
    9. **Split-brain prevented** by quorum requirements
    10. **Graceful operations**: leave before removing, transfer leadership before restart`,
          keyPoints: [
            'HA Consul cluster: 5 servers (tolerates 2 failures) across 3 datacenters',
            'Raft consensus requires quorum: (N/2)+1 servers must agree',
            'Client agents on every node: cache queries, reduce load on servers',
            'WAN gossip for cross-datacenter communication',
            'Split-brain prevented by quorum requirements',
            'Backup strategy: Consul snapshots every 6 hours, test restoration monthly',
            'Monitor: Raft peers, leader stability, commit latency',
            'Graceful operations: leave before removing node, transfer leadership before restart',
          ],
        },
        {
          id: 'service-discovery-performance',
          question:
            'Your service discovery system is experiencing performance issues: DNS queries are timing out, service registrations are slow, and clients are getting stale service information. Debug and optimize the system. Include specific diagnostic steps, performance bottlenecks to check, caching strategies, and configuration tuning. Provide before/after metrics showing improvements.',
          sampleAnswer: `**Service Discovery Performance Optimization**
    
    **Symptoms**:
    - DNS queries timing out (>1000ms)
    - Service registration taking 5-10 seconds
    - Clients routing to dead instances
    - Consul CPU at 90%
    
    **1. Diagnostic Steps**
    
    **Step 1: Check Consul Cluster Health**
    \`\`\`bash
    # Check leader and peers
    consul operator raft list-peers
    
    # Output should show:
    # Node         ID         Address       State     Voter
    # server-1     server-1   10.0.1.10:8300  leader    true
    # server-2     server-2   10.0.1.11:8300  follower  true
    # server-3     server-3   10.0.1.12:8300  follower  true
    
    # Check for frequent leader elections (bad!)
    consul monitor --log-level=info | grep "leader election"
    
    # Get metrics
    curl http://localhost:8500/v1/agent/metrics?format=prometheus
    \`\`\`
    
    **Step 2: Check DNS Query Performance**
    \`\`\`bash
    # Time DNS query
    time dig @localhost -p 8600 user-service.service.consul
    
    # Should be <50ms
    # If >500ms, DNS is the bottleneck
    
    # Check DNS cache hit rate
    consul monitor | grep "dns.query_time"
    \`\`\`
    
    **Step 3: Check Service Count and Churn**
    \`\`\`bash
    # How many services?
    consul catalog services | wc -l
    
    # How many instances?
    consul catalog nodes | wc -l
    
    # Check registration rate
    consul monitor | grep "catalog.register"
    
    # High churn (frequent register/deregister) is expensive
    \`\`\`
    
    **Step 4: Check Network Latency**
    \`\`\`bash
    # Measure round-trip time to Consul server
    ping -c 10 consul-server-1
    
    # Should be <10ms within same region
    # If >50ms, network is bottleneck
    
    # Check for packet loss
    mtr consul-server-1
    \`\`\`
    
    **Step 5: Analyze Metrics**
    \`\`\`bash
    # Key metrics to check:
    consul_raft_apply_time         # Should be <10ms
    consul_rpc_request_time        # Should be <100ms  
    consul_dns_query_time          # Should be <50ms
    consul_catalog_register_time   # Should be <100ms
    \`\`\`
    
    **2. Root Cause Analysis**
    
    **Issue 1: DNS Query Overload**
    
    **Problem**: Clients query Consul DNS on every request
    
    \`\`\`javascript
    // Bad: DNS query on every request
    async function callUserService() {
      // DNS query happens here (1000ms!)
      const response = await fetch('http://user-service.service.consul/users/123');
    }
    
    // Called 1000 times/second = 1000 DNS queries/sec
    \`\`\`
    
    **Solution: Client-Side DNS Caching**
    
    \`\`\`javascript
    const dns = require('dns');
    const { promisify } = require('util');
    const resolve4 = promisify(dns.resolve4);
    
    class CachedDNSResolver {
      constructor(ttl = 30000) {
        this.cache = new Map();
        this.ttl = ttl;
      }
      
      async resolve(hostname) {
        const now = Date.now();
        const cached = this.cache.get(hostname);
        
        // Return cached if fresh
        if (cached && now - cached.timestamp < this.ttl) {
          return cached.addresses;
        }
        
        // Resolve and cache
        const addresses = await resolve4(hostname);
        this.cache.set(hostname, {
          addresses,
          timestamp: now
        });
        
        return addresses;
      }
    }
    
    const resolver = new CachedDNSResolver(30000); // 30 second cache
    
    async function callUserService() {
      // DNS query cached for 30 seconds
      const addresses = await resolver.resolve('user-service.service.consul');
      const address = addresses[Math.floor(Math.random() * addresses.length)];
      
      const response = await fetch(\`http://\${address}:3000/users/123\`);
    }
    \`\`\`
    
    **Result**: DNS queries reduced from 1000/sec to 33/sec (97% reduction)
    
    **Issue 2: Consul DNS Configuration**
    
    **Problem**: Default DNS config not optimized
    
    **Before**:
    \`\`\`hcl
    # Default config
    dns_config {
      allow_stale = false  # Always query leader (slow!)
      max_stale = "0s"
    }
    \`\`\`
    
    **After (Optimized)**:
    \`\`\`hcl
    dns_config {
      # Allow stale reads from followers (fast!)
      allow_stale = true
      max_stale = "5s"
      
      # Enable DNS caching
      node_ttl = "30s"
      service_ttl = {
        "*" = "30s"
        "user-service" = "10s"  # Lower TTL for critical services
      }
      
      # Performance tuning
      udp_answer_limit = 3  # Limit UDP response size
      enable_truncate = true
    }
    \`\`\`
    
    **Result**: DNS query latency reduced from 500ms to 10ms (98% improvement)
    
    **Issue 3: Excessive Service Churn**
    
    **Problem**: Services register/deregister frequently
    
    \`\`\`javascript
    // Bad: Register on every request
    app.get('/users/:id', async (req, res) => {
      await consul.agent.service.register({...});
      // Handle request
      await consul.agent.service.deregister('user-service-1');
    });
    \`\`\`
    
    **Solution**: Register once on startup
    
    \`\`\`javascript
    // Register once
    await consul.agent.service.register({
      id: 'user-service-1',
      name: 'user-service',
      port: 3000,
      check: {
        http: 'http://localhost:3000/health',
        interval: '10s',
        timeout: '5s',
        deregister_critical_service_after: '1m'
      }
    });
    
    // Deregister only on shutdown
    process.on('SIGTERM', async () => {
      await consul.agent.service.deregister('user-service-1');
      process.exit(0);
    });
    \`\`\`
    
    **Issue 4: Health Check Intervals Too Aggressive**
    
    **Before**:
    \`\`\`hcl
    check {
      http = "http://localhost:3000/health"
      interval = "1s"    # Check every second (expensive!)
      timeout = "500ms"
    }
    \`\`\`
    
    **After**:
    \`\`\`hcl
    check {
      http = "http://localhost:3000/health"
      interval = "10s"   # Check every 10 seconds
      timeout = "5s"
      
      # Critical: Automatically deregister if unhealthy for 1 min
      deregister_critical_service_after = "1m"
    }
    \`\`\`
    
    **Result**: Health check load reduced by 90%
    
    **Issue 5: Raft Apply Latency**
    
    **Problem**: Raft consensus slow (writes to all replicas)
    
    **Diagnosis**:
    \`\`\`bash
    # Check Raft metrics
    curl localhost:8500/v1/agent/metrics | grep raft_apply
    
    # If consul_raft_apply_time >100ms, Raft is slow
    \`\`\`
    
    **Solution 1: Use Stale Reads for Read-Heavy Workloads**
    \`\`\`javascript
    // Query with stale=true (reads from followers, no consensus)
    const services = await consul.health.service({
      service: 'user-service',
      passing: true,
      stale: true  // Much faster!
    });
    \`\`\`
    
    **Solution 2: Tune Raft Multiplier**
    \`\`\`hcl
    performance {
      # Lower = faster (but less stable)
      # Higher = more stable (but slower)
      raft_multiplier = 1  # Default: 5
    }
    \`\`\`
    
    **Solution 3: Upgrade Network**
    - Ensure low-latency network between servers (<10ms)
    - Use placement groups or same AZ
    
    **Issue 6: Client Configuration**
    
    **Problem**: Clients not configured optimally
    
    **Before**:
    \`\`\`javascript
    // Bad: New connection per query
    async function getService(name) {
      const consul = new Consul();
      return await consul.health.service({ service: name });
    }
    \`\`\`
    
    **After**:
    \`\`\`javascript
    // Good: Reuse connection
    const consul = new Consul({
      host: 'consul.service.consul',
      port: 8500,
      promisify: true,
      defaults: {
        stale: true,  // Allow stale reads
        token: process.env.CONSUL_TOKEN
      }
    });
    
    // Cache service locations
    class ServiceCache {
      constructor(refreshInterval = 30000) {
        this.cache = new Map();
        setInterval(() => this.refresh(), refreshInterval);
      }
      
      async get(serviceName) {
        if (!this.cache.has(serviceName)) {
          await this.refresh(serviceName);
        }
        
        const instances = this.cache.get(serviceName);
        return instances[Math.floor(Math.random() * instances.length)];
      }
      
      async refresh(serviceName = null) {
        const services = serviceName ? [serviceName] : Array.from(this.cache.keys());
        
        for (const service of services) {
          const instances = await consul.health.service({
            service,
            passing: true,
            stale: true
          });
          
          this.cache.set(service, instances.map(i => ({
            address: i.Service.Address,
            port: i.Service.Port
          })));
        }
      }
    }
    
    const serviceCache = new ServiceCache(30000);
    \`\`\`
    
    **3. Performance Improvements Summary**
    
    | **Metric** | **Before** | **After** | **Improvement** |
    |------------|-----------|---------|----------------|
    | **DNS Query Latency (p99)** | 500ms | 10ms | 98% faster |
    | **DNS Query Rate** | 1000/sec | 33/sec | 97% reduction |
    | **Service Registration Time** | 5s | 200ms | 96% faster |
    | **Consul CPU Usage** | 90% | 25% | 72% reduction |
    | **Failed Requests (stale data)** | 5% | 0.1% | 98% reduction |
    | **Health Check Load** | 1000 checks/sec | 100 checks/sec | 90% reduction |
    
    **4. Monitoring Dashboard**
    
    \`\`\`yaml
    # Grafana dashboard queries
    
    # DNS query latency
    histogram_quantile(0.99, 
      rate(consul_dns_query_time_bucket[5m])
    )
    
    # Service cache hit rate
    rate(service_cache_hits[5m]) / 
      (rate(service_cache_hits[5m]) + rate(service_cache_misses[5m]))
    
    # Raft apply latency
    consul_raft_apply_time
    
    # Registration rate
    rate(consul_catalog_register[5m])
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Client-side DNS caching** (30s TTL) reduces queries by 97%
    2. **Allow stale reads** for followers improves latency by 98%
    3. **Service TTL configuration** enables DNS caching
    4. **Health check intervals** should be 10s (not 1s)
    5. **Reuse Consul client** connections
    6. **Cache service locations** and refresh periodically
    7. **Raft multiplier tuning** for write-heavy workloads
    8. **Monitor**: DNS latency, query rate, Raft apply time, CPU
    9. **Graceful registration**: once on startup, not per request
    10. **Use stale=true** for read-heavy workloads (99% of queries)`,
          keyPoints: [
            'Root cause: DNS cache disabled, anti-entropy interval too short, no client-side caching',
            'Enable DNS caching with 10-second TTL on Consul servers',
            'Reduce anti-entropy sync interval from 1s to 60s',
            'Client-side caching with 10-second TTL to reduce DNS queries by 90%',
            'Use stale=true for DNS queries to avoid leader bottleneck',
            'Monitor: DNS query latency, query rate, Raft apply time, CPU usage',
            'Expected improvements: 100ms → 5ms DNS latency, 90% reduction in DNS queries',
          ],
        },
      ],
    },
    {
      id: 'network-protocols',
      title: 'Network Protocols',
      content: `Understanding network protocols is essential for system design. This section covers key protocols beyond HTTP, including SMTP, FTP, SSH, MQTT, AMQP, and WebRTC.
    
    ## Application Layer Protocols
    
    ### **1. SMTP (Simple Mail Transfer Protocol)**
    
    **Purpose**: Email transmission between mail servers
    
    **Port**: 25 (plain), 587 (submission with TLS)
    
    **How it Works**:
    \`\`\`
    User → Email Client → SMTP Server (sender) → SMTP Server (recipient) → Email Client → User
    \`\`\`
    
    **Example Session**:
    \`\`\`
    telnet smtp.example.com 25
    
    S: 220 smtp.example.com ESMTP Postfix
    C: HELO client.example.com
    S: 250 smtp.example.com
    C: MAIL FROM:<sender@example.com>
    S: 250 OK
    C: RCPT TO:<recipient@example.com>
    S: 250 OK
    C: DATA
    S: 354 End data with <CR><LF>.<CR><LF>
    C: Subject: Test Email
    C: 
    C: This is the email body.
    C: .
    S: 250 OK: queued as 12345
    C: QUIT
    S: 221 Bye
    \`\`\`
    
    **Node.js Example**:
    \`\`\`javascript
    const nodemailer = require('nodemailer');
    
    const transporter = nodemailer.createTransport({
      host: 'smtp.example.com',
      port: 587,
      secure: false, // STARTTLS
      auth: {
        user: 'sender@example.com',
        pass: 'password'
      }
    });
    
    await transporter.sendMail({
      from: 'sender@example.com',
      to: 'recipient@example.com',
      subject: 'Hello',
      text: 'Email body',
      html: '<p>Email body</p>'
    });
    \`\`\`
    
    **Use Cases**:
    - Sending transactional emails (order confirmations, password resets)
    - Email notifications
    - Newsletters
    
    ---
    
    ### **2. FTP/SFTP (File Transfer Protocol)**
    
    **FTP**: Plain text file transfer (Port 21)
    **SFTP**: Secure FTP over SSH (Port 22)
    
    **Problems with FTP**:
    - No encryption (credentials sent in plain text)
    - Requires separate data connection (complex firewalls)
    - Active vs Passive modes confusion
    
    **Better Alternative: SFTP**:
    \`\`\`bash
    # Upload file
    sftp user@server.com
    put local-file.txt /remote/path/
    
    # Download file
    get /remote/path/file.txt local-file.txt
    
    # Sync directory (rsync over SSH)
    rsync -avz --progress /local/dir/ user@server.com:/remote/dir/
    \`\`\`
    
    **Node.js Example (SFTP)**:
    \`\`\`javascript
    const Client = require('ssh2-sftp-client');
    const sftp = new Client();
    
    await sftp.connect({
      host: 'server.com',
      port: 22,
      username: 'user',
      password: 'password'
    });
    
    // Upload
    await sftp.put('/local/file.txt', '/remote/file.txt');
    
    // Download
    await sftp.get('/remote/file.txt', '/local/file.txt');
    
    // List directory
    const list = await sftp.list('/remote/path');
    console.log(list);
    
    await sftp.end();
    \`\`\`
    
    **Use Cases**:
    - Log file collection
    - Backup transfers
    - Data exchange with partners
    - Deployment scripts
    
    ---
    
    ### **3. SSH (Secure Shell)**
    
    **Purpose**: Secure remote shell access and command execution
    
    **Port**: 22
    
    **Key Features**:
    - Encrypted communication
    - Public key authentication
    - Port forwarding (tunneling)
    - File transfer (SCP, SFTP)
    
    **SSH Tunneling**:
    
    **Local Port Forwarding** (access remote service locally):
    \`\`\`bash
    # Access remote MySQL (port 3306) on localhost:3307
    ssh -L 3307:localhost:3306 user@remote-server
    
    # Now connect to localhost:3307 to reach remote MySQL
    mysql -h 127.0.0.1 -P 3307 -u root -p
    \`\`\`
    
    **Remote Port Forwarding** (expose local service remotely):
    \`\`\`bash
    # Expose local service (localhost:8080) on remote server port 8080
    ssh -R 8080:localhost:8080 user@remote-server
    
    # Remote server can now access your localhost:8080 via its port 8080
    \`\`\`
    
    **Dynamic Port Forwarding** (SOCKS proxy):
    \`\`\`bash
    # Create SOCKS proxy on localhost:1080
    ssh -D 1080 user@remote-server
    
    # Configure browser to use SOCKS proxy localhost:1080
    # All traffic routes through remote server
    \`\`\`
    
    **Node.js SSH Client**:
    \`\`\`javascript
    const { Client } = require('ssh2');
    const conn = new Client();
    
    conn.on('ready', () => {
      console.log('Client :: ready');
      
      // Execute command
      conn.exec('uptime', (err, stream) => {
        stream.on('data', (data) => {
          console.log('STDOUT: ' + data);
        });
        
        stream.on('close', () => {
          conn.end();
        });
      });
    });
    
    conn.connect({
      host: 'server.com',
      port: 22,
      username: 'user',
      privateKey: require('fs').readFileSync('/path/to/private/key')
    });
    \`\`\`
    
    ---
    
    ### **4. MQTT (Message Queuing Telemetry Transport)**
    
    **Purpose**: Lightweight pub/sub messaging for IoT devices
    
    **Port**: 1883 (plain), 8883 (TLS)
    
    **Key Features**:
    - Extremely lightweight (header as small as 2 bytes)
    - Pub/sub model with topics
    - QoS levels (0, 1, 2)
    - Retained messages
    - Last Will and Testament (LWT)
    
    **Architecture**:
    \`\`\`
    Publisher → MQTT Broker (Mosquitto/EMQX) → Subscriber
    \`\`\`
    
    **Topic Hierarchy**:
    \`\`\`
    home/living-room/temperature
    home/living-room/humidity
    home/bedroom/temperature
    home/+/temperature          # Wildcard: all rooms' temperature
    home/#                      # Wildcard: everything under home
    \`\`\`
    
    **QoS Levels**:
    - **QoS 0** (At most once): Fire and forget, no acknowledgment
    - **QoS 1** (At least once): Acknowledged, may receive duplicates
    - **QoS 2** (Exactly once): Four-way handshake, guaranteed delivery
    
    **Node.js Example**:
    \`\`\`javascript
    const mqtt = require('mqtt');
    const client = mqtt.connect('mqtt://broker.example.com');
    
    // Subscribe
    client.on('connect', () => {
      client.subscribe('home/+/temperature', (err) => {
        if (!err) {
          console.log('Subscribed');
        }
      });
    });
    
    // Receive messages
    client.on('message', (topic, message) => {
      console.log(\`\${topic}: \${message.toString()}\`);
      // home/living-room/temperature: 72.5
    });
    
    // Publish
    setInterval(() => {
      const temp = (Math.random() * 30 + 60).toFixed(1);
      client.publish('home/living-room/temperature', temp, { qos: 1 });
    }, 5000);
    \`\`\`
    
    **Last Will and Testament** (LWT):
    \`\`\`javascript
    // Set LWT when connecting
    const client = mqtt.connect('mqtt://broker.example.com', {
      will: {
        topic: 'devices/sensor-1/status',
        payload: 'offline',
        qos: 1,
        retain: true
      }
    });
    
    // If client disconnects unexpectedly, broker publishes LWT
    \`\`\`
    
    **Use Cases**:
    - IoT sensor data collection
    - Smart home automation
    - Real-time dashboards
    - Vehicle telemetry
    
    ---
    
    ### **5. AMQP (Advanced Message Queuing Protocol)**
    
    **Purpose**: Enterprise message queuing and routing
    
    **Port**: 5672 (plain), 5671 (TLS)
    
    **Popular Implementation**: RabbitMQ
    
    **Key Features**:
    - Reliable message delivery
    - Complex routing (direct, fanout, topic, headers)
    - Message persistence
    - Transactions
    - Flow control
    
    **Architecture**:
    \`\`\`
    Producer → Exchange → Queue → Consumer
    \`\`\`
    
    **Exchange Types**:
    
    **Direct Exchange** (routing key exact match):
    \`\`\`
    Producer --routing_key: "error"--> Exchange --"error"--> Queue (bound to "error")
    \`\`\`
    
    **Topic Exchange** (routing key pattern match):
    \`\`\`
    Producer --"user.created"--> Exchange --"user.*"--> Queue A
                                          --"*.created"--> Queue B
    \`\`\`
    
    **Fanout Exchange** (broadcast to all queues):
    \`\`\`
    Producer --> Exchange --> Queue A
                          --> Queue B
                          --> Queue C
    \`\`\`
    
    **Node.js Example (RabbitMQ)**:
    \`\`\`javascript
    const amqp = require('amqplib');
    
    // Producer
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    await channel.assertExchange('logs', 'fanout', { durable: false });
    
    setInterval(() => {
      const msg = \`Log message \${Date.now()}\`;
      channel.publish('logs', '', Buffer.from(msg));
      console.log(\`Sent: \${msg}\`);
    }, 1000);
    
    // Consumer
    const connection2 = await amqp.connect('amqp://localhost');
    const channel2 = await connection2.createChannel();
    
    await channel2.assertExchange('logs', 'fanout', { durable: false });
    
    const q = await channel2.assertQueue('', { exclusive: true });
    await channel2.bindQueue(q.queue, 'logs', '');
    
    channel2.consume(q.queue, (msg) => {
      console.log(\`Received: \${msg.content.toString()}\`);
    }, { noAck: true });
    \`\`\`
    
    **Work Queue Pattern**:
    \`\`\`javascript
    // Multiple workers consume from same queue
    // Each message delivered to only ONE worker (load balancing)
    
    // Worker 1
    channel.prefetch(1); // Only take one message at a time
    channel.consume('task_queue', (msg) => {
      const task = msg.content.toString();
      console.log(\`Worker 1 processing: \${task}\`);
      
      // Simulate work
      setTimeout(() => {
        channel.ack(msg); // Acknowledge completion
      }, 1000);
    });
    
    // Worker 2 (same code)
    // Messages distributed: W1, W2, W1, W2, W1, W2...
    \`\`\`
    
    **Use Cases**:
    - Task queues (image processing, email sending)
    - Event-driven architectures
    - Microservices communication
    - Job scheduling
    
    ---
    
    ### **6. WebRTC (Web Real-Time Communication)**
    
    **Purpose**: Peer-to-peer audio, video, and data transfer in browsers
    
    **Protocols Used**:
    - **STUN**: Discover public IP address (NAT traversal)
    - **TURN**: Relay traffic when peer-to-peer fails
    - **ICE**: Combines STUN/TURN to establish connection
    - **SDP**: Session description (offer/answer)
    - **DTLS-SRTP**: Encrypted media transport
    
    **Architecture**:
    \`\`\`
    Peer A ←→ Signaling Server ←→ Peer B
      ↓                              ↓
      +--------- Direct P2P ---------+
      (via STUN/TURN if needed)
    \`\`\`
    
    **Connection Flow**:
    \`\`\`
    1. Peer A creates offer (SDP)
    2. Peer A sends offer to signaling server
    3. Signaling server forwards to Peer B
    4. Peer B creates answer (SDP)
    5. Peer B sends answer back
    6. ICE candidates exchanged
    7. Direct peer-to-peer connection established
    \`\`\`
    
    **Simple WebRTC Example**:
    \`\`\`javascript
    // Peer A (caller)
    const peerA = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        {
          urls: 'turn:turn.example.com',
          username: 'user',
          credential: 'password'
        }
      ]
    });
    
    // Add local stream
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true
    });
    stream.getTracks().forEach(track => peerA.addTrack(track, stream));
    
    // Create offer
    const offer = await peerA.createOffer();
    await peerA.setLocalDescription(offer);
    
    // Send offer to Peer B via signaling server
    signalingServer.send({ type: 'offer', sdp: offer });
    
    // Handle ICE candidates
    peerA.onicecandidate = (event) => {
      if (event.candidate) {
        signalingServer.send({ type: 'ice-candidate', candidate: event.candidate });
      }
    };
    
    // Receive answer from Peer B
    signalingServer.on('answer', async (answer) => {
      await peerA.setRemoteDescription(answer);
    });
    
    // Peer B (answerer)
    const peerB = new RTCPeerConnection({ iceServers: [...] });
    
    // Receive offer from Peer A
    signalingServer.on('offer', async (offer) => {
      await peerB.setRemoteDescription(offer);
      
      // Add local stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      stream.getTracks().forEach(track => peerB.addTrack(track, stream));
      
      // Create answer
      const answer = await peerB.createAnswer();
      await peerB.setLocalDescription(answer);
      
      // Send answer back
      signalingServer.send({ type: 'answer', sdp: answer });
    });
    
    // Display remote stream
    peerB.ontrack = (event) => {
      remoteVideo.srcObject = event.streams[0];
    };
    \`\`\`
    
    **Use Cases**:
    - Video conferencing (Zoom, Google Meet)
    - Voice calls (WhatsApp Web, Discord)
    - Screen sharing
    - P2P file transfer
    - Live streaming
    
    ---
    
    ## Protocol Comparison
    
    | **Protocol** | **Transport** | **Use Case** | **Pros** | **Cons** |
    |--------------|---------------|--------------|----------|----------|
    | **HTTP/HTTPS** | TCP | Web requests | Universal, cacheable | Stateless, overhead |
    | **WebSocket** | TCP | Real-time bidirectional | Full-duplex, efficient | No HTTP caching |
    | **MQTT** | TCP | IoT pub/sub | Lightweight, QoS | Limited routing |
    | **AMQP** | TCP | Enterprise messaging | Complex routing, reliable | Heavier, complex |
    | **gRPC** | TCP (HTTP/2) | Microservices RPC | Fast, streaming | Browser support limited |
    | **WebRTC** | UDP (SRTP) | P2P audio/video | Low latency, P2P | Complex setup |
    
    ---
    
    ## When to Use Each Protocol
    
    ### **Use MQTT When:**
    - IoT devices with limited bandwidth
    - Millions of publishers/subscribers
    - Need QoS guarantees
    - Battery-powered devices (efficient)
    
    ### **Use AMQP When:**
    - Enterprise message queuing
    - Complex routing requirements
    - Need guaranteed delivery and ordering
    - Transactions required
    
    ### **Use WebRTC When:**
    - Real-time audio/video required
    - Peer-to-peer preferred (low latency)
    - Browser-based communication
    - Screen sharing needed
    
    ### **Use WebSocket When:**
    - Real-time updates (chat, notifications)
    - Bidirectional communication
    - Lower latency than HTTP polling
    - Browser support essential
    
    ---
    
    ## Key Takeaways
    
    1. **SMTP** for email transmission (use with TLS on port 587)
    2. **SFTP/SCP** for secure file transfer (never plain FTP)
    3. **SSH** enables secure remote access and tunneling
    4. **MQTT** ideal for IoT pub/sub with QoS levels
    5. **AMQP** provides enterprise-grade message queuing with RabbitMQ
    6. **WebRTC** enables P2P audio/video with NAT traversal via STUN/TURN
    7. **Choose protocol based on**: latency requirements, reliability needs, client capabilities
    8. **Security**: Always use TLS (SMTPS, SFTP, MQTTS, AMQPS)
    9. **IoT**: MQTT preferred over HTTP (lighter, persistent connections)
    10. **Video calls**: WebRTC for P2P, or RTMP/HLS for streaming`,
      multipleChoice: [
        {
          id: 'network-protocol-mqtt-qos',
          question: 'In MQTT, what is the difference between QoS 1 and QoS 2?',
          options: [
            'QoS 1 is faster but less reliable than QoS 2',
            'QoS 1 guarantees at-least-once delivery (may have duplicates), QoS 2 guarantees exactly-once delivery',
            'QoS 1 uses TCP, QoS 2 uses UDP',
            'QoS 1 is for small messages, QoS 2 is for large messages',
          ],
          correctAnswer: 1,
          explanation:
            'QoS 1 (At least once) acknowledges message receipt but may deliver duplicates if acknowledgment is lost. QoS 2 (Exactly once) uses a four-way handshake to guarantee exactly-once delivery with no duplicates, at the cost of higher latency. Both use TCP. QoS 0 is fire-and-forget with no acknowledgment.',
        },
        {
          id: 'network-protocol-webrtc-nat',
          question: 'What is the purpose of STUN and TURN servers in WebRTC?',
          options: [
            'STUN encrypts video streams, TURN compresses audio',
            'STUN discovers the public IP address for NAT traversal, TURN relays traffic when direct P2P connection fails',
            'STUN stores video recordings, TURN transcodes video formats',
            'STUN handles signaling, TURN handles media transport',
          ],
          correctAnswer: 1,
          explanation:
            'STUN (Session Traversal Utilities for NAT) helps clients discover their public IP address and port mappings to establish direct peer-to-peer connections through NATs. TURN (Traversal Using Relays around NAT) acts as a relay server when direct P2P fails due to strict NATs or firewalls, routing traffic through the server. Neither handles encryption (DTLS-SRTP does) or signaling (WebSocket/HTTP do).',
        },
        {
          id: 'network-protocol-amqp-exchange',
          question:
            'In AMQP (RabbitMQ), what is the difference between a fanout exchange and a topic exchange?',
          options: [
            'Fanout is faster because it uses UDP instead of TCP',
            'Fanout broadcasts messages to all bound queues ignoring routing keys; topic uses pattern matching on routing keys',
            'Fanout stores messages persistently, topic does not',
            'Fanout works with MQTT, topic works with HTTP',
          ],
          correctAnswer: 1,
          explanation:
            'A fanout exchange broadcasts every message to all queues bound to it, ignoring routing keys entirely (useful for pub/sub). A topic exchange routes messages based on routing key pattern matching using wildcards (* matches one word, # matches multiple). For example, routing key "user.created" matches patterns "user.*" and "*.created". Both use TCP and can have persistent messages.',
        },
        {
          id: 'network-protocol-ssh-tunnel',
          question:
            'You need to access a MySQL database running on a remote server that only allows connections from localhost. Which SSH tunneling technique should you use?',
          options: [
            'Dynamic port forwarding with -D flag',
            'Remote port forwarding with -R flag',
            'Local port forwarding with -L flag',
            'Reverse port forwarding with -X flag',
          ],
          correctAnswer: 2,
          explanation:
            "Local port forwarding (ssh -L 3307:localhost:3306 user@remote) forwards a local port (3307) to a remote destination (localhost:3306 from the remote server's perspective). This allows you to connect to localhost:3307 on your machine, which tunnels through SSH to the remote server's localhost:3306. Dynamic forwarding creates a SOCKS proxy, remote forwarding exposes local services remotely, and -X is for X11 forwarding.",
        },
        {
          id: 'network-protocol-comparison',
          question:
            'Which protocol would be most appropriate for a battery-powered IoT sensor that sends temperature readings every minute to a cloud service?',
          options: [
            'HTTP/HTTPS with polling',
            'WebSocket for continuous connection',
            'MQTT with QoS 1',
            'WebRTC for real-time updates',
          ],
          correctAnswer: 2,
          explanation:
            "MQTT is ideal for IoT devices because: (1) It's extremely lightweight (2-byte header vs HTTP's typical 100+ bytes), (2) Maintains a persistent connection with low overhead, (3) QoS 1 provides reliable delivery with acknowledgments, (4) Designed for unreliable networks. HTTP polling wastes battery with frequent connection overhead. WebSocket is heavier than MQTT. WebRTC is for P2P video/audio, not sensor data.",
        },
      ],
      quiz: [
        {
          id: 'network-protocol-iot-architecture',
          question:
            "Design a scalable IoT platform for 1 million smart home devices (sensors, cameras, thermostats) sending data to the cloud. Choose appropriate protocols for device-to-cloud communication, cloud-to-device commands, video streaming, and mobile app notifications. Justify your protocol choices, explain the architecture, and describe how you'd handle device authentication, message persistence, and fault tolerance.",
          sampleAnswer: `**IoT Platform Architecture for 1M Smart Home Devices**
    
    **1. Device Categories and Protocol Selection**
    
    | **Device Type** | **Data Pattern** | **Protocol** | **Justification** |
    |-----------------|------------------|--------------|-------------------|
    | **Sensors** (temperature, humidity, motion) | Periodic telemetry (1/min) | **MQTT** | Lightweight, persistent connection, QoS support |
    | **Thermostats** (bidirectional control) | Telemetry + commands | **MQTT** | Pub/sub for commands, efficient bidirectional |
    | **Cameras** (video streaming) | Continuous video | **RTSP → HLS/DASH** | RTSP from device, HLS/DASH to clients |
    | **Mobile App** (real-time notifications) | Push notifications | **WebSocket + FCM/APNS** | WebSocket for live data, FCM/APNS for offline push |
    
    **2. High-Level Architecture**
    
    \`\`\`
    Devices (1M)
        ↓
    MQTT Broker Cluster (EMQX)
        ↓
    Message Router (Kafka)
        ↓
        +--> Stream Processing (Flink) --> Time-Series DB (TimescaleDB)
        +--> Rule Engine (Drools) --> Alerts (SNS/SQS)
        +--> Video Ingestion --> S3 + CloudFront
        ↓
    API Gateway (GraphQL/REST)
        ↓
    Mobile/Web Apps
    \`\`\`
    
    **3. Device-to-Cloud Communication (MQTT)**
    
    **Why MQTT**:
    - **Lightweight**: 2-byte header vs HTTP's 100+ bytes
    - **Persistent connection**: No connection overhead per message
    - **QoS levels**: Guarantee delivery for critical data
    - **Last Will Testament**: Detect offline devices
    - **Topic hierarchy**: Organize devices efficiently
    
    **Topic Structure**:
    \`\`\`
    devices/{device_id}/telemetry      # Device publishes data
    devices/{device_id}/commands       # Cloud publishes commands
    devices/{device_id}/status         # Online/offline status (LWT)
    devices/{device_id}/errors         # Error reporting
    \`\`\`
    
    **Device Implementation** (Temperature Sensor):
    \`\`\`javascript
    const mqtt = require('mqtt');
    
    // Connect with TLS and auth
    const client = mqtt.connect('mqtts://mqtt.example.com:8883', {
      clientId: \`device-\${deviceId}\`,
      username: deviceId,
      password: deviceSecret,
      clean: false, // Persist session
      will: {
        topic: \`devices/\${deviceId}/status\`,
        payload: 'offline',
        qos: 1,
        retain: true
      },
      reconnectPeriod: 5000 // Auto-reconnect
    });
    
    client.on('connect', () => {
      // Publish online status
      client.publish(\`devices/\${deviceId}/status\`, 'online', {
        qos: 1,
        retain: true
      });
      
      // Subscribe to commands
      client.subscribe(\`devices/\${deviceId}/commands\`, { qos: 1 });
    });
    
    // Publish telemetry every minute
    setInterval(() => {
      const data = {
        temperature: readTemperature(),
        humidity: readHumidity(),
        battery: readBattery(),
        timestamp: Date.now()
      };
      
      client.publish(
        \`devices/\${deviceId}/telemetry\`,
        JSON.stringify(data),
        { qos: 1 } // At-least-once delivery
      );
    }, 60000);
    
    // Handle commands
    client.on('message', (topic, message) => {
      if (topic === \`devices/\${deviceId}/commands\`) {
        const command = JSON.parse(message.toString());
        handleCommand(command);
      }
    });
    \`\`\`
    
    **4. MQTT Broker Cluster (EMQX)**
    
    **Configuration for 1M Devices**:
    \`\`\`yaml
    # emqx.conf
    node.max_ports = 2097152  # Support 2M connections
    
    # Cluster configuration
    cluster.discovery = k8s
    cluster.k8s.apiserver = https://kubernetes.default.svc:443
    cluster.k8s.namespace = iot-platform
    
    # Connection limits per node
    mqtt.max_packet_size = 1MB
    mqtt.max_clientid_len = 256
    mqtt.max_topic_alias = 65535
    
    # Session persistence
    mqtt.session_expiry_interval = 7200s  # 2 hours
    
    # Resource limits
    listener.tcp.external.max_connections = 500000  # 500K per node (2 nodes = 1M)
    listener.tcp.external.acceptors = 64
    listener.tcp.external.max_conn_rate = 1000  # 1K connections/sec
    
    # TLS
    listener.ssl.external.handshake_timeout = 15s
    listener.ssl.external.keyfile = /etc/certs/key.pem
    listener.ssl.external.certfile = /etc/certs/cert.pem
    
    # Authentication (PostgreSQL backend)
    auth.pgsql.server = postgres:5432
    auth.pgsql.username_query = SELECT password_hash FROM devices WHERE device_id = \${username}
    
    # ACL (topic-level permissions)
    auth.pgsql.acl_query = SELECT allow, topic, action FROM device_acls WHERE device_id = \${username}
    \`\`\`
    
    **Scaling**:
    - **2 EMQX nodes**: 500K connections each = 1M total
    - **Auto-scaling**: Add nodes when connections >400K (80% capacity)
    - **Load balancing**: AWS NLB with TCP passthrough
    
    **5. Message Persistence and Fault Tolerance**
    
    **Challenge**: Device publishes while broker restarts → message lost?
    
    **Solution 1: MQTT Persistent Sessions**:
    \`\`\`javascript
    // Device connects with clean=false
    const client = mqtt.connect('mqtts://mqtt.example.com', {
      clean: false, // Persist session
      clientId: deviceId // Same client ID on reconnect
    });
    
    // Broker stores:
    // - Subscriptions
    // - Unacknowledged messages
    // - QoS 1/2 messages
    \`\`\`
    
    **Solution 2: Bridge to Kafka for Guaranteed Persistence**:
    \`\`\`yaml
    # EMQX rule engine bridges to Kafka
    rules:
      - sql: SELECT * FROM "devices/+/telemetry"
        actions:
          - kafka_produce:
              topic: device-telemetry
              partition: \${clientid}  # Same device always same partition
              key: \${device_id}
              value: \${payload}
    \`\`\`
    
    **Kafka Configuration**:
    \`\`\`properties
    # Replication for fault tolerance
    replication.factor=3
    min.insync.replicas=2
    
    # Retention
    log.retention.hours=168  # 7 days
    log.segment.bytes=1073741824  # 1GB segments
    
    # Partitions (for parallelism)
    partitions=100  # 10K devices per partition
    \`\`\`
    
    **6. Device Authentication and Security**
    
    **Authentication Flow**:
    \`\`\`
    1. Device provisioning:
       - Generate device_id and device_secret
       - Store in PostgreSQL with hash
       
    2. Device connection:
       - MQTT username = device_id
       - MQTT password = device_secret
       - EMQX queries PostgreSQL for validation
       
    3. Topic-level ACL:
       - Device can only publish to devices/{device_id}/*
       - Device can only subscribe to devices/{device_id}/commands
    \`\`\`
    
    **Device Provisioning**:
    \`\`\`javascript
    // Provisioning API
    app.post('/api/devices', async (req, res) => {
      const deviceId = uuidv4();
      const deviceSecret = crypto.randomBytes(32).toString('hex');
      const secretHash = await bcrypt.hash(deviceSecret, 10);
      
      await db.devices.create({
        device_id: deviceId,
        password_hash: secretHash,
        owner_id: req.user.id,
        created_at: new Date()
      });
      
      // Create ACL rules
      await db.device_acls.createMany([
        {
          device_id: deviceId,
          topic: \`devices/\${deviceId}/telemetry\`,
          action: 'publish',
          allow: true
        },
        {
          device_id: deviceId,
          topic: \`devices/\${deviceId}/commands\`,
          action: 'subscribe',
          allow: true
        }
      ]);
      
      res.json({
        device_id: deviceId,
        device_secret: deviceSecret, // Return ONCE, never again!
        mqtt_host: 'mqtts://mqtt.example.com:8883'
      });
    });
    \`\`\`
    
    **7. Cloud-to-Device Commands**
    
    **Challenge**: Send command to specific device
    
    **Solution**:
    \`\`\`javascript
    // API endpoint
    app.post('/api/devices/:deviceId/commands', async (req, res) => {
      const { deviceId } = req.params;
      const { command, params } = req.body;
      
      // Check device ownership
      const device = await db.devices.findOne({
        where: { device_id: deviceId, owner_id: req.user.id }
      });
      
      if (!device) {
        return res.status(404).json({ error: 'Device not found' });
      }
      
      // Check device is online
      const isOnline = await redis.get(\`device:\${deviceId}:online\`);
      if (!isOnline) {
        return res.status(503).json({ error: 'Device offline' });
      }
      
      // Publish command to device's command topic
      const message = JSON.stringify({
        command,
        params,
        timestamp: Date.now(),
        request_id: uuidv4()
      });
      
      await mqttClient.publish(
        \`devices/\${deviceId}/commands\`,
        message,
        { qos: 1 }
      );
      
      res.json({ status: 'sent' });
    });
    \`\`\`
    
    **8. Video Streaming Architecture**
    
    **Challenge**: 100K cameras streaming video
    
    **Architecture**:
    \`\`\`
    Camera (RTSP) → Media Server (Wowza/Kurento) → S3 (recordings)
                            ↓
                      HLS/DASH (adaptive)
                            ↓
                      CloudFront CDN → Mobile/Web App
    \`\`\`
    
    **Why not MQTT for video**:
    - MQTT designed for small messages (KB)
    - Video requires GB/hour bandwidth
    - RTSP optimized for continuous streams
    
    **Recording Flow**:
    \`\`\`javascript
    // Camera pushes RTSP stream
    // rtsp://camera-ip:554/stream
    
    // Media server ingests and creates HLS segments
    ffmpeg -i rtsp://camera-ip:554/stream \\
      -codec: copy \\
      -hls_time 6 \\
      -hls_list_size 10 \\
      -hls_flags delete_segments \\
      /tmp/camera-123/stream.m3u8
    
    // Upload segments to S3
    aws s3 sync /tmp/camera-123/ s3://video-streams/camera-123/
    \`\`\`
    
    **9. Mobile App Communication**
    
    **Real-Time Updates (WebSocket)**:
    \`\`\`javascript
    // User subscribes to their devices
    const ws = new WebSocket('wss://api.example.com/ws');
    
    ws.send(JSON.stringify({
      type: 'subscribe',
      devices: ['device-1', 'device-2', 'device-3']
    }));
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // data: { device_id: 'device-1', temperature: 72.5, ... }
      updateUI(data);
    };
    \`\`\`
    
    **Push Notifications (FCM/APNS)**:
    \`\`\`javascript
    // Rule engine triggers alert
    if (temperature > 85) {
      await sendPushNotification({
        token: userDeviceToken,
        title: 'High Temperature Alert',
        body: \`Living room: \${temperature}°F\`,
        data: { device_id: deviceId, type: 'alert' }
      });
    }
    \`\`\`
    
    **10. Monitoring and Observability**
    
    **Metrics to Track**:
    \`\`\`yaml
    # MQTT Broker
    - emqx_connections_count: Current connections
    - emqx_messages_received_rate: Messages/sec
    - emqx_messages_sent_rate: Messages/sec
    - emqx_session_count: Active sessions
    
    # Devices
    - devices_online_count: Online devices by type
    - messages_per_device_p50/p95: Message frequency
    - device_battery_level: Battery health
    
    # Infrastructure
    - kafka_consumer_lag: Processing delay
    - timescaledb_write_rate: Ingestion rate
    - api_latency_p99: API performance
    \`\`\`
    
    **Key Takeaways**:
    
    1. **MQTT for IoT**: Lightweight, QoS, persistent connections
    2. **EMQX cluster**: 500K connections per node, bridge to Kafka
    3. **Kafka for persistence**: Guaranteed message delivery, replay capability
    4. **Device auth**: Unique device_id + secret, topic-level ACL
    5. **Commands**: Publish to devices/{device_id}/commands with QoS 1
    6. **Video**: RTSP → HLS/DASH → CDN (not MQTT)
    7. **Mobile**: WebSocket for real-time, FCM/APNS for offline push
    8. **Scaling**: Horizontal scaling of MQTT brokers, partitioned Kafka
    9. **Fault tolerance**: MQTT sessions + Kafka replication
    10. **Monitor**: Connection count, message rate, consumer lag, battery health`,
          keyPoints: [
            'MQTT for IoT devices: Lightweight, persistent connections, QoS support, Last Will Testament',
            'EMQX cluster: 500K connections per node, TLS authentication, topic-level ACL',
            'Bridge MQTT to Kafka for guaranteed persistence and replay capability',
            'Video streaming: RTSP from device → HLS/DASH to clients (not MQTT)',
            'Mobile notifications: WebSocket for real-time + FCM/APNS for offline push',
            'Device authentication: Unique device_id + secret, stored in PostgreSQL',
            'Scaling: Horizontal MQTT brokers, partitioned Kafka (100 partitions for 1M devices)',
            'Fault tolerance: MQTT persistent sessions + Kafka replication (3x)',
          ],
        },
        {
          id: 'network-protocol-cdn-design',
          question:
            'Design a Content Delivery Network (CDN) for a global media streaming service serving 100M users. Choose protocols for content delivery, origin fetch, cache invalidation, and real-time analytics. Explain how you would handle cache coherence, origin shield, HTTP/3 benefits, and measure CDN performance. Include specific protocol choices and architectural decisions.',
          sampleAnswer: `**CDN Architecture for Global Media Streaming**

**1. High-Level Architecture**

\`\`\`
Users (100M)
    ↓
Edge Servers (1000+ PoPs globally)
    ↓ (cache miss)
Regional Origin Shields (10 regions)
    ↓ (cache miss)
Origin Servers (Content Source)
    ↑
CDN Control Plane (Cache invalidation, analytics)
\`\`\`

**2. Protocol Selection**

| **Component** | **Protocol** | **Justification** |
|---------------|--------------|-------------------|
| **User → Edge** | HTTP/3 (QUIC) | 0-RTT, connection migration, better mobile performance |
| **Edge → Origin Shield** | HTTP/2 with TLS 1.3 | Multiplexing, header compression, fast handshake |
| **Origin Shield → Origin** | HTTP/2 + gRPC | Efficient for metadata + content fetch |
| **Cache Invalidation** | gRPC streaming | Real-time bidirectional updates |
| **Analytics** | Protocol Buffers over HTTPS | Compact, efficient serialization |

---

**3. User → Edge: HTTP/3 (QUIC over UDP)**

**Why HTTP/3**:
- **0-RTT resumption**: Returning users connect instantly
- **Connection migration**: Mobile users changing networks (WiFi ↔ 4G) don't drop connections
- **Head-of-line blocking elimination**: Lost packets only block affected stream
- **Better congestion control**: BBR (Bottleneck Bandwidth and RTT) vs cubic

**Edge Server Configuration** (NGINX/Caddy):

\`\`\`nginx
http {
    # HTTP/3 support
    listen 443 quic reuseport;
    listen 443 ssl http2;  # Fallback to HTTP/2
    
    # Alt-Svc header to advertise HTTP/3
    add_header Alt-Svc 'h3=":443"; ma=86400';
    
    # QUIC settings
    quic_gso on;
    quic_retry on;
    
    # SSL/TLS
    ssl_protocols TLSv1.3;
    ssl_early_data on;  # 0-RTT
    
    # Caching
    proxy_cache cdn_cache;
    proxy_cache_valid 200 24h;
    proxy_cache_valid 404 1m;
    proxy_cache_key \$scheme\$host\$request_uri;
    
    # Cache lock (prevent thundering herd)
    proxy_cache_lock on;
    proxy_cache_lock_timeout 5s;
    
    location /video/ {
        # Serve from cache if available
        proxy_cache_use_stale error timeout updating;
        
        # On cache miss, fetch from origin shield
        proxy_pass https://origin-shield-\${geo};
        
        # Add cache status header
        add_header X-Cache-Status \$upstream_cache_status;
        add_header X-CDN-Pop \$server_name;
    }
}
\`\`\`

**Benefits of HTTP/3 for Streaming**:
- **15-30% faster video start**: 0-RTT eliminates 1 round trip
- **50% fewer rebuffers on mobile**: Connection migration prevents stalls
- **Better throughput**: No HOL blocking means one lost packet doesn't block entire stream

---

**4. Edge → Origin Shield: HTTP/2 with TLS 1.3**

**Why Origin Shield**:
- Reduces origin load by aggregating requests from multiple edges
- Collapses duplicate requests (100 edge servers requesting same video → 1 request to origin)
- Provides second-tier caching

**Origin Shield Architecture**:

\`\`\`
Edge Servers (US West) ────┐
Edge Servers (US East) ────┼──→ Origin Shield (US)  ─┐
Edge Servers (US Central) ─┘                          │
                                                       ├──→ Origin Servers
Edge Servers (Europe) ─────────→ Origin Shield (EU) ─┤
Edge Servers (Asia) ───────────→ Origin Shield (APAC)─┘
\`\`\`

**Why HTTP/2 for Edge → Origin Shield**:
- **Multiplexing**: 1 connection handles many concurrent video requests
- **Header compression**: HPACK reduces redundant headers
- **Server push**: Origin Shield can push manifest + first segment together
- **TLS 1.3**: Faster handshake (1-RTT vs 2-RTT in TLS 1.2)

**Edge Server → Origin Shield Request**:

\`\`\`javascript
// Edge server making request to origin shield
const http2 = require('http2');

const client = http2.connect('https://origin-shield-us.example.com');

const req = client.request({
  ':path': '/video/movie123/segment_00042.ts',
  'x-cdn-edge-id': process.env.EDGE_ID,
  'x-cdn-pop': process.env.POP_LOCATION,
  'x-forwarded-for': originalClientIP
});

req.on('response', (headers) => {
  const cacheStatus = headers['x-cache-status'];
  // 'hit' or 'miss' from origin shield
  
  if (cacheStatus === 'miss') {
    metrics.increment('origin_shield_miss');
  }
});

req.on('data', (chunk) => {
  // Stream video data to client
  clientResponse.write(chunk);
});

req.on('end', () => {
  clientResponse.end();
  client.close();
});
\`\`\`

---

**5. Cache Invalidation: gRPC Streaming**

**Challenge**: When content is updated (live sports score, breaking news), invalidate cached copies across 1000+ edge servers.

**gRPC Bidirectional Streaming Solution**:

\`\`\`protobuf
// cdn_control.proto
service CDNControl {
  // Each edge server opens persistent connection to control plane
  rpc StreamInvalidations (stream InvalidationAck) returns (stream InvalidationRequest);
}

message InvalidationRequest {
  string request_id = 1;
  repeated string cache_keys = 2;  // URLs or patterns to invalidate
  int64 timestamp = 3;
  InvalidationType type = 4;  // PURGE, SOFT_PURGE, TAG_BASED
}

message InvalidationAck {
  string request_id = 1;
  string edge_server_id = 2;
  InvalidationStatus status = 3;
  int32 keys_invalidated = 4;
}

enum InvalidationType {
  PURGE = 0;         // Delete from cache immediately
  SOFT_PURGE = 1;    // Mark stale, revalidate on next request
  TAG_BASED = 2;     // Invalidate all content with specific tag
}
\`\`\`

**Edge Server Implementation**:

\`\`\`javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('cdn_control.proto');
const cdnControl = grpc.loadPackageDefinition(packageDefinition).CDNControl;

const client = new cdnControl.CDNControl(
  'control-plane.example.com:50051',
  grpc.credentials.createSsl()
);

// Open bidirectional stream
const stream = client.StreamInvalidations();

// Listen for invalidation requests from control plane
stream.on('data', async (request) => {
  console.log(\`Received invalidation: \${request.request_id}\`);
  
  let keysInvalidated = 0;
  
  for (const key of request.cache_keys) {
    if (request.type === 'PURGE') {
      // Delete from cache
      await cache.delete(key);
      keysInvalidated++;
    } else if (request.type === 'SOFT_PURGE') {
      // Mark as stale
      await cache.markStale(key);
      keysInvalidated++;
    } else if (request.type === 'TAG_BASED') {
      // Find all content with tag and invalidate
      const keys = await cache.findByTag(key);
      await cache.deleteMany(keys);
      keysInvalidated += keys.length;
    }
  }
  
  // Send acknowledgment back to control plane
  stream.write({
    request_id: request.request_id,
    edge_server_id: process.env.EDGE_ID,
    status: 'COMPLETED',
    keys_invalidated: keysInvalidated
  });
});

stream.on('error', (error) => {
  console.error('Stream error:', error);
  // Reconnect with exponential backoff
  setTimeout(() => reconnect(), 5000);
});

// Heartbeat to keep connection alive
setInterval(() => {
  stream.write({
    edge_server_id: process.env.EDGE_ID,
    status: 'HEARTBEAT'
  });
}, 30000);
\`\`\`

**Control Plane (Initiating Invalidation)**:

\`\`\`javascript
// When content is updated, broadcast invalidation to all edge servers
async function invalidateContent(cacheKeys) {
  const requestId = generateUUID();
  const request = {
    request_id: requestId,
    cache_keys: cacheKeys,
    timestamp: Date.now(),
    type: 'PURGE'
  };
  
  // Track acknowledgments
  const acks = new Map();
  const targetEdgeServers = await getActiveEdgeServers();
  
  // Send to all connected edge servers
  for (const [edgeId, stream] of connectedEdges) {
    stream.write(request);
    acks.set(edgeId, { sent: Date.now(), received: false });
  }
  
  // Wait for acks (or timeout after 10 seconds)
  await Promise.race([
    waitForAllAcks(acks),
    timeout(10000)
  ]);
  
  // Log completion
  const successCount = Array.from(acks.values()).filter(a => a.received).length;
  console.log(\`Invalidation completed: \${successCount}/\${targetEdgeServers.length} edge servers\`);
  
  // Alert if <95% success
  if (successCount / targetEdgeServers.length < 0.95) {
    sendAlert('Cache invalidation partial failure');
  }
}
\`\`\`

**Why gRPC Streaming**:
- **Real-time**: Sub-second invalidation propagation
- **Bidirectional**: Edge servers can ack completion
- **Efficient**: Single connection per edge server (not HTTP polling)
- **Reliable**: Built-in retries, flow control

---

**6. Real-Time Analytics: Protocol Buffers over HTTPS**

**Edge Server → Analytics Pipeline**:

\`\`\`protobuf
// analytics.proto
message AccessLog {
  string edge_server_id = 1;
  string client_ip = 2;
  string url = 3;
  int32 http_status = 4;
  int64 bytes_sent = 5;
  int32 response_time_ms = 6;
  string cache_status = 7;  // hit, miss, stale
  string user_agent = 8;
  string geo_country = 9;
  int64 timestamp = 10;
}

message AnalyticsBatch {
  repeated AccessLog logs = 1;
}
\`\`\`

**Edge Server Batching & Sending**:

\`\`\`javascript
const logs = [];

// Log each request
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    logs.push({
      edge_server_id: process.env.EDGE_ID,
      client_ip: req.ip,
      url: req.url,
      http_status: res.statusCode,
      bytes_sent: res.getHeader('content-length'),
      response_time_ms: Date.now() - start,
      cache_status: res.getHeader('x-cache-status'),
      user_agent: req.headers['user-agent'],
      geo_country: geoip.lookup(req.ip).country,
      timestamp: Date.now()
    });
  });
  
  next();
});

// Batch and send every 10 seconds
setInterval(async () => {
  if (logs.length === 0) return;
  
  const batch = { logs: logs.splice(0, 10000) };  // Max 10K per batch
  const serialized = AnalyticsBatch.encode(batch).finish();
  
  // Send compressed protobuf to analytics pipeline
  await fetch('https://analytics.example.com/ingest', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-protobuf',
      'Content-Encoding': 'gzip'
    },
    body: gzip(serialized)
  });
}, 10000);
\`\`\`

**Why Protocol Buffers**:
- **Compact**: 5-10x smaller than JSON
- **Fast**: ~10x faster serialization/deserialization
- **Schema evolution**: Add fields without breaking old clients
- **Cross-language**: Same .proto works in Go, Python, Node, etc.

---

**7. Performance Measurement**

**Key CDN Metrics**:

\`\`\`javascript
// Real User Monitoring (RUM)
const metrics = {
  // Latency
  ttfb: 'Time to First Byte',  // Target: <50ms
  ttlb: 'Time to Last Byte',    // Video download time
  
  // Cache efficiency
  cache_hit_ratio: 'Cache hits / Total requests',  // Target: >95%
  origin_offload: '1 - (Origin requests / Total requests)',  // Target: >95%
  
  // Quality of Experience
  video_startup_time: 'Time to start playback',  // Target: <2s
  rebuffer_ratio: 'Rebuffer events / Video views',  // Target: <1%
  
  // Throughput
  throughput: 'Mbps delivered to user',
  concurrent_streams: 'Active video streams',
  
  // Errors
  error_rate: '5xx / Total requests',  // Target: <0.01%
  origin_errors: '5xx from origin / Origin requests'
};
\`\`\`

**Performance Comparison**:

| **Metric** | **HTTP/1.1** | **HTTP/2** | **HTTP/3 (QUIC)** |
|------------|--------------|------------|-------------------|
| **TTFB (returning user)** | 150ms | 100ms | **50ms** (0-RTT) |
| **Video startup** | 3s | 2.5s | **1.8s** |
| **Rebuffer rate (mobile)** | 5% | 3% | **1.5%** (connection migration) |
| **Throughput (lossy network)** | 5 Mbps | 7 Mbps | **10 Mbps** (better congestion control) |

---

**8. Cache Coherence Strategy**

**Multi-Tier Caching**:

\`\`\`
Edge Cache (1000 servers)
    ↓
Origin Shield Cache (10 servers)
    ↓
Origin Cache (CDN-friendly caching headers)
\`\`\`

**Cache-Control Headers from Origin**:

\`\`\`http
HTTP/1.1 200 OK
Cache-Control: public, max-age=86400, s-maxage=604800, stale-while-revalidate=3600
CDN-Cache-Control: max-age=604800
Surrogate-Control: max-age=2592000
Vary: Accept-Encoding
ETag: "abc123"
X-Cache-Tag: movie:123, category:action
\`\`\`

**Tag-Based Invalidation**:

\`\`\`javascript
// When movie 123 is updated
await invalidateContent([
  '/video/movie123/*',           // Specific path
  'tag:movie:123',                // All content tagged with this movie
  'tag:category:action'           // All action movies
]);
\`\`\`

---

**Key Takeaways**:

1. **HTTP/3 for users**: 0-RTT, connection migration, better mobile performance
2. **HTTP/2 for backend**: Multiplexing, efficient for edge-to-origin communication
3. **gRPC for control plane**: Real-time cache invalidation with bidirectional streams
4. **Protocol Buffers for analytics**: Compact, fast, schema evolution
5. **Origin shield**: Reduces origin load by 95%+, provides second-tier caching
6. **Tag-based invalidation**: Efficient way to purge related content
7. **Performance**: HTTP/3 reduces TTFB by 50%, rebuffers by 66% on mobile
8. **Monitoring**: Cache hit ratio >95%, TTFB <50ms, rebuffer rate <1%`,
          keyPoints: [
            'HTTP/3 (QUIC) for user-facing: 0-RTT, connection migration, eliminates head-of-line blocking',
            'HTTP/2 for edge-to-origin: Multiplexing, header compression, efficient backend communication',
            'gRPC streaming for cache invalidation: Real-time bidirectional updates across 1000+ edge servers',
            'Protocol Buffers for analytics: 5-10x smaller than JSON, faster serialization',
            'Origin Shield: Collapses requests, reduces origin load by 95%, second-tier caching',
            'Tag-based cache invalidation: Purge related content efficiently (movie:123, category:action)',
            'Performance gains: 50% faster TTFB, 66% fewer mobile rebuffers with HTTP/3',
            'Key metrics: Cache hit ratio >95%, TTFB <50ms, rebuffer rate <1%',
          ],
        },
        {
          id: 'network-protocol-p2p-streaming',
          question:
            'Design a peer-to-peer (P2P) video streaming protocol for a live streaming platform to reduce CDN bandwidth costs by 70%. Explain how you would handle peer discovery, chunk distribution, incentive mechanisms, and fallback to CDN. Compare WebRTC Data Channels vs BitTorrent-style protocols, and discuss security, NAT traversal, and quality of experience trade-offs.',
          sampleAnswer: `**P2P Live Streaming Protocol Design**

**Goal**: Reduce CDN costs by 70% while maintaining <2s latency for live streams.

---

**1. High-Level Architecture**

\`\`\`
Live Stream Source
    ↓
CDN Edge Servers (30% traffic)
    ↓
Tracker/Signaling Server (WebSocket)
    ↓
P2P Mesh Network (70% traffic)
    ↓
Viewers (100M)
\`\`\`

**Hybrid Approach**: CDN + P2P
- **CDN**: Seed the first chunk, serve users without P2P capability
- **P2P**: Distribute most chunks peer-to-peer
- **Fallback**: Switch to CDN if P2P fails

---

**2. Protocol Selection: WebRTC Data Channels vs BitTorrent**

| **Aspect** | **WebRTC Data Channels** | **BitTorrent** |
|------------|--------------------------|----------------|
| **Latency** | **Low (1-2s)** | High (10-30s) |
| **Browser Support** | **Native** | Needs plugin |
| **NAT Traversal** | **Built-in (STUN/TURN)** | Requires setup |
| **Use Case** | **Live streaming** | File distribution |
| **Complexity** | Moderate | Low |
| **Connection Setup** | Fast (100-200ms) | Slow (handshake) |

**Choice**: **WebRTC Data Channels** for live streaming

**Why**:
- Native browser support (no plugins)
- Low latency (suitable for live)
- Built-in NAT traversal (STUN/TURN)
- Encrypted by default (DTLS)

---

**3. Peer Discovery & Signaling**

**Signaling Server** (WebSocket):

\`\`\`javascript
// Client connects to tracker
const ws = new WebSocket('wss://tracker.example.com');

ws.onopen = () => {
  // Register as peer for this stream
  ws.send(JSON.stringify({
    type: 'join',
    stream_id: 'stream-12345',
    peer_id: generatePeerId(),
    capabilities: {
      upload_bandwidth: measureUploadBandwidth(),
      webrtc_support: true,
      nat_type: detectNATType()
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'peers') {
    // Tracker sends list of peers
    const peers = message.peers;
    
    // Connect to 5-10 peers
    peers.slice(0, 10).forEach(peer => {
      connectToPeer(peer);
    });
  }
};
\`\`\`

**Tracker Server** (Assigns peers):

\`\`\`javascript
const streams = new Map(); // stream_id -> Set of peers

wss.on('connection', (ws) => {
  ws.on('message', (data) => {
    const message = JSON.parse(data);
    
    if (message.type === 'join') {
      const { stream_id, peer_id, capabilities } = message;
      
      if (!streams.has(stream_id)) {
        streams.set(stream_id, new Set());
      }
      
      const peers = streams.get(stream_id);
      peers.add({ peer_id, ws, capabilities, joined_at: Date.now() });
      
      // Send list of existing peers to new joiner
      const peerList = Array.from(peers)
        .filter(p => p.peer_id !== peer_id)
        .slice(0, 20); // Top 20 peers
      
      ws.send(JSON.stringify({
        type: 'peers',
        peers: peerList.map(p => ({
          peer_id: p.peer_id,
          upload_bandwidth: p.capabilities.upload_bandwidth
        }))
      }));
      
      // Notify existing peers about new joiner
      for (const peer of peers) {
        if (peer.peer_id !== peer_id) {
          peer.ws.send(JSON.stringify({
            type: 'peer_joined',
            peer: { peer_id, capabilities }
          }));
        }
      }
    }
  });
});
\`\`\`

**Peer Selection Strategy**:
1. **Geographic proximity**: Prefer peers in same region (lower latency)
2. **Upload bandwidth**: Prefer peers with high upload capacity
3. **Chunk availability**: Connect to peers with chunks you need
4. **Connection limit**: Maintain 5-10 active connections

---

**4. Chunk Distribution Protocol**

**Chunk Format**:

\`\`\`javascript
interface VideoChunk {
  stream_id: string;
  chunk_id: number;     // Incremental sequence number
  timestamp: number;    // Playback timestamp
  data: ArrayBuffer;    // Video data (HLS segment)
  duration: number;     // Chunk duration (2-4 seconds)
  size: number;         // Bytes
  hash: string;         // SHA-256 for integrity
}
\`\`\`

**P2P Connection Setup** (WebRTC):

\`\`\`javascript
async function connectToPeer(peerInfo) {
  const pc = new RTCPeerConnection({
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      {
        urls: 'turn:turn.example.com:3478',
        username: 'user',
        credential: 'pass'
      }
    ]
  });
  
  // Create data channel
  const dataChannel = pc.createDataChannel('video-chunks', {
    ordered: false,  // Don't wait for lost packets
    maxRetransmits: 0  // No retransmissions (live stream)
  });
  
  dataChannel.onopen = () => {
    console.log('Connected to peer:', peerInfo.peer_id);
    
    // Request needed chunks
    requestChunks(dataChannel, getNeededChunks());
  };
  
  dataChannel.onmessage = (event) => {
    const chunk = decodeChunk(event.data);
    
    // Verify chunk integrity
    if (verifyChunk(chunk)) {
      // Add to buffer
      chunkBuffer.set(chunk.chunk_id, chunk);
      
      // Forward to other peers (become a relay)
      relayChunkToOtherPeers(chunk);
    }
  };
  
  // ICE/SDP signaling via tracker
  pc.onicecandidate = (event) => {
    if (event.candidate) {
      ws.send(JSON.stringify({
        type: 'ice_candidate',
        target_peer: peerInfo.peer_id,
        candidate: event.candidate
      }));
    }
  };
  
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  
  ws.send(JSON.stringify({
    type: 'offer',
    target_peer: peerInfo.peer_id,
    sdp: offer
  }));
}
\`\`\`

**Chunk Request/Response**:

\`\`\`javascript
// Request chunks
function requestChunks(dataChannel, chunkIds) {
  dataChannel.send(JSON.stringify({
    type: 'request',
    chunk_ids: chunkIds
  }));
}

// Handle chunk requests
dataChannel.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'request') {
    // Send requested chunks
    message.chunk_ids.forEach(id => {
      const chunk = chunkBuffer.get(id);
      if (chunk) {
        dataChannel.send(encodeChunk(chunk));
        metrics.increment('chunks_uploaded');
      }
    });
  }
};
\`\`\`

---

**5. Chunk Selection Strategy (Rarest First)**

\`\`\`javascript
function getNeededChunks() {
  const currentPlayhead = video.currentTime;
  const chunkDuration = 2; // 2 seconds per chunk
  const currentChunkId = Math.floor(currentPlayhead / chunkDuration);
  
  // Priority:
  // 1. Next 3 chunks (critical for playback)
  // 2. Future 10 chunks (buffering)
  // 3. Rarest chunks in network
  
  const needed = [];
  
  // Critical chunks
  for (let i = 0; i < 3; i++) {
    const id = currentChunkId + i;
    if (!chunkBuffer.has(id)) {
      needed.push({ id, priority: 'critical', rarity: 0 });
    }
  }
  
  // Buffer chunks
  for (let i = 3; i < 10; i++) {
    const id = currentChunkId + i;
    if (!chunkBuffer.has(id)) {
      const rarity = getChunkRarity(id);
      needed.push({ id, priority: 'buffer', rarity });
    }
  }
  
  // Sort by priority then rarity (rarest first)
  return needed
    .sort((a, b) => {
      if (a.priority === 'critical' && b.priority !== 'critical') return -1;
      if (a.priority !== 'critical' && b.priority === 'critical') return 1;
      return b.rarity - a.rarity;
    })
    .map(c => c.id);
}

function getChunkRarity(chunkId) {
  // Query peers for chunk availability
  let peersWithChunk = 0;
  
  for (const peer of connectedPeers) {
    if (peer.availableChunks.has(chunkId)) {
      peersWithChunk++;
    }
  }
  
  return 1 / (peersWithChunk + 1);  // Higher = rarer
}
\`\`\`

---

**6. Incentive Mechanism (Tit-for-Tat)**

**Problem**: Leechers (download but don't upload) hurt the network.

**Solution**: BitTorrent-style reciprocity

\`\`\`javascript
class PeerManager {
  constructor() {
    this.peers = new Map();
    this.uploadedToPeer = new Map();
    this.downloadedFromPeer = new Map();
  }
  
  recordUpload(peerId, bytes) {
    this.uploadedToPeer.set(
      peerId,
      (this.uploadedToPeer.get(peerId) || 0) + bytes
    );
  }
  
  recordDownload(peerId, bytes) {
    this.downloadedFromPeer.set(
      peerId,
      (this.downloadedFromPeer.get(peerId) || 0) + bytes
    );
  }
  
  // Periodically evaluate peers (every 30 seconds)
  evaluatePeers() {
    const ratios = new Map();
    
    for (const peer of this.peers.values()) {
      const uploaded = this.uploadedToPeer.get(peer.id) || 0;
      const downloaded = this.downloadedFromPeer.get(peer.id) || 0;
      
      // Sharing ratio
      const ratio = downloaded > 0 ? uploaded / downloaded : 0;
      ratios.set(peer.id, ratio);
    }
    
    // Unchoke top 4 contributors
    const topPeers = Array.from(ratios.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([peerId]) => peerId);
    
    // Unchoke 1 random peer (optimistic unchoking)
    const randomPeer = getRandomPeer(this.peers);
    topPeers.push(randomPeer.id);
    
    // Update peer states
    for (const peer of this.peers.values()) {
      if (topPeers.includes(peer.id)) {
        peer.unchoke();  // Allow downloading from this peer
      } else {
        peer.choke();    // Stop sending data to this peer
      }
    }
  }
}
\`\`\`

**Visualization**:

\`\`\`
Peer A (uploads 10 MB, downloads 5 MB) → Ratio: 2.0 → ✅ Unchoked
Peer B (uploads 8 MB, downloads 4 MB)  → Ratio: 2.0 → ✅ Unchoked
Peer C (uploads 2 MB, downloads 10 MB) → Ratio: 0.2 → ❌ Choked
Peer D (uploads 0 MB, downloads 5 MB)  → Ratio: 0.0 → ❌ Choked
Peer E (random)                         → Ratio: N/A → ✅ Unchoked (optimistic)
\`\`\`

---

**7. CDN Fallback**

**When to fall back to CDN**:
1. **P2P unavailable**: No peers or poor connectivity
2. **Buffer starvation**: Not receiving chunks fast enough
3. **High latency**: P2P latency >2s
4. **Quality degradation**: Excessive buffering or rebuffering

\`\`\`javascript
class VideoPlayer {
  constructor() {
    this.p2pEnabled = true;
    this.cdnFallbackActive = false;
    this.bufferHealth = 10; // seconds of buffer
  }
  
  async loadChunk(chunkId) {
    // Try P2P first
    if (this.p2pEnabled && !this.cdnFallbackActive) {
      const chunk = await this.loadFromP2P(chunkId, { timeout: 2000 });
      
      if (chunk) {
        this.bufferHealth += chunk.duration;
        return chunk;
      }
    }
    
    // Fallback to CDN
    console.log('Falling back to CDN for chunk', chunkId);
    const chunk = await this.loadFromCDN(chunkId);
    
    // Check if we should disable P2P
    if (this.bufferHealth < 3) {
      this.cdnFallbackActive = true;
      setTimeout(() => {
        this.cdnFallbackActive = false;  // Retry P2P in 30s
      }, 30000);
    }
    
    return chunk;
  }
  
  async loadFromP2P(chunkId, options) {
    return Promise.race([
      this.requestChunkFromPeers(chunkId),
      timeout(options.timeout)
    ]).catch(() => null);
  }
  
  async loadFromCDN(chunkId) {
    const response = await fetch(\`https://cdn.example.com/chunks/\${chunkId}.ts\`);
    return response.arrayBuffer();
  }
}
\`\`\`

---

**8. NAT Traversal (STUN/TURN)**

**NAT Types**:
1. **Full Cone**: Easy, direct P2P works
2. **Restricted Cone**: Moderate, needs STUN
3. **Port Restricted Cone**: Moderate, needs STUN
4. **Symmetric**: Hard, needs TURN relay

**Solution**: Use STUN first, fallback to TURN

\`\`\`javascript
const iceServers = [
  // Public STUN servers (free)
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  
  // TURN relay servers (costs money)
  {
    urls: 'turn:turn.example.com:3478',
    username: getTurnCredentials().username,
    credential: getTurnCredentials().password
  }
];

const pc = new RTCPeerConnection({ iceServers });

pc.onicecandidate = (event) => {
  if (event.candidate) {
    console.log('ICE candidate type:', event.candidate.type);
    // host, srflx (STUN), relay (TURN)
    
    if (event.candidate.type === 'relay') {
      metrics.increment('turn_relay_used');
      // TURN relay costs money - track usage
    }
  }
};
\`\`\`

**TURN Relay Costs**:
- TURN used for ~15-20% of peers (symmetric NAT)
- Relay bandwidth: ~30% of CDN bandwidth
- **Net savings**: 70% P2P + 15% TURN + 15% CDN = **55% CDN cost reduction**

---

**9. Security**

**Threats**:
1. **Malicious chunks**: Peer sends corrupted data
2. **Sybil attack**: Single entity creates many fake peers
3. **Eclipse attack**: Isolate peer from honest peers
4. **Privacy**: IP addresses visible to peers

**Mitigations**:

\`\`\`javascript
// 1. Chunk verification (SHA-256 hash)
function verifyChunk(chunk) {
  const hash = sha256(chunk.data);
  return hash === chunk.hash;
}

// 2. Peer reputation system
class ReputationSystem {
  constructor() {
    this.scores = new Map();
  }
  
  recordBadChunk(peerId) {
    const score = this.scores.get(peerId) || 100;
    this.scores.set(peerId, score - 10);
    
    if (score - 10 < 50) {
      blockPeer(peerId);
    }
  }
  
  recordGoodChunk(peerId) {
    const score = this.scores.get(peerId) || 100;
    this.scores.set(peerId, Math.min(score + 1, 100));
  }
}

// 3. Sybil resistance (proof of work)
function generatePeerId() {
  let nonce = 0;
  let hash;
  
  do {
    hash = sha256(\`\${Date.now()}-\${nonce}\`);
    nonce++;
  } while (!hash.startsWith('0000')); // Require 4 leading zeros
  
  return hash;
}

// 4. Privacy (use TURN relay to hide IP)
const pc = new RTCPeerConnection({
  iceTransportPolicy: 'relay'  // Force TURN (hide real IP)
});
\`\`\`

---

**10. Quality of Experience**

**Metrics**:

\`\`\`javascript
const qoe = {
  startup_time: 'Time to first frame',  // Target: <2s
  rebuffer_rate: 'Rebuffer events / minute',  // Target: <0.1
  average_bitrate: 'Average video quality',  // Target: 1080p
  p2p_ratio: 'P2P bytes / Total bytes',  // Target: >70%
  cdn_fallback_rate: 'CDN requests / Total',  // Target: <30%
  peer_count: 'Active peer connections',  // Target: 5-10
  upload_contribution: 'Bytes uploaded',  // Encourage sharing
};
\`\`\`

**User Experience**:
- **Transparent**: User doesn't notice P2P vs CDN
- **Adaptive**: Falls back to CDN seamlessly
- **Privacy-aware**: Option to disable P2P
- **Bandwidth control**: Limit upload/download rates

---

**Key Takeaways**:

1. **WebRTC Data Channels**: Best for live P2P streaming (low latency, native support)
2. **Hybrid CDN + P2P**: CDN seeds, P2P distributes (70% cost savings)
3. **Rarest-first strategy**: Prioritize chunks fewer peers have
4. **Tit-for-tat incentives**: Reward uploaders, throttle leechers
5. **CDN fallback**: Seamless switch when P2P fails
6. **STUN/TURN**: NAT traversal (~15% peers need TURN relay)
7. **Security**: Chunk hashing, reputation system, proof-of-work peer IDs
8. **Net savings**: 55-70% CDN cost reduction after TURN relay costs`,
          keyPoints: [
            'WebRTC Data Channels over BitTorrent: Low latency (1-2s), native browser support, built-in NAT traversal',
            'Hybrid CDN + P2P: CDN seeds first chunk and serves fallback, P2P handles 70% of traffic',
            'Peer discovery: WebSocket tracker assigns 5-10 peers based on geography, bandwidth, chunk availability',
            'Rarest-first chunk selection: Prioritize chunks fewer peers have to maximize distribution',
            'Tit-for-tat incentives: Unchoke top 4 contributors + 1 random peer to prevent leeching',
            'NAT traversal: STUN for most peers (~85%), TURN relay for symmetric NAT (~15%)',
            'Security: SHA-256 chunk verification, reputation system, proof-of-work peer IDs',
            'Cost savings: 70% P2P - 15% TURN = 55% net CDN cost reduction',
          ],
        },
      ],
    },
    {
      id: 'rate-limiting',
      title: 'Rate Limiting & Throttling',
      content: `Rate limiting and throttling are critical for protecting APIs and services from abuse, ensuring fair resource usage, and maintaining system stability. This section covers algorithms, implementation patterns, and distributed rate limiting strategies.
    
    ## What is Rate Limiting?
    
    **Rate Limiting** restricts the number of requests a client can make to an API within a time window.
    
    **Goals**:
    - Prevent abuse and DDoS attacks
    - Ensure fair resource allocation
    - Protect backend services from overload
    - Enforce pricing tiers (free vs paid)
    
    **Example**:
    \`\`\`
    User can make:
    - 100 requests per minute (free tier)
    - 1000 requests per minute (pro tier)
    - Unlimited requests (enterprise tier)
    \`\`\`
    
    ---
    
    ## Rate Limiting Algorithms
    
    ### **1. Token Bucket**
    
    **Most Popular Algorithm** - Used by AWS, Stripe, Shopify
    
    **How it Works**:
    - Bucket has capacity of N tokens
    - Tokens added at rate R per second
    - Each request consumes 1 token
    - Request rejected if bucket empty
    
    **Visualization**:
    \`\`\`
    Bucket capacity: 10 tokens
    Refill rate: 2 tokens/second
    
    Time 0s: [**********] (10 tokens) → Request (9 tokens)
    Time 1s: [**********] (10 tokens, refilled)
    Time 2s: [**********] (10 tokens)
    
    Burst: Can use all 10 immediately
    Sustained: Limited to 2/second long-term
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class TokenBucket {
      constructor(capacity, refillRate) {
        this.capacity = capacity;
        this.refillRate = refillRate; // tokens per second
        this.tokens = capacity;
        this.lastRefill = Date.now();
      }
      
      tryConsume(tokens = 1) {
        this.refill();
        
        if (this.tokens >= tokens) {
          this.tokens -= tokens;
          return true;
        }
        
        return false;
      }
      
      refill() {
        const now = Date.now();
        const timePassed = (now - this.lastRefill) / 1000; // seconds
        const tokensToAdd = timePassed * this.refillRate;
        
        this.tokens = Math.min(this.capacity, this.tokens + tokensToAdd);
        this.lastRefill = now;
      }
      
      getWaitTime() {
        if (this.tokens >= 1) return 0;
        return (1 - this.tokens) / this.refillRate * 1000; // ms
      }
    }
    
    // Usage
    const bucket = new TokenBucket(100, 2); // 100 capacity, 2/sec
    
    app.use((req, res, next) => {
      const userId = req.user.id;
      const userBucket = buckets.get(userId) || new TokenBucket(100, 2);
      
      if (userBucket.tryConsume()) {
        buckets.set(userId, userBucket);
        next();
      } else {
        const waitTime = Math.ceil(userBucket.getWaitTime() / 1000);
        res.status(429).json({
          error: 'Rate limit exceeded',
          retryAfter: waitTime
        });
      }
    });
    \`\`\`
    
    **Pros**:
    - Allows bursts (good for variable traffic)
    - Memory efficient
    - Smooth rate limiting
    
    **Cons**:
    - Slightly complex to implement
    - Need to track last refill time
    
    ---
    
    ### **2. Leaky Bucket**
    
    **Smooth, Constant Rate** - Used for traffic shaping
    
    **How it Works**:
    - Requests added to queue (bucket)
    - Processed at fixed rate
    - Overflow requests rejected
    
    **Visualization**:
    \`\`\`
    Requests → [Queue: 5/10] → Process at 2/sec → Backend
                  ↓ (full)
               Reject
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class LeakyBucket {
      constructor(capacity, leakRate) {
        this.capacity = capacity;
        this.leakRate = leakRate; // requests per second
        this.queue = [];
        this.processing = false;
      }
      
      async tryAdd(request) {
        if (this.queue.length >= this.capacity) {
          return false; // Bucket full
        }
        
        this.queue.push(request);
        
        if (!this.processing) {
          this.startLeaking();
        }
        
        return true;
      }
      
      startLeaking() {
        this.processing = true;
        
        const interval = 1000 / this.leakRate; // ms between requests
        
        const leak = setInterval(() => {
          if (this.queue.length === 0) {
            clearInterval(leak);
            this.processing = false;
            return;
          }
          
          const request = this.queue.shift();
          this.processRequest(request);
        }, interval);
      }
      
      async processRequest(request) {
        // Process request at constant rate
        await handleRequest(request);
      }
    }
    \`\`\`
    
    **Pros**:
    - Smooth, constant output rate
    - Good for protecting downstream services
    
    **Cons**:
    - No bursts allowed
    - Requests delayed (queued)
    - More memory (stores queue)
    
    ---
    
    ### **3. Fixed Window**
    
    **Simple but Flawed**
    
    **How it Works**:
    - Count requests in fixed time windows
    - Reset counter at window boundary
    
    **Example**:
    \`\`\`
    Window: 1 minute
    Limit: 100 requests
    
    12:00:00 - 12:00:59 → 100 requests (allowed)
    12:01:00 - 12:01:59 → Counter resets, 100 more allowed
    \`\`\`
    
    **Problem: Burst at Window Boundaries**:
    \`\`\`
    12:00:50 → 100 requests (allowed)
    12:01:01 → 100 requests (allowed)
    Total: 200 requests in 11 seconds! (Burst at boundary)
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class FixedWindow {
      constructor(limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.counters = new Map(); // userId -> {count, windowStart}
      }
      
      tryRequest(userId) {
        const now = Date.now();
        const userData = this.counters.get(userId) || { count: 0, windowStart: now };
        
        // Check if we're in a new window
        if (now - userData.windowStart >= this.windowSizeMs) {
          userData.count = 0;
          userData.windowStart = now;
        }
        
        if (userData.count < this.limit) {
          userData.count++;
          this.counters.set(userId, userData);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Very simple
    - Memory efficient
    
    **Cons**:
    - Burst problem at boundaries
    - Not accurate
    
    ---
    
    ### **4. Sliding Window Log**
    
    **Accurate but Memory Intensive**
    
    **How it Works**:
    - Store timestamp of each request
    - Count requests in rolling window
    - Remove old timestamps
    
    **Implementation**:
    \`\`\`javascript
    class SlidingWindowLog {
      constructor(limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.logs = new Map(); // userId -> [timestamps]
      }
      
      tryRequest(userId) {
        const now = Date.now();
        const userLog = this.logs.get(userId) || [];
        
        // Remove timestamps outside window
        const windowStart = now - this.windowSizeMs;
        const validLog = userLog.filter(timestamp => timestamp > windowStart);
        
        if (validLog.length < this.limit) {
          validLog.push(now);
          this.logs.set(userId, validLog);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Very accurate
    - No burst problem
    
    **Cons**:
    - Memory intensive (stores all timestamps)
    - Expensive (filter on every request)
    
    ---
    
    ### **5. Sliding Window Counter** ⭐
    
    **Best of Both Worlds** - Combines fixed window + sliding window
    
    **How it Works**:
    - Use two fixed windows
    - Estimate count in sliding window using weighted average
    
    **Formula**:
    \`\`\`
    Current window: 70% complete
    Previous window count: 100
    Current window count: 40
    
    Estimated count = (100 × 0.3) + (40 × 1.0) = 30 + 40 = 70
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class SlidingWindowCounter {
      constructor(limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.counters = new Map(); // userId -> {current, previous, windowStart}
      }
      
      tryRequest(userId) {
        const now = Date.now();
        const userData = this.counters.get(userId) || {
          current: 0,
          previous: 0,
          windowStart: now
        };
        
        const elapsed = now - userData.windowStart;
        
        // New window?
        if (elapsed >= this.windowSizeMs) {
          userData.previous = userData.current;
          userData.current = 0;
          userData.windowStart = now;
        }
        
        // Calculate weighted count
        const windowProgress = elapsed / this.windowSizeMs;
        const previousWeight = 1 - windowProgress;
        const estimatedCount = 
          (userData.previous * previousWeight) + userData.current;
        
        if (estimatedCount < this.limit) {
          userData.current++;
          this.counters.set(userId, userData);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Accurate (no burst at boundaries)
    - Memory efficient (only 2 counters)
    - Fast (no filtering)
    
    **Cons**:
    - Slightly more complex
    
    **Recommended for most use cases!**
    
    ---
    
    ## Distributed Rate Limiting
    
    **Challenge**: Multiple API servers need to share rate limit state
    
    ### **Solution 1: Redis with Sliding Window**
    
    \`\`\`javascript
    const Redis = require('ioredis');
    const redis = new Redis();
    
    async function rateLimitRedis(userId, limit, windowSec) {
      const key = \`rate_limit:\${userId}\`;
      const now = Date.now();
      const windowStart = now - (windowSec * 1000);
      
      // Use Redis sorted set (score = timestamp)
      const pipeline = redis.pipeline();
      
      // Remove old entries
      pipeline.zremrangebyscore(key, '-inf', windowStart);
      
      // Count requests in window
      pipeline.zcard(key);
      
      // Add current request
      pipeline.zadd(key, now, \`\${now}-\${Math.random()}\`);
      
      // Set expiry
      pipeline.expire(key, windowSec * 2);
      
      const results = await pipeline.exec();
      const count = results[1][1]; // Count from zcard
      
      if (count < limit) {
        return { allowed: true, remaining: limit - count - 1 };
      } else {
        // Remove the request we just added
        await redis.zrem(key, \`\${now}-\${Math.random()}\`);
        return { allowed: false, remaining: 0 };
      }
    }
    
    // Usage
    app.use(async (req, res, next) => {
      const result = await rateLimitRedis(req.user.id, 100, 60);
      
      if (result.allowed) {
        res.setHeader('X-RateLimit-Remaining', result.remaining);
        next();
      } else {
        res.status(429).json({ error: 'Rate limit exceeded' });
      }
    });
    \`\`\`
    
    ### **Solution 2: Redis with Token Bucket (More Efficient)**
    
    \`\`\`javascript
    async function tokenBucketRedis(userId, capacity, refillRate) {
      const key = \`token_bucket:\${userId}\`;
      
      // Lua script for atomic token bucket operation
      const script = \`
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refillRate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'lastRefill')
        local tokens = tonumber(bucket[1]) or capacity
        local lastRefill = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local timePassed = (now - lastRefill) / 1000
        local tokensToAdd = timePassed * refillRate
        tokens = math.min(capacity, tokens + tokensToAdd)
        
        -- Try to consume
        if tokens >= 1 then
          tokens = tokens - 1
          redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
          redis.call('EXPIRE', key, 3600)
          return {1, math.floor(tokens)}
        else
          return {0, 0}
        end
      \`;
      
      const result = await redis.eval(
        script,
        1,
        key,
        capacity,
        refillRate,
        Date.now()
      );
      
      return {
        allowed: result[0] === 1,
        remaining: result[1]
      };
    }
    \`\`\`
    
    **Why Lua Script?**
    - Atomic operation (no race conditions)
    - Single round-trip to Redis
    - Consistent across all servers
    
    ---
    
    ## Rate Limiting Patterns
    
    ### **1. Per-User Rate Limiting**
    
    \`\`\`javascript
    app.use(async (req, res, next) => {
      const userId = req.user.id;
      const result = await rateLimit(userId, 100, 60);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Too many requests' });
      }
    });
    \`\`\`
    
    ### **2. Per-IP Rate Limiting** (for anonymous users)
    
    \`\`\`javascript
    app.use(async (req, res, next) => {
      const ip = req.ip || req.connection.remoteAddress;
      const result = await rateLimit(ip, 20, 60);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Too many requests from this IP' });
      }
    });
    \`\`\`
    
    ### **3. Per-Endpoint Rate Limiting**
    
    \`\`\`javascript
    // Different limits for different endpoints
    const limits = {
      'POST /api/login': { limit: 5, window: 60 }, // 5 per minute
      'GET /api/users': { limit: 100, window: 60 }, // 100 per minute
      'POST /api/upload': { limit: 10, window: 3600 } // 10 per hour
    };
    
    app.use(async (req, res, next) => {
      const endpoint = \`\${req.method} \${req.path}\`;
      const config = limits[endpoint] || { limit: 60, window: 60 };
      
      const key = \`\${req.user.id}:\${endpoint}\`;
      const result = await rateLimit(key, config.limit, config.window);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Rate limit exceeded for this endpoint' });
      }
    });
    \`\`\`
    
    ### **4. Tiered Rate Limiting**
    
    \`\`\`javascript
    const tiers = {
      free: { limit: 100, window: 3600 },
      pro: { limit: 1000, window: 3600 },
      enterprise: { limit: 10000, window: 3600 }
    };
    
    app.use(async (req, res, next) => {
      const userTier = req.user.tier || 'free';
      const config = tiers[userTier];
      
      const result = await rateLimit(req.user.id, config.limit, config.window);
      
      res.setHeader('X-RateLimit-Limit', config.limit);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({
          error: 'Rate limit exceeded',
          upgrade: 'Upgrade to Pro for higher limits'
        });
      }
    });
    \`\`\`
    
    ---
    
    ## Response Headers
    
    **Standard Headers** (from [RFC 6585](https://tools.ietf.org/html/rfc6585)):
    
    \`\`\`
    X-RateLimit-Limit: 100          # Max requests per window
    X-RateLimit-Remaining: 75       # Requests left in current window
    X-RateLimit-Reset: 1699564800   # Unix timestamp when limit resets
    Retry-After: 30                 # Seconds until retry (when rate limited)
    \`\`\`
    
    **Example**:
    \`\`\`javascript
    app.use(async (req, res, next) => {
      const result = await tokenBucket.tryConsume(req.user.id);
      
      res.setHeader('X-RateLimit-Limit', 100);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      res.setHeader('X-RateLimit-Reset', result.reset);
      
      if (result.allowed) {
        next();
      } else {
        res.setHeader('Retry-After', result.retryAfter);
        res.status(429).json({
          error: 'Too Many Requests',
          retryAfter: result.retryAfter
        });
      }
    });
    \`\`\`
    
    ---
    
    ## Common Mistakes
    
    ### **❌ Mistake 1: Rate Limiting After Authentication**
    
    \`\`\`javascript
    // Bad: Attacker can DDoS by sending invalid credentials
    app.post('/api/login', authenticate, rateLimit, (req, res) => {
      // Login logic
    });
    
    // Good: Rate limit BEFORE authentication
    app.post('/api/login', rateLimit, authenticate, (req, res) => {
      // Login logic
    });
    \`\`\`
    
    ### **❌ Mistake 2: Not Handling Clock Drift**
    
    \`\`\`javascript
    // Bad: Uses server timestamp (clock drift issues)
    const now = Date.now();
    
    // Good: Use Redis TIME command for distributed consistency
    const [seconds, microseconds] = await redis.time();
    const now = seconds * 1000 + Math.floor(microseconds / 1000);
    \`\`\`
    
    ### **❌ Mistake 3: No Exponential Backoff for Retries**
    
    \`\`\`javascript
    // Bad: Client retries immediately
    if (response.status === 429) {
      retry();
    }
    
    // Good: Exponential backoff
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      const delay = retryAfter ? parseInt(retryAfter) * 1000 : 1000;
      setTimeout(() => retry(), delay * Math.pow(2, attempts));
    }
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **Token Bucket recommended** for most use cases (allows bursts, memory efficient)
    2. **Sliding Window Counter** for accurate rate limiting without burst problems
    3. **Redis + Lua scripts** for distributed rate limiting (atomic operations)
    4. **Rate limit BEFORE authentication** to prevent DDoS
    5. **Different limits for different endpoints** (login: 5/min, read: 1000/min)
    6. **Tiered limits** enforce pricing (free: 100/hr, pro: 1000/hr)
    7. **Return standard headers** (X-RateLimit-*, Retry-After)
    8. **Per-user AND per-IP** rate limiting for security
    9. **Monitor rate limit hits** to detect abuse or legitimate high usage
    10. **Graceful degradation**: Return 429 with clear error message and retry guidance`,
      multipleChoice: [
        {
          id: 'rate-limit-algorithm',
          question:
            'Which rate limiting algorithm allows burst traffic while maintaining a long-term average rate?',
          options: [
            'Fixed Window Counter',
            'Leaky Bucket',
            'Token Bucket',
            'Sliding Window Log',
          ],
          correctAnswer: 2,
          explanation:
            "Token Bucket allows burst traffic because clients can accumulate tokens up to the bucket capacity and use them all at once. However, the long-term rate is limited by the refill rate. Leaky Bucket processes at a constant rate (no bursts). Fixed Window has boundary burst problems. Sliding Window Log is accurate but doesn't specifically enable bursts.",
        },
        {
          id: 'rate-limit-distributed',
          question:
            'Why is a Lua script preferred over multiple Redis commands for distributed rate limiting?',
          options: [
            'Lua scripts are faster than Redis commands',
            'Lua scripts provide atomic execution, preventing race conditions across multiple API servers',
            'Lua scripts use less memory than Redis data structures',
            'Lua scripts automatically distribute across Redis cluster nodes',
          ],
          correctAnswer: 1,
          explanation:
            "Lua scripts execute atomically in Redis, meaning all operations complete without interruption. This prevents race conditions when multiple API servers check and update rate limit counters simultaneously. Without atomicity, two servers could both check the counter (e.g., 99 requests), both see it's below the limit (100), and both allow the request, resulting in 101 requests. Lua scripts ensure the check-and-increment happens atomically.",
        },
        {
          id: 'rate-limit-429',
          question:
            'A user receives a 429 Too Many Requests response. Which HTTP header should the API return to indicate when the user can retry?',
          options: [
            'X-RateLimit-Reset',
            'Retry-After',
            'X-RateLimit-Remaining',
            'Cache-Control',
          ],
          correctAnswer: 1,
          explanation:
            'Retry-After (RFC 6585) specifically indicates when the client should retry after being rate limited, either as seconds (e.g., "60") or HTTP date. X-RateLimit-Reset shows when the rate limit window resets, X-RateLimit-Remaining shows remaining requests, and Cache-Control is for caching, not rate limiting.',
        },
        {
          id: 'rate-limit-placement',
          question:
            'Where should rate limiting middleware be placed in the request processing pipeline to best protect against authentication brute-force attacks?',
          options: [
            'After authentication, to only limit authenticated users',
            'Before authentication, to limit all login attempts including invalid ones',
            'After database queries, to limit only successful logins',
            'In a separate microservice, to avoid impacting API performance',
          ],
          correctAnswer: 1,
          explanation:
            'Rate limiting must occur BEFORE authentication to protect against brute-force attacks. If rate limiting happens after authentication, an attacker can make unlimited login attempts with invalid credentials, potentially guessing passwords or causing a DDoS. Rate limiting before authentication (by IP address) limits all login attempts, preventing such attacks.',
        },
        {
          id: 'rate-limit-boundary-problem',
          question:
            'The Fixed Window Counter algorithm has a "boundary problem" where users can potentially send double the rate limit in a short period. Which algorithm solves this issue while remaining memory efficient?',
          options: [
            'Leaky Bucket',
            'Token Bucket',
            'Sliding Window Counter',
            'Fixed Window Log',
          ],
          correctAnswer: 2,
          explanation:
            "Sliding Window Counter solves the boundary problem by using a weighted average of the previous and current window counts, providing accurate rate limiting without bursts at boundaries. It's memory efficient (stores only 2 counters per user). Leaky Bucket prevents bursts but queues requests. Token Bucket allows bursts. Fixed Window Log is accurate but memory intensive (stores all timestamps).",
        },
      ],
      quiz: [
        {
          id: 'rate-limit-distributed-system',
          question:
            'Design a distributed rate limiting system for an API with 100 servers handling 1 million requests per second. The system must support per-user limits (10,000 req/hr), per-IP limits (1,000 req/hr), and per-endpoint limits. Explain your architecture, choice of algorithm, Redis configuration, handling of edge cases (clock drift, Redis failures), and monitoring strategy.',
          sampleAnswer: `**Distributed Rate Limiting System Design**
    
    **1. Architecture**
    
    \`\`\`
    Client Request
        ↓
    Load Balancer
        ↓
    API Server (1 of 100)
        ↓
    Rate Limit Middleware
        ↓
    Redis Cluster (rate limit state)
        ↓
    Backend Services
    \`\`\`
    
    **2. Algorithm Choice: Token Bucket with Redis**
    
    **Why Token Bucket**:
    - Allows bursts (better UX)
    - Memory efficient (2 values per key)
    - Natural fit for time-based limits
    - Fast (constant time operations)
    
    **3. Redis Cluster Configuration**
    
    \`\`\`yaml
    # redis.conf
    cluster-enabled yes
    cluster-node-timeout 5000
    
    # Persistence (for rate limit state)
    save 900 1        # Save after 900 sec if 1 key changed
    appendonly yes    # AOF for durability
    appendfsync everysec
    
    # Memory
    maxmemory 16gb
    maxmemory-policy allkeys-lru  # Evict old rate limit keys
    
    # Replication
    min-replicas-to-write 1  # Require 1 replica acknowledgment
    \`\`\`
    
    **Cluster Setup**:
    - 6 Redis nodes (3 masters + 3 replicas)
    - Hash slot distribution: 16384 slots / 3 masters
    - Each master handles ~333K requests/sec
    - Replication for high availability
    
    **4. Implementation**
    
    **Lua Script** (atomic token bucket):
    \`\`\`lua
    -- rate_limit.lua
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4]) or 1
    
    -- Get current state
    local tokens = tonumber(redis.call('HGET', key, 'tokens'))
    local last_refill = tonumber(redis.call('HGET', key, 'last_refill'))
    
    -- Initialize if not exists
    if not tokens then
      tokens = capacity
      last_refill = now
    end
    
    -- Refill tokens
    local time_passed = (now - last_refill) / 1000  -- seconds
    local tokens_to_add = time_passed * refill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    
    -- Try to consume
    if tokens >= cost then
      tokens = tokens - cost
      redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
      redis.call('EXPIRE', key, 7200)  -- 2 hour TTL
      
      return {1, math.floor(tokens), 0}  -- allowed, remaining, retry_after
    else
      -- Calculate retry after (seconds)
      local tokens_needed = cost - tokens
      local retry_after = math.ceil(tokens_needed / refill_rate)
      
      return {0, 0, retry_after}  -- denied, remaining=0, retry_after
    end
    \`\`\`
    
    **Node.js Middleware**:
    \`\`\`javascript
    const Redis = require('ioredis');
    const fs = require('fs');
    
    // Redis cluster client
    const redis = new Redis.Cluster([
      { host: 'redis-1', port: 6379 },
      { host: 'redis-2', port: 6379 },
      { host: 'redis-3', port: 6379 }
    ], {
      redisOptions: {
        password: process.env.REDIS_PASSWORD
      }
    });
    
    // Load Lua script
    const rateLimitScript = fs.readFileSync('rate_limit.lua', 'utf8');
    const scriptSha = await redis.script('LOAD', rateLimitScript);
    
    // Rate limit configuration
    const limits = {
      perUser: { capacity: 10000, refillRate: 2.78 }, // 10K per hour = 2.78/sec
      perIP: { capacity: 1000, refillRate: 0.278 },   // 1K per hour = 0.278/sec
      perEndpoint: {
        'POST /api/login': { capacity: 5, refillRate: 0.083 }, // 5 per minute
        'POST /api/upload': { capacity: 10, refillRate: 0.0028 } // 10 per hour
      }
    };
    
    async function rateLimit(identifier, config) {
      const key = \`rate_limit:\${identifier}\`;
      
      try {
        // Use Redis TIME for consistency across servers (handles clock drift)
        const [seconds, microseconds] = await redis.time();
        const now = seconds * 1000 + Math.floor(microseconds / 1000);
        
        const result = await redis.evalsha(
          scriptSha,
          1,
          key,
          config.capacity,
          config.refillRate,
          now,
          1 // cost
        );
        
        return {
          allowed: result[0] === 1,
          remaining: result[1],
          retryAfter: result[2]
        };
      } catch (error) {
        // Fail open on Redis errors (allow request but log)
        logger.error('Rate limit check failed', { identifier, error });
        return { allowed: true, remaining: -1, retryAfter: 0 };
      }
    }
    
    // Middleware
    app.use(async (req, res, next) => {
      const userId = req.user?.id;
      const ip = req.ip || req.connection.remoteAddress;
      const endpoint = \`\${req.method} \${req.path}\`;
      
      // Check multiple limits
      const checks = [];
      
      // Per-user limit (if authenticated)
      if (userId) {
        checks.push({
          name: 'per-user',
          identifier: \`user:\${userId}\`,
          config: limits.perUser
        });
      }
      
      // Per-IP limit
      checks.push({
        name: 'per-ip',
        identifier: \`ip:\${ip}\`,
        config: limits.perIP
      });
      
      // Per-endpoint limit
      const endpointConfig = limits.perEndpoint[endpoint];
      if (endpointConfig) {
        checks.push({
          name: 'per-endpoint',
          identifier: \`\${userId || ip}:\${endpoint}\`,
          config: endpointConfig
        });
      }
      
      // Run all checks in parallel
      const results = await Promise.all(
        checks.map(check => 
          rateLimit(check.identifier, check.config)
            .then(result => ({ ...result, check }))
        )
      );
      
      // Find first limit exceeded
      const blocked = results.find(r => !r.allowed);
      
      if (blocked) {
        const config = blocked.check.config;
        
        res.setHeader('X-RateLimit-Limit', config.capacity);
        res.setHeader('X-RateLimit-Remaining', 0);
        res.setHeader('Retry-After', blocked.retryAfter);
        
        return res.status(429).json({
          error: 'Too Many Requests',
          limit: blocked.check.name,
          retryAfter: blocked.retryAfter
        });
      }
      
      // Set headers for successful request
      const userLimit = results.find(r => r.check.name === 'per-user');
      if (userLimit) {
        res.setHeader('X-RateLimit-Limit', limits.perUser.capacity);
        res.setHeader('X-RateLimit-Remaining', userLimit.remaining);
      }
      
      next();
    });
    \`\`\`
    
    **5. Handling Edge Cases**
    
    **Clock Drift**:
    \`\`\`javascript
    // Use Redis TIME instead of server time
    const [seconds, microseconds] = await redis.time();
    const now = seconds * 1000 + Math.floor(microseconds / 1000);
    
    // Redis provides consistent time across all servers
    \`\`\`
    
    **Redis Failure**:
    \`\`\`javascript
    async function rateLimit(identifier, config) {
      try {
        // Normal rate limiting
        return await rateLimitRedis(identifier, config);
      } catch (error) {
        logger.error('Redis error, failing open', { error });
        
        // Strategy 1: Fail open (allow request)
        return { allowed: true, remaining: -1 };
        
        // Strategy 2: Fail closed (deny request)
        // return { allowed: false, remaining: 0, retryAfter: 60 };
        
        // Strategy 3: Local rate limiting (fallback)
        return await rateLimitLocal(identifier, config);
      }
    }
    \`\`\`
    
    **Redis Cluster Split-Brain**:
    \`\`\`yaml
    # redis.conf
    cluster-require-full-coverage yes  # Stop serving if cluster unhealthy
    min-replicas-to-write 1           # Require replica acknowledgment
    \`\`\`
    
    **6. Monitoring & Alerting**
    
    **Metrics to Track**:
    \`\`\`javascript
    const metrics = {
      // Rate limit hits
      rateLimitHitsTotal: new Counter({
        name: 'rate_limit_hits_total',
        help: 'Total rate limit hits',
        labelNames: ['limit_type', 'endpoint']
      }),
      
      // Allowed requests
      rateLimitAllowed: new Counter({
        name: 'rate_limit_allowed_total',
        help: 'Allowed requests',
        labelNames: ['limit_type']
      }),
      
      // Redis latency
      redisLatency: new Histogram({
        name: 'redis_latency_seconds',
        help: 'Redis operation latency'
      }),
      
      // Tokens remaining (gauge)
      tokensRemaining: new Gauge({
        name: 'rate_limit_tokens_remaining',
        help: 'Tokens remaining per user'
      })
    };
    
    // Track in middleware
    if (blocked) {
      metrics.rateLimitHitsTotal.inc({
        limit_type: blocked.check.name,
        endpoint: req.path
      });
    } else {
      metrics.rateLimitAllowed.inc({
        limit_type: 'per-user'
      });
    }
    \`\`\`
    
    **Grafana Dashboard**:
    \`\`\`yaml
    # Key metrics to visualize
    - Rate limit hit rate (by type, endpoint)
    - Top rate-limited users/IPs
    - Redis latency (p50, p95, p99)
    - Redis cluster health (nodes up, replication lag)
    - Request distribution across limits
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    # Prometheus alerts
    - alert: HighRateLimitHitRate
      expr: rate(rate_limit_hits_total[5m]) > 100
      annotations:
        summary: "High rate limit hit rate: {{ \$value }}/sec"
    
    - alert: RedisClusterDown
      expr: redis_cluster_state != 1
      annotations:
        summary: "Redis cluster unhealthy"
    
    - alert: RedisHighLatency
      expr: histogram_quantile(0.95, redis_latency_seconds) > 0.1
      annotations:
        summary: "Redis p95 latency > 100ms"
    \`\`\`
    
    **7. Performance Optimizations**
    
    **Connection Pooling**:
    \`\`\`javascript
    // Reuse Redis connection
    const redis = new Redis.Cluster([...], {
      poolSize: 10,      // Connection pool per node
      enableReadyCheck: true,
      maxRetriesPerRequest: 3
    });
    \`\`\`
    
    **Pipelining for Multiple Checks**:
    \`\`\`javascript
    // Instead of sequential checks
    const result1 = await rateLimit('user:123', limits.perUser);
    const result2 = await rateLimit('ip:1.2.3.4', limits.perIP);
    
    // Use pipeline (parallel)
    const pipeline = redis.pipeline();
    pipeline.evalsha(scriptSha, 1, 'user:123', ...);
    pipeline.evalsha(scriptSha, 1, 'ip:1.2.3.4', ...);
    const results = await pipeline.exec();
    \`\`\`
    
    **8. Testing Strategy**
    
    **Load Test**:
    \`\`\`bash
    # Simulate 1M req/sec
    artillery run --target https://api.example.com \\
      --count 10000 \\
      --rate 100 \\
      rate-limit-test.yml
    \`\`\`
    
    **Test Cases**:
    \`\`\`javascript
    describe('Rate Limiting', () => {
      it('should allow requests under limit', async () => {
        for (let i = 0; i < 100; i++) {
          const res = await request(app).get('/api/data');
          expect(res.status).toBe(200);
        }
      });
      
      it('should block requests over limit', async () => {
        // Make 101 requests
        for (let i = 0; i < 101; i++) {
          const res = await request(app).get('/api/data');
          if (i < 100) {
            expect(res.status).toBe(200);
          } else {
            expect(res.status).toBe(429);
            expect(res.headers['retry-after']).toBeDefined();
          }
        }
      });
      
      it('should reset after window expires', async () => {
        // Hit limit
        for (let i = 0; i < 100; i++) {
          await request(app).get('/api/data');
        }
        
        // Wait for refill
        await sleep(60000); // 1 minute
        
        // Should allow again
        const res = await request(app).get('/api/data');
        expect(res.status).toBe(200);
      });
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Token Bucket + Redis** for distributed rate limiting
    2. **Lua scripts** ensure atomic operations (prevent race conditions)
    3. **Redis TIME** handles clock drift across servers
    4. **Fail open** on Redis errors (better UX) but log for monitoring
    5. **Multiple limits** (per-user, per-IP, per-endpoint) for flexibility
    6. **Redis Cluster** (3 masters + 3 replicas) for 1M req/sec
    7. **Connection pooling** and pipelining for performance
    8. **Monitor**: hit rate, Redis latency, cluster health
    9. **Alert**: high hit rate, Redis down, high latency
    10. **Test**: load testing, edge cases, Redis failures`,
          keyPoints: [
            'Token Bucket + Redis Cluster for distributed rate limiting at scale',
            'Lua scripts provide atomic operations to prevent race conditions across servers',
            'Use Redis TIME command to handle clock drift in distributed systems',
            'Implement multiple rate limit dimensions: per-user, per-IP, per-endpoint',
            'Fail open on Redis errors for better UX, but monitor and alert',
            'Redis Cluster (3 masters + 3 replicas) handles 1M requests/second',
          ],
        },
        {
          id: 'rate-limit-disc-2',
          question:
            'You have a public API with three tiers: Free (100 requests/day), Pro ($50/month, 10,000 requests/day), and Enterprise (custom pricing, unlimited). Design the rate limiting strategy including how to handle burst traffic, trial periods, overages, and billing integration. Discuss implementation, user experience, and edge cases.',
          sampleAnswer: `**Tiered Rate Limiting Strategy for Public API**

**1. Rate Limit Structure**

\`\`\`typescript
interface RateLimitTier {
  name: string;
  dailyLimit: number;
  burstLimit: number;
  overage: {
    allowed: boolean;
    maxOverage: number;
    costPerRequest: number;
  };
}

const tiers: Record<string, RateLimitTier> = {
  free: {
    name: 'Free',
    dailyLimit: 100,
    burstLimit: 10, // 10 requests per minute burst
    overage: {
      allowed: false,
      maxOverage: 0,
      costPerRequest: 0
    }
  },
  pro: {
    name: 'Pro',
    dailyLimit: 10_000,
    burstLimit: 100, // 100 requests per minute burst
    overage: {
      allowed: true,
      maxOverage: 1000, // Allow 10% overage
      costPerRequest: 0.01 // $0.01 per request over limit
    }
  },
  enterprise: {
    name: 'Enterprise',
    dailyLimit: Infinity,
    burstLimit: 1000, // 1000 requests per minute burst
    overage: {
      allowed: true,
      maxOverage: Infinity,
      costPerRequest: 0 // Custom billing
    }
  }
};
\`\`\`

**2. Multi-Dimensional Rate Limiting**

\`\`\`typescript
interface RateLimitKey {
  userId: string;
  tier: string;
  apiKey: string;
}

class TieredRateLimiter {
  private redis: Redis;
  
  async checkRateLimit(key: RateLimitKey): Promise<RateLimitResult> {
    const tier = tiers[key.tier];
    
    // Check daily limit
    const dailyUsage = await this.getDailyUsage(key);
    
    // Check burst limit (per minute)
    const burstUsage = await this.getBurstUsage(key);
    
    // Evaluate limits
    if (dailyUsage >= tier.dailyLimit) {
      // Over daily limit - check overage
      if (!tier.overage.allowed) {
        return {
          allowed: false,
          reason: 'daily_limit_exceeded',
          retryAfter: this.secondsUntilMidnight(),
          usage: {
            daily: dailyUsage,
            dailyLimit: tier.dailyLimit,
            burst: burstUsage,
            burstLimit: tier.burstLimit
          }
        };
      }
      
      // Overage allowed - check max overage
      const overage = dailyUsage - tier.dailyLimit;
      if (overage >= tier.overage.maxOverage) {
        return {
          allowed: false,
          reason: 'max_overage_exceeded',
          retryAfter: this.secondsUntilMidnight(),
          overageCost: overage * tier.overage.costPerRequest
        };
      }
      
      // Allow with overage charge
      await this.incrementUsage(key);
      await this.recordOverageCharge(key, tier.overage.costPerRequest);
      
      return {
        allowed: true,
        isOverage: true,
        overageCost: (overage + 1) * tier.overage.costPerRequest,
        usage: { daily: dailyUsage + 1, dailyLimit: tier.dailyLimit }
      };
    }
    
    // Check burst limit
    if (burstUsage >= tier.burstLimit) {
      return {
        allowed: false,
        reason: 'burst_limit_exceeded',
        retryAfter: 60, // 1 minute
        usage: { burst: burstUsage, burstLimit: tier.burstLimit }
      };
    }
    
    // Within limits
    await this.incrementUsage(key);
    
    return {
      allowed: true,
      isOverage: false,
      usage: {
        daily: dailyUsage + 1,
        dailyLimit: tier.dailyLimit,
        burst: burstUsage + 1,
        burstLimit: tier.burstLimit
      }
    };
  }
  
  private async getDailyUsage(key: RateLimitKey): Promise<number> {
    const dailyKey = \`rate_limit:daily:\${key.userId}:\${this.getToday()}\`;
    const count = await this.redis.get(dailyKey);
    return parseInt(count || '0');
  }
  
  private async getBurstUsage(key: RateLimitKey): Promise<number> {
    const burstKey = \`rate_limit:burst:\${key.userId}:\${this.getCurrentMinute()}\`;
    const count = await this.redis.get(burstKey);
    return parseInt(count || '0');
  }
  
  private async incrementUsage(key: RateLimitKey): Promise<void> {
    const dailyKey = \`rate_limit:daily:\${key.userId}:\${this.getToday()}\`;
    const burstKey = \`rate_limit:burst:\${key.userId}:\${this.getCurrentMinute()}\`;
    
    await this.redis
      .multi()
      .incr(dailyKey)
      .expire(dailyKey, 86400) // 24 hours
      .incr(burstKey)
      .expire(burstKey, 60) // 1 minute
      .exec();
  }
  
  private async recordOverageCharge(key: RateLimitKey, cost: number): Promise<void> {
    const chargeKey = \`overage_charges:\${key.userId}:\${this.getMonth()}\`;
    await this.redis.incrbyfloat(chargeKey, cost);
    
    // Add to billing queue
    await this.queueBillingEvent({
      userId: key.userId,
      amount: cost,
      description: 'API overage charge',
      timestamp: Date.now()
    });
  }
}
\`\`\`

**3. Trial Period Handling**

\`\`\`typescript
interface TrialConfig {
  duration: number; // days
  limits: RateLimitTier;
  upgradePrompt: boolean;
}

async function checkTrial(userId: string): Promise<TrialStatus> {
  const trial = await db.trials.findOne({ userId });
  
  if (!trial) {
    // No active trial
    return { active: false };
  }
  
  const now = Date.now();
  const endTime = trial.startTime + (trial.duration * 86400000);
  
  if (now > endTime) {
    // Trial expired
    await db.users.update({ userId }, { tier: 'free' });
    return {
      active: false,
      expired: true,
      message: 'Your trial has ended. Upgrade to continue with higher limits.'
    };
  }
  
  const daysRemaining = Math.ceil((endTime - now) / 86400000);
  
  // Prompt upgrade at 80% and 100% of usage
  const usage = await getDailyUsage({ userId, tier: 'trial', apiKey: '' });
  const usagePercent = (usage / trial.limits.dailyLimit) * 100;
  
  if (usagePercent >= 80 && trial.upgradePrompt) {
    return {
      active: true,
      daysRemaining,
      showUpgradePrompt: true,
      message: \`You've used \${usagePercent.toFixed(0)}% of your trial limits. Upgrade to Pro for 100x more requests.\`
    };
  }
  
  return {
    active: true,
    daysRemaining,
    showUpgradePrompt: false
  };
}
\`\`\`

**4. Overage Billing Integration**

\`\`\`typescript
// Monthly billing job
async function processMonthlyOverages() {
  const month = getCurrentMonth();
  
  // Get all users with overage charges
  const keys = await redis.keys(\`overage_charges:*:\${month}\`);
  
  for (const key of keys) {
    const userId = key.split(':')[1];
    const totalOverageCharge = parseFloat(await redis.get(key) || '0');
    
    if (totalOverageCharge > 0) {
      // Charge via Stripe
      const user = await db.users.findOne({ userId });
      
      try {
        await stripe.invoiceItems.create({
          customer: user.stripeCustomerId,
          amount: Math.round(totalOverageCharge * 100), // cents
          currency: 'usd',
          description: \`API overage charges for \${month}\`
        });
        
        // Send notification
        await sendEmail({
          to: user.email,
          subject: 'API Overage Invoice',
          body: \`You used \${totalOverageCharge.toFixed(2)} USD in API overages this month.\`
        });
        
        // Clear overage counter
        await redis.del(key);
        
      } catch (error) {
        // Log billing failure
        logger.error(\`Billing failed for user \${userId}\`, error);
        
        // Retry later
        await redis.lpush('billing_retry_queue', JSON.stringify({
          userId,
          amount: totalOverageCharge,
          month
        }));
      }
    }
  }
}
\`\`\`

**5. User Experience**

**Headers for Transparency**:

\`\`\`typescript
app.use((req, res, next) => {
  const result = await rateLimiter.checkRateLimit(req.user);
  
  // Set headers
  res.setHeader('X-RateLimit-Limit', result.usage.dailyLimit);
  res.setHeader('X-RateLimit-Remaining', 
    Math.max(0, result.usage.dailyLimit - result.usage.daily));
  res.setHeader('X-RateLimit-Reset', getMidnightTimestamp());
  
  if (result.isOverage) {
    res.setHeader('X-RateLimit-Overage', 'true');
    res.setHeader('X-RateLimit-Overage-Cost', result.overageCost.toFixed(2));
  }
  
  if (!result.allowed) {
    res.setHeader('Retry-After', result.retryAfter);
    return res.status(429).json({
      error: 'Rate limit exceeded',
      reason: result.reason,
      retryAfter: result.retryAfter,
      upgradeUrl: 'https://example.com/upgrade'
    });
  }
  
  next();
});
\`\`\`

**Dashboard**:

\`\`\`typescript
// Real-time usage dashboard
app.get('/api/usage', async (req, res) => {
  const userId = req.user.id;
  const tier = req.user.tier;
  
  const daily = await getDailyUsage({ userId, tier, apiKey: '' });
  const limits = tiers[tier];
  
  const overageCharges = await redis.get(\`overage_charges:\${userId}:\${getMonth()}\`);
  
  res.json({
    tier: tier,
    usage: {
      daily: daily,
      dailyLimit: limits.dailyLimit,
      percentUsed: (daily / limits.dailyLimit) * 100,
      remaining: Math.max(0, limits.dailyLimit - daily)
    },
    burst: {
      limit: limits.burstLimit,
      current: await getBurstUsage({ userId, tier, apiKey: '' })
    },
    overage: {
      allowed: limits.overage.allowed,
      current: Math.max(0, daily - limits.dailyLimit),
      cost: parseFloat(overageCharges || '0').toFixed(2)
    },
    resetTime: getMidnightTimestamp()
  });
});
\`\`\`

**6. Edge Cases**

**Timezone Handling**:
\`\`\`typescript
// Use UTC for daily resets to avoid confusion
function getToday(): string {
  return new Date().toISOString().split('T')[0]; // YYYY-MM-DD in UTC
}
\`\`\`

**Tier Upgrades Mid-Day**:
\`\`\`typescript
// When user upgrades, don't reset counter - just increase limit
async function handleTierUpgrade(userId: string, newTier: string) {
  await db.users.update({ userId }, { tier: newTier });
  
  // Usage counter persists - user immediately gets higher limit
  // No need to reset daily counter
  
  // Send confirmation
  await sendEmail({
    to: user.email,
    subject: 'Tier Upgraded',
    body: \`You now have \${tiers[newTier].dailyLimit} requests per day.\`
  });
}
\`\`\`

**Billing Failures**:
\`\`\`typescript
// If billing fails, don't immediately block user
// Grace period: 7 days
async function checkBillingGrace(userId: string): Promise<boolean> {
  const failedBillings = await db.billingFailures.find({
    userId,
    resolved: false
  });
  
  if (failedBillings.length === 0) {
    return true; // No issues
  }
  
  const oldestFailure = failedBillings[0].timestamp;
  const daysSinceFailure = (Date.now() - oldestFailure) / 86400000;
  
  if (daysSinceFailure > 7) {
    // Grace period expired - downgrade to free tier
    await db.users.update({ userId }, { tier: 'free' });
    
    await sendEmail({
      to: user.email,
      subject: 'Account Downgraded',
      body: 'Your payment failed. Update payment info to restore access.'
    });
    
    return false;
  }
  
  return true; // Still in grace period
}
\`\`\`

**Key Takeaways**:
1. **Multi-dimensional limits**: daily, burst, overage
2. **Transparent pricing**: Show usage and costs in headers and dashboard
3. **Graceful overage**: Allow Pro users to exceed limits with per-request charges
4. **Trial management**: Auto-downgrade after trial, prompt upgrades
5. **Billing integration**: Monthly overage invoices via Stripe
6. **UX-first**: Clear error messages with upgrade paths
7. **Edge cases**: Handle timezones (UTC), mid-day upgrades, billing failures with grace periods`,
          keyPoints: [
            'Multi-dimensional rate limiting: daily limits, burst limits, and overage allowances',
            'Tiered pricing: Free (hard limit), Pro (overage allowed), Enterprise (unlimited)',
            'Transparent UX: Show usage in headers (X-RateLimit-*) and real-time dashboard',
            'Overage billing: Track per-request charges, invoice monthly via Stripe',
            'Trial management: Auto-downgrade after expiration, prompt upgrades at 80% usage',
            'Edge cases: UTC for resets, preserve usage on tier upgrades, 7-day billing grace period',
          ],
        },
        {
          id: 'rate-limit-disc-3',
          question:
            'Compare Token Bucket, Leaky Bucket, Fixed Window Counter, and Sliding Window algorithms for rate limiting. For each algorithm, explain the implementation, pros/cons, memory requirements, and ideal use cases. Which would you choose for a high-traffic REST API and why?',
          sampleAnswer: `**Comprehensive Comparison of Rate Limiting Algorithms**

---

## **1. Token Bucket**

**How it Works**:
- Bucket holds tokens (max capacity = bucket size)
- Tokens added at constant rate (refill rate)
- Each request consumes 1 token
- If bucket empty, request denied

**Pseudocode**:
\`\`\`python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # max tokens
        self.tokens = capacity    # current tokens
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.now()
    
    def allow_request(self):
        # Refill tokens based on time elapsed
        now = time.now()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we have tokens
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- token_bucket.lua
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state
local state = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(state[1]) or capacity
local last_refill = tonumber(state[2]) or now

-- Refill tokens
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if we can allow request
if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Allows burst traffic (accumulate tokens when idle)
✅ Simple to understand and implement
✅ Constant memory (just stores token count)
✅ Works well for APIs with variable traffic

**Cons**:
❌ Can't prevent sustained bursts if bucket is large
❌ Requires timestamps (clock drift in distributed systems)

**Memory**: O(1) - stores 2 values (tokens, last_refill)

**Ideal Use Cases**:
- REST APIs with variable traffic
- Allow legitimate bursts (user loads page → multiple API calls)
- Most common choice for public APIs

---

## **2. Leaky Bucket**

**How it Works**:
- Requests enter bucket (queue)
- Requests "leak" out at constant rate
- If bucket full, new requests dropped

**Pseudocode**:
\`\`\`python
class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity
        self.queue = []
        self.leak_rate = leak_rate  # requests per second
        self.last_leak = time.now()
    
    def allow_request(self):
        # Leak requests
        now = time.now()
        elapsed = now - self.last_leak
        requests_to_leak = int(elapsed * self.leak_rate)
        
        for _ in range(min(requests_to_leak, len(self.queue))):
            self.queue.pop(0)
        
        self.last_leak = now
        
        # Add new request
        if len(self.queue) < self.capacity:
            self.queue.append(now)
            return True
        return False
\`\`\`

**Pros**:
✅ Smooths out traffic (enforces constant rate)
✅ Prevents bursts (good for protecting downstream)
✅ Fair queuing

**Cons**:
❌ No bursts allowed (bad UX for legitimate spikes)
❌ Requires queue (memory intensive)
❌ Complexity in distributed systems (shared queue)

**Memory**: O(n) - stores queue of requests

**Ideal Use Cases**:
- Protecting downstream services that can't handle bursts
- Network traffic shaping
- Message queue rate limiting

---

## **3. Fixed Window Counter**

**How it Works**:
- Divide time into fixed windows (e.g., 1-minute windows)
- Count requests in current window
- Reset counter at window boundary

**Pseudocode**:
\`\`\`python
class FixedWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size  # seconds
        self.counters = {}  # {window_id: count}
    
    def allow_request(self):
        now = time.now()
        window_id = int(now / self.window_size)
        
        count = self.counters.get(window_id, 0)
        
        if count < self.limit:
            self.counters[window_id] = count + 1
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- fixed_window.lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])  -- window size in seconds

local current = redis.call('INCR', key)

if current == 1 then
    redis.call('EXPIRE', key, window)
end

if current <= limit then
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Extremely simple to implement
✅ Very memory efficient (single counter)
✅ Fast (just increment)

**Cons**:
❌ **Boundary problem**: 2x limit possible at window boundary
  - Example: 100 req/min limit
  - User sends 100 requests at 12:00:59
  - User sends 100 requests at 12:01:00
  - Result: 200 requests in 1 second!
❌ Unfair (early requests in window have advantage)

**Memory**: O(1) - single counter per window

**Ideal Use Cases**:
- Simple rate limiting where boundary problem is acceptable
- Low-traffic APIs
- Internal APIs (not user-facing)

---

## **4. Sliding Window Log**

**How it Works**:
- Store timestamp of each request
- Count requests in last N seconds (sliding window)
- Remove old timestamps

**Pseudocode**:
\`\`\`python
class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.log = []  # list of timestamps
    
    def allow_request(self):
        now = time.now()
        cutoff = now - self.window_size
        
        # Remove old timestamps
        self.log = [ts for ts in self.log if ts > cutoff]
        
        if len(self.log) < self.limit:
            self.log.append(now)
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- sliding_window_log.lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current entries
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add new entry
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Most accurate (no boundary problem)
✅ True sliding window
✅ Fair

**Cons**:
❌ Memory intensive (stores all timestamps)
❌ Slower (need to clean up old entries)
❌ O(n) operations

**Memory**: O(n) - stores timestamp for each request in window

**Ideal Use Cases**:
- When accuracy is critical
- Low request rates
- Compliance/auditing requirements

---

## **5. Sliding Window Counter** (Hybrid)

**How it Works**:
- Combines Fixed Window + Sliding Window
- Uses weighted average of current and previous window

**Formula**:
\`\`\`
count = previous_window_count * (1 - position_in_current_window) + current_window_count
\`\`\`

**Pseudocode**:
\`\`\`python
class SlidingWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.windows = {}  # {window_id: count}
    
    def allow_request(self):
        now = time.now()
        window_id = int(now / self.window_size)
        previous_window_id = window_id - 1
        
        # Calculate position in current window (0.0 to 1.0)
        position = (now % self.window_size) / self.window_size
        
        previous_count = self.windows.get(previous_window_id, 0)
        current_count = self.windows.get(window_id, 0)
        
        # Weighted count
        estimated_count = previous_count * (1 - position) + current_count
        
        if estimated_count < self.limit:
            self.windows[window_id] = current_count + 1
            return True
        return False
\`\`\`

**Pros**:
✅ Solves boundary problem
✅ Memory efficient (only 2 counters)
✅ Fast (just math)
✅ Good approximation of sliding window

**Cons**:
❌ Still an approximation (not exact)
❌ Slightly more complex than fixed window

**Memory**: O(1) - stores 2 counters

**Ideal Use Cases**:
- High-traffic APIs
- Need accuracy without memory overhead
- Best balance of accuracy and efficiency

---

## **Comparison Table**

| Algorithm | Accuracy | Memory | Speed | Bursts | Complexity |
|-----------|----------|--------|-------|--------|------------|
| **Token Bucket** | Good | O(1) | Fast | ✅ Yes | Low |
| **Leaky Bucket** | Excellent | O(n) | Slow | ❌ No | High |
| **Fixed Window** | Poor (boundary) | O(1) | Fastest | ✅ Yes | Lowest |
| **Sliding Log** | Perfect | O(n) | Slowest | ✅ Yes | Medium |
| **Sliding Counter** | Very Good | O(1) | Fast | ✅ Yes | Medium |

---

## **Recommendation for High-Traffic REST API**

**Choose: Token Bucket or Sliding Window Counter**

**Why Token Bucket**:
1. **Allows bursts** - users can accumulate tokens during idle periods
2. **Memory efficient** - O(1) memory per user
3. **Simple** - easy to implement and debug
4. **Industry standard** - AWS API Gateway, Stripe, GitHub all use it
5. **Good UX** - doesn't penalize legitimate burst patterns

**Example**: User loads dashboard → 5 API calls simultaneously
- Token Bucket: ✅ All 5 succeed (had 100 tokens saved up)
- Leaky Bucket: ❌ 4 of 5 queued/rejected
- Fixed Window: ✅ Depends on boundary
- Sliding Log: ✅ All succeed if under limit

**Why Sliding Window Counter as Alternative**:
- If boundary problem is critical
- Need more accurate limiting
- Still memory efficient (O(1))

**Implementation Choice**:
\`\`\`typescript
// Production: Use Token Bucket with Redis
const rateLimiter = new TokenBucketRateLimiter({
  capacity: 100,      // 100 requests
  refillRate: 10,     // 10 tokens per second = 600/minute
  redis: redisClient
});
\`\`\`

**Why NOT the others for high-traffic REST API**:
- **Leaky Bucket**: No bursts = bad UX
- **Fixed Window**: Boundary problem = potential abuse
- **Sliding Log**: O(n) memory = too expensive at scale`,
          keyPoints: [
            'Token Bucket: Allows bursts, O(1) memory, industry standard for REST APIs',
            'Leaky Bucket: Smooths traffic, no bursts, O(n) memory, good for downstream protection',
            'Fixed Window: Simple but has boundary problem (2x rate possible at window edge)',
            'Sliding Window Log: Most accurate but O(n) memory (stores all timestamps)',
            'Sliding Window Counter: Best hybrid - solves boundary problem with O(1) memory',
            'For high-traffic REST API: Choose Token Bucket (allows legitimate bursts, efficient)',
          ],
        },
      ],
    },
    {
      id: 'api-versioning',
      title: 'API Versioning',
      content: `API versioning is critical for maintaining backward compatibility while evolving APIs. This section covers versioning strategies, best practices, and migration patterns for distributed systems.
    
    ## Why API Versioning?
    
    **Problem**: Breaking changes break clients
    
    \`\`\`
    // Version 1
    { "name": "John Doe" }
    
    // Version 2 (BREAKING CHANGE!)
    { "firstName": "John", "lastName": "Doe" }
    
    // Existing clients break!
    const name = data.name; // undefined!
    \`\`\`
    
    **Solution**: Version your API
    
    \`\`\`
    GET /api/v1/users → { "name": "John Doe" }
    GET /api/v2/users → { "firstName": "John", "lastName": "Doe" }
    \`\`\`
    
    ---
    
    ## Versioning Strategies
    
    ### **1. URL Path Versioning** ⭐ (Most Common)
    
    **Format**: \`/api/v{version}/resource\`
    
    \`\`\`
    GET /api/v1/users
    GET /api/v2/users
    GET /api/v3/users
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    // Express.js
    const v1Router = express.Router();
    const v2Router = express.Router();
    
    // V1 routes
    v1Router.get('/users', (req, res) => {
      res.json({
        users: users.map(u => ({ name: u.fullName }))
      });
    });
    
    // V2 routes
    v2Router.get('/users', (req, res) => {
      res.json({
        users: users.map(u => ({
          firstName: u.firstName,
          lastName: u.lastName
        }))
      });
    });
    
    app.use('/api/v1', v1Router);
    app.use('/api/v2', v2Router);
    \`\`\`
    
    **Pros**:
    - Simple and explicit
    - Easy to route and cache
    - Clear deprecation path
    - Works with any client
    
    **Cons**:
    - URL changes (breaks bookmarks)
    - Multiple codebases to maintain
    
    **Best For**: Public APIs, major version changes
    
    ---
    
    ### **2. Header Versioning**
    
    **Format**: Custom header like \`API-Version\` or \`Accept\` header
    
    \`\`\`
    GET /api/users
    API-Version: 1
    
    GET /api/users
    API-Version: 2
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.use((req, res, next) => {
      const version = req.headers['api-version'] || '1';
      req.apiVersion = parseInt(version);
      next();
    });
    
    app.get('/api/users', (req, res) => {
      if (req.apiVersion === 1) {
        return res.json({
          users: users.map(u => ({ name: u.fullName }))
        });
      }
      
      if (req.apiVersion === 2) {
        return res.json({
          users: users.map(u => ({
            firstName: u.firstName,
            lastName: u.lastName
          }))
        });
      }
      
      res.status(400).json({ error: 'Unsupported API version' });
    });
    \`\`\`
    
    **Pros**:
    - Clean URLs (no version in path)
    - Single endpoint
    - Flexible versioning
    
    **Cons**:
    - Not visible in URL (harder to test)
    - Caching more complex (must include header in cache key)
    - Harder for API consumers to discover
    
    **Best For**: Internal APIs, minor version changes
    
    ---
    
    ### **3. Content Negotiation** (Accept Header)
    
    **Format**: \`Accept: application/vnd.company.v2+json\`
    
    \`\`\`
    GET /api/users
    Accept: application/vnd.myapi.v1+json
    
    GET /api/users
    Accept: application/vnd.myapi.v2+json
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.get('/api/users', (req, res) => {
      const accept = req.headers['accept'] || '';
      
      if (accept.includes('vnd.myapi.v1+json')) {
        return res
          .type('application/vnd.myapi.v1+json')
          .json({ users: users.map(u => ({ name: u.fullName })) });
      }
      
      if (accept.includes('vnd.myapi.v2+json')) {
        return res
          .type('application/vnd.myapi.v2+json')
          .json({
            users: users.map(u => ({
              firstName: u.firstName,
              lastName: u.lastName
            }))
          });
      }
      
      // Default to latest version
      return res.json({
        users: users.map(u => ({
          firstName: u.firstName,
          lastName: u.lastName
        }))
      });
    });
    \`\`\`
    
    **Pros**:
    - RESTful (proper use of HTTP)
    - Clean URLs
    
    **Cons**:
    - Complex for clients
    - Hard to test (curl requires correct headers)
    - Caching complex
    
    **Best For**: Strict RESTful APIs
    
    ---
    
    ### **4. Query Parameter Versioning**
    
    **Format**: \`/api/users?version=2\`
    
    \`\`\`
    GET /api/users?version=1
    GET /api/users?version=2
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.get('/api/users', (req, res) => {
      const version = parseInt(req.query.version) || 1;
      
      if (version === 1) {
        return res.json({
          users: users.map(u => ({ name: u.fullName }))
        });
      }
      
      if (version === 2) {
        return res.json({
          users: users.map(u => ({
            firstName: u.firstName,
            lastName: u.lastName
          }))
        });
      }
      
      res.status(400).json({ error: 'Unsupported version' });
    });
    \`\`\`
    
    **Pros**:
    - Simple to implement
    - Easy to test
    - Optional (can default to latest)
    
    **Cons**:
    - Not RESTful (version is not a resource property)
    - Query params meant for filtering, not versioning
    - Pollutes query string
    
    **Best For**: Internal tools, quick prototypes
    
    ---
    
    ## Semantic Versioning for APIs
    
    **Format**: MAJOR.MINOR.PATCH (e.g., v2.1.0)
    
    - **MAJOR**: Breaking changes
    - **MINOR**: New features (backward compatible)
    - **PATCH**: Bug fixes (backward compatible)
    
    **Examples**:
    
    \`\`\`
    v1.0.0 → v1.0.1: Bug fix (backward compatible)
    v1.0.0 → v1.1.0: Added new endpoint (backward compatible)
    v1.0.0 → v2.0.0: Changed response format (BREAKING)
    \`\`\`
    
    **In Practice**:
    \`\`\`
    Only major version in URL: /api/v2/users
    Full version in response header: API-Version: 2.1.0
    \`\`\`
    
    ---
    
    ## Deprecation Strategy
    
    ### **1. Announce Deprecation**
    
    \`\`\`javascript
    app.use('/api/v1', (req, res, next) => {
      res.setHeader('Deprecation', 'true');
      res.setHeader('Sunset', 'Sat, 31 Dec 2024 23:59:59 GMT');
      res.setHeader('Link', '<https://api.example.com/docs/migration>; rel="alternate"');
      next();
    });
    \`\`\`
    
    ### **2. Monitor Usage**
    
    \`\`\`javascript
    app.use('/api/v1', (req, res, next) => {
      logger.warn('Deprecated API used', {
        endpoint: req.path,
        client: req.headers['user-agent'],
        ip: req.ip
      });
      
      metrics.deprecatedApiCalls.inc({
        version: 'v1',
        endpoint: req.path
      });
      
      next();
    });
    \`\`\`
    
    ### **3. Gradual Shutdown**
    
    **Phase 1: Soft Deprecation** (3 months)
    - Add deprecation headers
    - Email clients
    - Monitor usage
    
    **Phase 2: Hard Deprecation** (1 month)
    - Return 410 Gone for new clients
    - Allow existing clients (via whitelist)
    
    \`\`\`javascript
    const allowedClients = new Set(['client-a', 'client-b']);
    
    app.use('/api/v1', (req, res, next) => {
      const clientId = req.headers['x-client-id'];
      
      if (!allowedClients.has(clientId)) {
        return res.status(410).json({
          error: 'API version 1 is no longer supported',
          message: 'Please upgrade to v2: https://api.example.com/docs/migration'
        });
      }
      
      next();
    });
    \`\`\`
    
    **Phase 3: Complete Shutdown** (after 4 months)
    - Return 410 Gone for all clients
    - Remove v1 code
    
    ---
    
    ## Backward Compatibility Patterns
    
    ### **1. Additive Changes** (Non-Breaking)
    
    ✅ **Safe to add**:
    - New endpoints
    - New optional fields
    - New query parameters (optional)
    - New HTTP headers
    
    \`\`\`javascript
    // V1 response
    {
      "id": 1,
      "name": "John Doe"
    }
    
    // V1.1 response (backward compatible)
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"  // NEW field (clients ignore if unknown)
    }
    \`\`\`
    
    ### **2. Field Transformation** (Breaking)
    
    ❌ **Breaking changes**:
    - Renaming fields
    - Changing field types
    - Removing fields
    - Changing error format
    
    **Solution: Maintain both versions**:
    \`\`\`javascript
    // Shared data layer
    function getUser(id) {
      return {
        id: id,
        firstName: 'John',
        lastName: 'Doe',
        fullName: 'John Doe', // Computed for v1
        age: 30
      };
    }
    
    // V1 endpoint
    app.get('/api/v1/users/:id', (req, res) => {
      const user = getUser(req.params.id);
      res.json({
        id: user.id,
        name: user.fullName // V1 format
      });
    });
    
    // V2 endpoint
    app.get('/api/v2/users/:id', (req, res) => {
      const user = getUser(req.params.id);
      res.json({
        id: user.id,
        firstName: user.firstName, // V2 format
        lastName: user.lastName
      });
    });
    \`\`\`
    
    ### **3. Adapter Pattern**
    
    \`\`\`javascript
    // Data model (internal)
    class User {
      constructor(id, firstName, lastName) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
      }
    }
    
    // V1 Adapter
    class UserV1Adapter {
      static toResponse(user) {
        return {
          id: user.id,
          name: \`\${user.firstName} \${user.lastName}\`
        };
      }
      
      static fromRequest(data) {
        const [firstName, ...lastNameParts] = data.name.split(' ');
        return new User(data.id, firstName, lastNameParts.join(' '));
      }
    }
    
    // V2 Adapter
    class UserV2Adapter {
      static toResponse(user) {
        return {
          id: user.id,
          firstName: user.firstName,
          lastName: user.lastName
        };
      }
      
      static fromRequest(data) {
        return new User(data.id, data.firstName, data.lastName);
      }
    }
    
    // Use in routes
    app.get('/api/v1/users/:id', async (req, res) => {
      const user = await userService.getUser(req.params.id);
      res.json(UserV1Adapter.toResponse(user));
    });
    
    app.get('/api/v2/users/:id', async (req, res) => {
      const user = await userService.getUser(req.params.id);
      res.json(UserV2Adapter.toResponse(user));
    });
    \`\`\`
    
    ---
    
    ## GraphQL Versioning
    
    **GraphQL philosophy**: No versioning, only schema evolution
    
    **Approach**: Deprecate fields, don't remove them
    
    \`\`\`graphql
    type User {
      id: ID!
      name: String! @deprecated(reason: "Use firstName and lastName instead")
      firstName: String!
      lastName: String!
    }
    \`\`\`
    
    **Clients can transition gradually**:
    \`\`\`graphql
    # Old clients
    query {
      user(id: "123") {
        name  # Still works
      }
    }
    
    # New clients
    query {
      user(id: "123") {
        firstName
        lastName
      }
    }
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **URL path versioning** most common and recommended (/api/v2/users)
    2. **Semantic versioning**: Only major version in URL, full version in headers
    3. **Deprecation strategy**: Announce → Monitor → Whitelist → Shutdown (4+ months)
    4. **Additive changes are safe**: New fields, new endpoints (non-breaking)
    5. **Breaking changes**: Rename fields, change types, remove fields → new version
    6. **Adapter pattern** maintains single codebase while supporting multiple versions
    7. **GraphQL**: No versioning, deprecate fields instead
    8. **Headers**: Use Deprecation, Sunset, Link headers for deprecation
    9. **Monitor deprecated API usage** to identify clients needing migration
    10. **Keep old versions for 6-12 months** minimum before shutdown`,
      multipleChoice: [
        {
          id: 'api-version-strategy',
          question:
            'Which API versioning strategy is most commonly used for public REST APIs?',
          options: [
            'Query parameter versioning (?version=2)',
            'URL path versioning (/api/v2/users)',
            'Header versioning (API-Version: 2)',
            'Content negotiation (Accept: application/vnd.api.v2+json)',
          ],
          correctAnswer: 1,
          explanation:
            "URL path versioning (/api/v2/users) is most common for public REST APIs because it's explicit, easy to test with curl/browsers, cacheable, and doesn't require special headers. Query parameters pollute URLs, headers are less discoverable, and content negotiation is complex for clients.",
        },
        {
          id: 'api-breaking-change',
          question:
            'Which change to an API is considered a breaking change requiring a new major version?',
          options: [
            'Adding a new optional field to the response',
            'Adding a new endpoint',
            'Renaming an existing response field from "name" to "fullName"',
            'Adding a new optional query parameter',
          ],
          correctAnswer: 2,
          explanation:
            "Renaming a field is breaking because existing clients expect the old field name and will break when it's removed. Adding new optional fields, endpoints, or parameters are additive changes that are backward compatible (old clients can ignore them).",
        },
        {
          id: 'api-deprecation-header',
          question:
            'Which HTTP header should an API return to indicate that an endpoint is deprecated and will be removed on a specific date?',
          options: [
            'X-API-Deprecated: true',
            'Warning: 299 - "Deprecated API"',
            'Sunset: Sat, 31 Dec 2024 23:59:59 GMT',
            'Cache-Control: no-cache',
          ],
          correctAnswer: 2,
          explanation:
            'The Sunset header (RFC 8594) indicates when a resource will be removed, using an HTTP date format. Deprecation header indicates current deprecation status, but Sunset provides the critical removal date. Warning is for cache warnings, and Cache-Control is for caching directives.',
        },
        {
          id: 'api-semantic-versioning',
          question:
            'In semantic versioning (MAJOR.MINOR.PATCH), when should you increment the MINOR version?',
          options: [
            'When making backward-incompatible changes',
            'When adding new features in a backward-compatible manner',
            'When fixing bugs without changing functionality',
            'When updating documentation',
          ],
          correctAnswer: 1,
          explanation:
            "MINOR version increments for new backward-compatible features. MAJOR increments for breaking changes. PATCH increments for backward-compatible bug fixes. Documentation updates don't require version changes.",
        },
        {
          id: 'api-graphql-versioning',
          question:
            'How does GraphQL handle API versioning differently from REST?',
          options: [
            'GraphQL uses URL path versioning like /graphql/v2',
            'GraphQL uses header versioning with GraphQL-Version header',
            'GraphQL avoids versioning by deprecating fields and evolving the schema gradually',
            'GraphQL requires a new schema file for each version',
          ],
          correctAnswer: 2,
          explanation:
            "GraphQL philosophy is to avoid versioning by evolving the schema gradually. Fields are marked with @deprecated directive rather than removed, allowing old and new clients to coexist. Clients request only the fields they need, so new fields don't break old clients.",
        },
      ],
      quiz: [
        {
          id: 'api-versioning-migration',
          question:
            'You need to make a breaking change to your public API used by 10,000 clients: changing the user response from {"name": "John Doe"} to {"firstName": "John", "lastName": "Doe"}. Design a complete migration strategy including: versioning approach, deprecation timeline, client communication, monitoring, handling legacy clients, and ensuring zero downtime. Provide specific implementation details.',
          sampleAnswer: `**Complete API Migration Strategy**
    
    **1. Versioning Approach: URL Path Versioning**
    
    **Why**: Public API with 10K clients needs clarity and simplicity
    
    \`\`\`
    Current: GET /api/v1/users/:id → {"id": 1, "name": "John Doe"}
    New:     GET /api/v2/users/:id → {"id": 1, "firstName": "John", "lastName": "Doe"}
    \`\`\`
    
    **2. Timeline** (6-month migration)
    
    | **Phase** | **Duration** | **Actions** |
    |-----------|--------------|-------------|
    | **Announcement** | Week 1-2 | Announce v2, deprecate v1, publish migration guide |
    | **Soft Deprecation** | Month 1-4 | V1 works fully, add deprecation headers, monitor usage |
    | **Hard Deprecation** | Month 5 | V1 rate-limited, email heavy users, whitelist critical clients |
    | **Shutdown** | Month 6 | V1 returns 410 Gone, remove code after 30 days |
    
    **3. Implementation**
    
    **Shared Data Layer**:
    \`\`\`javascript
    // models/user.js
    class User {
      constructor(data) {
        this.id = data.id;
        this.firstName = data.first_name;
        this.lastName = data.last_name;
      }
      
      // Computed property for v1
      get fullName() {
        return \`\${this.firstName} \${this.lastName}\`;
      }
      
      // V1 format
      toV1() {
        return {
          id: this.id,
          name: this.fullName
        };
      }
      
      // V2 format
      toV2() {
        return {
          id: this.id,
          firstName: this.firstName,
          lastName: this.lastName
        };
      }
    }
    
    module.exports = User;
    \`\`\`
    
    **V1 Endpoint** (deprecated):
    \`\`\`javascript
    // routes/v1/users.js
    const express = require('express');
    const router = express.Router();
    const User = require('../../models/user');
    
    // Deprecation middleware
    router.use((req, res, next) => {
      // Add deprecation headers
      res.setHeader('Deprecation', 'true');
      res.setHeader('Sunset', 'Mon, 30 Jun 2024 23:59:59 GMT');
      res.setHeader('Link', '<https://api.example.com/docs/v2-migration>; rel="alternate"');
      res.setHeader('X-API-Version', '1.0.0');
      
      // Log usage
      logger.warn('V1 API usage', {
        endpoint: req.originalUrl,
        client: req.headers['x-client-id'] || 'unknown',
        userAgent: req.headers['user-agent'],
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      // Track metrics
      metrics.apiVersionUsage.inc({
        version: 'v1',
        endpoint: req.path,
        client: req.headers['x-client-id'] || 'unknown'
      });
      
      next();
    });
    
    router.get('/users/:id', async (req, res) => {
      try {
        const user = await User.findById(req.params.id);
        if (!user) {
          return res.status(404).json({ error: 'User not found' });
        }
        
        res.json(user.toV1());
      } catch (error) {
        logger.error('V1 API error', { error, userId: req.params.id });
        res.status(500).json({ error: 'Internal server error' });
      }
    });
    
    module.exports = router;
    \`\`\`
    
    **V2 Endpoint** (new):
    \`\`\`javascript
    // routes/v2/users.js
    const express = require('express');
    const router = express.Router();
    const User = require('../../models/user');
    
    router.use((req, res, next) => {
      res.setHeader('X-API-Version', '2.0.0');
      next();
    });
    
    router.get('/users/:id', async (req, res) => {
      try {
        const user = await User.findById(req.params.id);
        if (!user) {
          return res.status(404).json({ error: 'User not found' });
        }
        
        res.json(user.toV2());
      } catch (error) {
        logger.error('V2 API error', { error, userId: req.params.id });
        res.status(500).json({ error: 'Internal server error' });
      }
    });
    
    module.exports = router;
    \`\`\`
    
    **Mount Routes**:
    \`\`\`javascript
    // app.js
    const v1Routes = require('./routes/v1/users');
    const v2Routes = require('./routes/v2/users');
    
    app.use('/api/v1', v1Routes);
    app.use('/api/v2', v2Routes);
    \`\`\`
    
    **4. Client Communication**
    
    **Week 1: Announcement Email**:
    \`\`\`
    Subject: [Action Required] API v2 Released - v1 Deprecated
    
    Dear API Consumer,
    
    We've released API v2 with improved user data structure:
    
    Breaking Change:
    - V1: {"name": "John Doe"}
    - V2: {"firstName": "John", "lastName": "Doe"}
    
    Timeline:
    - Now: V2 available, V1 fully functional
    - Month 4: V1 soft deprecated (still works)
    - Month 5: V1 rate-limited (500 req/day)
    - Month 6: V1 shutdown (returns 410 Gone)
    
    Migration Guide: https://api.example.com/docs/v2-migration
    
    Action Required:
    1. Test your integration with V2
    2. Migrate by Month 5 to avoid rate limits
    3. Contact support if you need more time
    
    Best regards,
    API Team
    \`\`\`
    
    **Month 3: Reminder Email** (to clients still on v1):
    \`\`\`
    Subject: [Urgent] API v1 Deprecation in 3 Months
    
    Dear API Consumer,
    
    Our logs show you're still using API v1:
    - Your usage: 10,000 requests/day
    - Endpoints used: GET /api/v1/users/:id
    
    V1 will be rate-limited in 2 months (500 req/day).
    Please migrate to v2 immediately.
    
    Need help? Reply to this email or visit our support page.
    \`\`\`
    
    **5. Monitoring Dashboard**
    
    **Grafana Dashboard**:
    \`\`\`yaml
    panels:
      - title: "API Version Usage"
        query: sum by (version) (rate(api_version_usage[5m]))
        type: time-series
        
      - title: "Top V1 Clients"
        query: topk(10, sum by (client) (rate(api_version_usage{version="v1"}[24h])))
        type: table
        
      - title: "V1 Usage by Endpoint"
        query: sum by (endpoint) (rate(api_version_usage{version="v1"}[1h]))
        type: bar-chart
        
      - title: "Migration Progress"
        query: |
          (sum(rate(api_version_usage{version="v2"}[1h])) / 
           (sum(rate(api_version_usage{version="v1"}[1h])) + 
            sum(rate(api_version_usage{version="v2"}[1h])))) * 100
        type: gauge
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    - alert: V1UsageStillHigh
      expr: sum(rate(api_version_usage{version="v1"}[1h])) > 100
      for: 1h
      annotations:
        summary: "V1 API still receiving >100 req/sec after Month 4"
    
    - alert: NewV1Client
      expr: increase(api_version_usage{version="v1"}[5m]) > 0 and Month > 5
      annotations:
        summary: "New client using deprecated V1 API"
    \`\`\`
    
    **6. Hard Deprecation** (Month 5)
    
    **Rate Limiting for V1**:
    \`\`\`javascript
    // V1 rate limit: 500 requests/day
    router.use(async (req, res, next) => {
      const clientId = req.headers['x-client-id'] || req.ip;
      const key = \`v1_rate_limit:\${clientId}\`;
      
      const requests = await redis.incr(key);
      if (requests === 1) {
        await redis.expire(key, 86400); // 24 hours
      }
      
      if (requests > 500) {
        return res.status(429).json({
          error: 'V1 API rate limit exceeded',
          limit: 500,
          window: '24 hours',
          message: 'Please migrate to V2: https://api.example.com/docs/v2-migration',
          contact: 'support@example.com for extension'
        });
      }
      
      res.setHeader('X-RateLimit-Limit', '500');
      res.setHeader('X-RateLimit-Remaining', String(500 - requests));
      
      next();
    });
    \`\`\`
    
    **Whitelist Critical Clients**:
    \`\`\`javascript
    const v1Whitelist = new Set([
      'client-a', // Enterprise client, extension granted
      'client-b'  // Government client, slow approval process
    ]);
    
    router.use((req, res, next) => {
      const clientId = req.headers['x-client-id'];
      
      if (v1Whitelist.has(clientId)) {
        // Skip rate limiting
        return next();
      }
      
      // Apply rate limiting
      rateLimitV1(req, res, next);
    });
    \`\`\`
    
    **7. Complete Shutdown** (Month 6)
    
    **Return 410 Gone**:
    \`\`\`javascript
    router.use((req, res) => {
      res.status(410).json({
        error: 'API v1 has been permanently removed',
        message: 'Please use API v2',
        documentation: 'https://api.example.com/docs/v2',
        migrationGuide: 'https://api.example.com/docs/v2-migration',
        support: 'support@example.com'
      });
    });
    \`\`\`
    
    **Remove Code** (30 days after shutdown):
    \`\`\`bash
    # After confirming zero V1 traffic for 30 days
    git rm -r routes/v1/
    git commit -m "Remove deprecated API v1 code"
    \`\`\`
    
    **8. Testing Strategy**
    
    **Integration Tests**:
    \`\`\`javascript
    describe('API Versioning', () => {
      describe('V1 (deprecated)', () => {
        it('should return v1 format', async () => {
          const res = await request(app).get('/api/v1/users/123');
          
          expect(res.body).toEqual({
            id: 123,
            name: 'John Doe'
          });
          
          expect(res.headers['deprecation']).toBe('true');
          expect(res.headers['sunset']).toBeDefined();
        });
      });
      
      describe('V2', () => {
        it('should return v2 format', async () => {
          const res = await request(app).get('/api/v2/users/123');
          
          expect(res.body).toEqual({
            id: 123,
            firstName: 'John',
            lastName: 'Doe'
          });
          
          expect(res.headers['x-api-version']).toBe('2.0.0');
        });
      });
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **6-month timeline** gives clients adequate time to migrate
    2. **URL path versioning** (/api/v1, /api/v2) is clearest for public APIs
    3. **Shared data layer** with toV1()/toV2() methods maintains single codebase
    4. **Deprecation headers** (Sunset, Link) inform clients programmatically
    5. **Monitor usage** by client to identify who needs to migrate
    6. **Email communication** at announcement, 3 months, 1 month before shutdown
    7. **Rate limiting** in Month 5 encourages migration (500 req/day)
    8. **Whitelist** for critical clients needing extensions
    9. **410 Gone** after Month 6 signals permanent removal
    10. **Remove code** 30 days after confirming zero traffic`,
          keyPoints: [
            '6-month migration timeline: Announce → Monitor → Rate limit → Whitelist → Shutdown',
            'URL path versioning (/api/v1, /api/v2) most common for public APIs',
            'Share data layer with toV1()/toV2() transformation methods to maintain single codebase',
            'Use Deprecation and Sunset HTTP headers to programmatically inform clients',
            'Monitor API usage by client/version to identify who needs to migrate',
            'Rate limit deprecated version (Month 5) to encourage migration, whitelist critical clients',
          ],
        },
        {
          id: 'api-versioning-disc-2',
          question:
            'Compare URL path versioning (/api/v2/users), header versioning (Accept: application/vnd.myapi.v2+json), and query parameter versioning (/api/users?version=2) for REST APIs. For each approach, discuss pros/cons, caching implications, and when you would choose each. Provide implementation examples.',
          sampleAnswer: `**Comprehensive Comparison of API Versioning Strategies**

---

## **1. URL Path Versioning** ⭐ (Most Common)

**Format**: \`/api/v{version}/resource\`

**Example**:
\`\`\`
GET /api/v1/users/123
GET /api/v2/users/123
\`\`\`

**Implementation**:

\`\`\`typescript
// Express routing
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// v1Router
const v1Router = express.Router();
v1Router.get('/users/:id', (req, res) => {
  const user = await db.users.findById(req.params.id);
  
  // V1 format: single "name" field
  res.json({
    id: user.id,
    name: \`\${user.firstName} \${user.lastName}\`,
    email: user.email
  });
});

// v2Router
const v2Router = express.Router();
v2Router.get('/users/:id', (req, res) => {
  const user = await db.users.findById(req.params.id);
  
  // V2 format: separate firstName/lastName
  res.json({
    id: user.id,
    firstName: user.firstName,
    lastName: user.lastName,
    email: user.email
  });
});
\`\`\`

### **Pros**:

✅ **Clear and explicit** - version immediately visible in URL
✅ **Easy to test** - can use curl, Postman easily
✅ **Cacheable** - different URLs = different cache entries
✅ **Browser friendly** - can bookmark different versions
✅ **Simple routing** - standard HTTP routing works
✅ **API gateway compatible** - easy to route by path
✅ **No special client code** - just change URL

### **Cons**:

❌ **URL pollution** - version in every endpoint
❌ **Breaking changes only** - can't use for minor changes
❌ **More boilerplate** - need separate routers per version

### **Caching**:

**Excellent** - Each version is a unique URL

\`\`\`
Cache-Control: public, max-age=3600
/api/v1/users/123  →  Cached separately
/api/v2/users/123  →  Cached separately
\`\`\`

CDNs and browser caches work perfectly.

### **When to Use**:

- **Public APIs** (most common choice)
- **RESTful services**
- **Long version lifetimes** (v1, v2, v3...)
- **Breaking changes**

---

## **2. Header Versioning** (Accept Header)

**Format**: \`Accept: application/vnd.myapi.v{version}+json\`

**Example**:
\`\`\`
GET /api/users/123
Accept: application/vnd.myapi.v1+json

GET /api/users/123
Accept: application/vnd.myapi.v2+json
\`\`\`

**Implementation**:

\`\`\`typescript
app.get('/api/users/:id', (req, res) => {
  const acceptHeader = req.get('Accept') || 'application/vnd.myapi.v1+json';
  const versionMatch = acceptHeader.match(/v(\\d+)/);
  const version = versionMatch ? parseInt(versionMatch[1]) : 1;
  
  const user = await db.users.findById(req.params.id);
  
  let response;
  switch (version) {
    case 1:
      response = {
        id: user.id,
        name: \`\${user.firstName} \${user.lastName}\`,
        email: user.email
      };
      break;
    
    case 2:
      response = {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email
      };
      break;
    
    default:
      return res.status(400).json({ error: 'Unsupported API version' });
  }
  
  res.setHeader('Content-Type', \`application/vnd.myapi.v\${version}+json\`);
  res.json(response);
});
\`\`\`

### **Pros**:

✅ **RESTful** - URLs represent resources, not versions
✅ **Clean URLs** - no version pollution in path
✅ **Content negotiation** - follows HTTP standards
✅ **Flexible** - can version by media type

### **Cons**:

❌ **Hard to test** - can't use browser directly
❌ **Caching complexity** - same URL, different content
❌ **Not obvious** - version hidden in headers
❌ **Client complexity** - must set headers correctly
❌ **Debugging harder** - need to inspect headers
❌ **CDN complexity** - requires Vary: Accept header

### **Caching**:

**Complex** - Requires \`Vary: Accept\` header

\`\`\`typescript
res.setHeader('Cache-Control', 'public, max-age=3600');
res.setHeader('Vary', 'Accept');
// CDN must cache separately based on Accept header
\`\`\`

**Problem**: Not all CDNs handle \`Vary\` correctly.

**CloudFront Fix**:
\`\`\`javascript
// Whitelist Accept header
const cloudFrontConfig = {
  headers: {
    whitelist: ['Accept']
  }
};
\`\`\`

### **When to Use**:

- **Internal APIs** (between backend services)
- **Following strict REST principles**
- **Academic/research APIs**
- **Hypermedia APIs** (HATEOAS)

---

## **3. Query Parameter Versioning**

**Format**: \`/api/resource?version=2\` or \`/api/resource?v=2\`

**Example**:
\`\`\`
GET /api/users/123?version=1
GET /api/users/123?version=2
\`\`\`

**Implementation**:

\`\`\`typescript
app.get('/api/users/:id', (req, res) => {
  const version = parseInt(req.query.version || '1');
  const user = await db.users.findById(req.params.id);
  
  let response;
  if (version === 1) {
    response = {
      id: user.id,
      name: \`\${user.firstName} \${user.lastName}\`,
      email: user.email
    };
  } else if (version === 2) {
    response = {
      id: user.id,
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email
    };
  } else {
    return res.status(400).json({ error: 'Unsupported version' });
  }
  
  res.json(response);
});
\`\`\`

### **Pros**:

✅ **Easy to test** - add ?version=2 to URL
✅ **Optional versioning** - default to latest if omitted
✅ **Browser friendly** - can bookmark
✅ **Simple client** - just change query param

### **Cons**:

❌ **Not RESTful** - query params shouldn't change resource representation
❌ **URL pollution** - messy with many params
❌ **Caching ambiguity** - \`/users/123\` and \`/users/123?version=1\` same resource?
❌ **Optional feels wrong** - version should be mandatory
❌ **Analytics harder** - query params often stripped
❌ **Routing complexity** - can't route by query param easily

### **Caching**:

**Unclear** - Do these cache separately?

\`\`\`
/api/users/123           (default v1?)
/api/users/123?version=1
/api/users/123?version=2
\`\`\`

**Solution**: Include version in Vary header or force version parameter

\`\`\`typescript
res.setHeader('Cache-Control', 'public, max-age=3600');
res.setHeader('Vary', 'version');  // Non-standard
\`\`\`

### **When to Use**:

- **Internal tools** where RESTfulness doesn't matter
- **Gradual rollouts** (e.g., A/B testing)
- **Optional features** (not core versioning)

---

## **4. Custom Header Versioning**

**Format**: \`X-API-Version: 2\` or \`API-Version: 2\`

**Example**:
\`\`\`
GET /api/users/123
X-API-Version: 2
\`\`\`

Similar to Accept header versioning but simpler.

### **Pros**:

✅ **Clean URLs**
✅ **Explicit versioning**
✅ **Simple header**

### **Cons**:

❌ **Not standard** (Accept is HTTP standard)
❌ **Caching complexity** (requires \`Vary: X-API-Version\`)
❌ **Hard to test**
❌ **Client complexity**

---

## **Comparison Table**

| Aspect | URL Path | Header (Accept) | Query Param | Custom Header |
|--------|----------|-----------------|-------------|---------------|
| **RESTful** | Debatable | ✅ Yes | ❌ No | Debatable |
| **Cacheable** | ✅ Easy | ⚠️ Complex | ⚠️ Unclear | ⚠️ Complex |
| **Testability** | ✅ Easy | ❌ Hard | ✅ Easy | ❌ Hard |
| **Visibility** | ✅ Obvious | ❌ Hidden | ✅ Visible | ❌ Hidden |
| **Client Simple** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **CDN Support** | ✅ Perfect | ⚠️ Needs Vary | ⚠️ Unclear | ⚠️ Needs Vary |
| **API Gateway** | ✅ Easy routing | ⚠️ Harder | ⚠️ Harder | ⚠️ Harder |
| **URL Pollution** | ❌ Yes | ✅ No | ❌ Yes | ✅ No |

---

## **Real-World Examples**

**URL Path** (Most Popular):
- **Stripe**: \`https://api.stripe.com/v1/charges\`
- **Twitter**: \`https://api.twitter.com/2/tweets\`
- **GitHub**: \`https://api.github.com/repos\` (path-based)
- **Twilio**: \`https://api.twilio.com/2010-04-01/Accounts\`

**Header Versioning**:
- **Azure**: \`api-version\` in query or header
- **Some GitHub APIs**: \`Accept: application/vnd.github.v3+json\`

**Query Parameter**:
- **Google Maps**: \`?v=3.exp\`
- **Azure (alternative)**: \`?api-version=2021-04-01\`

---

## **My Recommendation**

**For Public REST APIs**: **URL Path Versioning** (/api/v2/)

**Reasons**:
1. **Simplicity wins** - easy for clients to implement
2. **Caching works** - CDNs, browsers work out-of-the-box
3. **Debugging easy** - logs show version immediately
4. **Industry standard** - Stripe, Twitter, GitHub all use it
5. **Backwards compatible** - old clients keep working

**For Internal APIs**: **Header Versioning** (Accept or Custom)

**Reasons**:
1. **Cleaner URLs** - better REST semantics
2. **Service mesh friendly** - headers easier to route
3. **No URL pollution** - single endpoint
4. **Flexible** - can version independently by resource

**Avoid**: **Query Parameters** for core versioning
- Use query params for optional features, not core versioning
- Caching and semantics are unclear

---

## **Implementation Best Practices**

**1. Version Number Format**:

\`\`\`
Good:  /api/v1/, /api/v2/ (simple integers)
Okay:  /api/v1.2/, /api/v2.0/ (semantic versioning)
Bad:   /api/2023-01-15/ (dates)
\`\`\`

**2. Default Version**:

\`\`\`typescript
// Always explicit, never default to latest
app.get('/api/users/:id', (req, res) => {
  return res.status(400).json({
    error: 'API version required',
    hint: 'Use /api/v2/users/:id'
  });
});
\`\`\`

**3. Version in Response**:

\`\`\`typescript
res.setHeader('X-API-Version', '2.0.0');
res.setHeader('X-API-Deprecated', 'false');
\`\`\`

**4. Documentation**:

\`\`\`markdown
# API Versioning

We use URL path versioning:
- Current: /api/v2/
- Deprecated: /api/v1/ (sunset: 2024-06-01)
- Legacy: /api/v0/ (removed)

Version format: /api/v{major}/
\`\`\`

**Key Takeaway**: **URL path versioning wins for public APIs** due to simplicity, caching, and industry adoption. Use header versioning for internal services where clean URLs and REST semantics matter more than ease of testing.`,
          keyPoints: [
            'URL path versioning (/api/v2/): Best for public APIs (easy testing, perfect caching, industry standard)',
            'Header versioning (Accept): RESTful and clean URLs, but complex caching and hard to test',
            'Query parameter: Easy to test but not RESTful, caching ambiguous, avoid for core versioning',
            'Caching: URL path works out-of-box; headers need Vary header (CDN complexity)',
            'For public APIs: Choose URL path (Stripe, GitHub, Twitter all use it)',
            'For internal APIs: Header versioning acceptable (cleaner URLs, service mesh friendly)',
          ],
        },
        {
          id: 'api-versioning-disc-3',
          question:
            'You need to make a breaking change to your API (changing response format) but 40% of clients are still on v1. Design a strategy to minimize disruption including: versioning approach, migration timeline, client tracking, backwards compatibility layers, and rollout plan. How would you handle clients that refuse to migrate?',
          sampleAnswer: `**Breaking Change Migration Strategy**

---

## **1. Situation Analysis**

**Breaking Change**: Response format modification

**V1 Response**:
\`\`\`json
{
  "user": {
    "id": 123,
    "name": "John Doe",
    "address": "123 Main St, City, 12345"
  }
}
\`\`\`

**V2 Response** (Breaking):
\`\`\`json
{
  "id": 123,
  "firstName": "John",
  "lastName": "Doe",
  "address": {
    "street": "123 Main St",
    "city": "City",
    "zipCode": "12345"
  }
}
\`\`\`

**Current State**:
- 60% clients on v1
- 40% clients on v2
- Need to sunset v1

---

## **2. Versioning Approach**

**Use URL Path Versioning**: \`/api/v1/\` and \`/api/v2/\`

\`\`\`typescript
// Shared data layer
class UserRepository {
  async getUser(id: number): Promise<User> {
    return db.users.findById(id);
  }
}

// V1 Controller
app.get('/api/v1/users/:id', async (req, res) => {
  const user = await userRepo.getUser(req.params.id);
  
  // Transform to V1 format
  res.json({
    user: {
      id: user.id,
      name: \`\${user.firstName} \${user.lastName}\`,
      address: \`\${user.address.street}, \${user.address.city}, \${user.address.zipCode}\`
    }
  });
});

// V2 Controller
app.get('/api/v2/users/:id', async (req, res) => {
  const user = await userRepo.getUser(req.params.id);
  
  // Native V2 format
  res.json({
    id: user.id,
    firstName: user.firstName,
    lastName: user.lastName,
    address: {
      street: user.address.street,
      city: user.address.city,
      zipCode: user.address.zipCode
    }
  });
});
\`\`\`

---

## **3. Client Tracking System**

**Middleware to Track API Version Usage**:

\`\`\`typescript
interface ClientUsage {
  clientId: string;
  version: 'v1' | 'v2';
  endpoint: string;
  lastSeen: Date;
  requestCount: number;
}

// Middleware
app.use((req, res, next) => {
  const apiKey = req.headers['x-api-key'] as string;
  const version = req.path.startsWith('/api/v1/') ? 'v1' : 
                  req.path.startsWith('/api/v2/') ? 'v2' : 'unknown';
  
  // Track in Redis
  const key = \`api_usage:\${apiKey}:\${version}:\${getToday()}\`;
  redis.incr(key);
  redis.expire(key, 90 * 86400); // 90 days
  
  // Update last seen
  redis.hset(\`client:\${apiKey}\`, {
    lastVersion: version,
    lastSeen: Date.now(),
    endpoint: req.path
  });
  
  next();
});

// Dashboard query
async function getV1Clients(): Promise<ClientInfo[]> {
  const pattern = 'api_usage:*:v1:*';
  const keys = await redis.keys(pattern);
  
  const clients = await Promise.all(
    keys.map(async (key) => {
      const [_, apiKey, version, date] = key.split(':');
      const count = await redis.get(key);
      const info = await redis.hgetall(\`client:\${apiKey}\`);
      
      return {
        apiKey,
        requestsToday: parseInt(count || '0'),
        lastSeen: new Date(parseInt(info.lastSeen)),
        email: info.email,
        company: info.company
      };
    })
  );
  
  return clients.filter(c => c.requestsToday > 0);
}
\`\`\`

---

## **4. Migration Timeline (12 Months)**

### **Month 0-1: Preparation**

**Actions**:
1. **Analyze impact**:
   - Identify all affected endpoints
   - List breaking changes
   - Estimate migration effort

2. **Build v2**:
   - Implement new endpoints
   - Comprehensive testing
   - Performance benchmarks

3. **Create migration guide**:
   \`\`\`markdown
   # V1 → V2 Migration Guide
   
   ## Breaking Changes
   
   ### Response Format
   Before (v1):
   \`\`\`json
   { "user": { "name": "John Doe" } }
   \`\`\`
   
   After (v2):
   \`\`\`json
   { "firstName": "John", "lastName": "Doe" }
   \`\`\`
   
   ### Code Changes
   \`\`\`javascript
   // V1
   const name = response.user.name;
   
   // V2
   const name = \`\${response.firstName} \${response.lastName}\`;
   \`\`\`
   \`\`\`

### **Month 1-2: Soft Launch**

**Actions**:
1. **Release v2 (beta)**:
   - Opt-in only
   - No deprecation warnings yet
   - Monitor closely

2. **Contact top clients**:
   - Email 20 largest clients
   - Offer migration support
   - Schedule calls

3. **Track adoption**:
   - Dashboard showing v1 vs v2 usage
   - Identify early adopters

### **Month 2-3: Announce Deprecation**

**Email to ALL clients**:

\`\`\`
Subject: API v1 Deprecation Notice - Action Required

Hi {{name}},

We're deprecating API v1 on {{sunset_date}} (10 months from now).

## What's Changing
- Response format modernized (see docs)
- New features only in v2
- V1 will stop working on {{sunset_date}}

## Action Required
1. Review migration guide: {{migration_url}}
2. Update to /api/v2/ endpoints
3. Test in sandbox: {{sandbox_url}}

## Timeline
- Now: V2 available
- Month 6: V1 rate limited to 500 req/day
- Month 10: V1 returns 410 Gone

Questions? Reply to this email.
\`\`\`

**Add headers to v1**:

\`\`\`typescript
app.use('/api/v1/*', (req, res, next) => {
  res.setHeader('Deprecation', 'true');
  res.setHeader('Sunset', 'Sat, 01 Jan 2025 00:00:00 GMT');
  res.setHeader('Link', '</api/v2/docs>; rel="successor-version"');
  next();
});
\`\`\`

### **Month 3-6: Active Migration Period**

**Actions**:
1. **Weekly emails** to clients still on v1
2. **Office hours** for migration support
3. **Incentives**:
   - Free month for migrating early
   - Priority support
   - Beta access to new features

**Track progress**:
\`\`\`typescript
async function getMigrationProgress() {
  const v1Count = await redis.get('active_clients:v1');
  const v2Count = await redis.get('active_clients:v2');
  
  return {
    v1: parseInt(v1Count || '0'),
    v2: parseInt(v2Count || '0'),
    percentMigrated: (v2Count / (v1Count + v2Count)) * 100
  };
}
\`\`\`

### **Month 6-8: Enforcement**

**Aggressive Rate Limiting on v1**:

\`\`\`typescript
app.use('/api/v1/*', async (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  
  // Rate limit v1 to 500 requests/day
  const dailyLimit = 500;
  const count = await redis.incr(\`rate_limit:v1:\${apiKey}:\${getToday()}\`);
  
  if (count === 1) {
    await redis.expire(\`rate_limit:v1:\${apiKey}:\${getToday()}\`, 86400);
  }
  
  if (count > dailyLimit) {
    return res.status(429).json({
      error: 'Rate limit exceeded for deprecated v1 API',
      message: 'Please migrate to v2',
      migrationGuide: 'https://docs.example.com/migration',
      v2Endpoint: req.path.replace('/v1/', '/v2/')
    });
  }
  
  // Warning headers
  res.setHeader('X-RateLimit-Remaining', Math.max(0, dailyLimit - count));
  res.setHeader('X-API-Deprecated', 'true');
  
  next();
});
\`\`\`

**Email warnings**:
\`\`\`
Subject: URGENT: API v1 Rate Limited - Migrate Now

You've hit the v1 rate limit (500 req/day).

Migrate to v2 for unlimited access:
{{migration_guide_url}}

V1 shuts down in 2 months.
\`\`\`

### **Month 8-10: Final Push**

**Identify holdouts**:
\`\`\`typescript
const v1Clients = await getV1Clients();

for (const client of v1Clients) {
  // Direct phone calls for top 10 clients
  if (client.requestsPerDay > 10000) {
    console.log(\`CALL: \${client.company} - \${client.phone}\`);
  }
  
  // Final warning emails
  await sendEmail({
    to: client.email,
    subject: 'FINAL NOTICE: API v1 Shutdown in 30 Days',
    body: \`Your API keys will stop working on \${sunsetDate}\`
  });
}
\`\`\`

**Create whitelist for critical clients**:
\`\`\`typescript
const WHITELISTED_CLIENTS = [
  'api-key-enterprise-client-1', // Needs 60-day extension
  'api-key-government-client'     // Procurement delays
];

app.use('/api/v1/*', (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  
  if (WHITELISTED_CLIENTS.includes(apiKey)) {
    // Allow but log
    logger.warn(\`Whitelisted v1 access: \${apiKey}\`);
    return next();
  }
  
  // Enforce rate limit for others
  // ... (rate limiting code)
});
\`\`\`

### **Month 10: Sunset v1**

**Return 410 Gone**:

\`\`\`typescript
app.use('/api/v1/*', (req, res) => {
  const apiKey = req.headers['x-api-key'];
  
  // Check whitelist
  if (WHITELISTED_CLIENTS.includes(apiKey)) {
    // Allow for 30 more days
    if (Date.now() < WHITELIST_EXPIRY) {
      return next();
    }
  }
  
  res.status(410).json({
    error: 'Gone',
    message: 'API v1 has been permanently removed',
    migrationGuide: 'https://docs.example.com/migration',
    v2Endpoint: req.path.replace('/v1/', '/v2/'),
    supportEmail: 'support@example.com'
  });
});
\`\`\`

### **Month 11-12: Cleanup**

1. **Monitor for zero traffic**:
   \`\`\`typescript
   const v1Traffic = await redis.get(\`traffic:v1:\${getToday()}\`);
   if (v1Traffic === '0') {
     console.log('Safe to remove v1 code');
   }
   \`\`\`

2. **Remove v1 code** after 30 days of zero traffic
3. **Update documentation**
4. **Archive migration resources**

---

## **5. Handling Clients Who Refuse to Migrate**

### **Enterprise Clients (High Value)**

**Option 1: Paid Extension**:
\`\`\`
Offer 90-day extension for $5000/month
- Covers maintenance cost
- Time-limited (no indefinite extensions)
- Strict deadline enforced
\`\`\`

**Option 2: Managed Migration**:
\`\`\`
We migrate their code for them ($15,000 fixed price)
- Assigned engineer
- Complete in 2 weeks
- Guaranteed compatibility
\`\`\`

### **Small Clients (Low Value)**

**Hard Cutoff**:
- No extensions
- 410 Gone after sunset date
- "We understand this is inconvenient, but v1 has security vulnerabilities"

**Offer Alternatives**:
- Competitor APIs
- Open-source alternatives
- Refund if within contract period

### **Legal/Contractual Obligations**

**If SLA mentions "API stability"**:

1. **Argue v2 is not a breaking change to service**:
   - Same functionality, different format
   - 12-month notice exceeds industry standard

2. **Offer free migration**:
   - Absorb cost to maintain relationship
   - Document for future contracts

3. **Force majeure clause**:
   - Security vulnerabilities make v1 unmaintainable
   - Reasonable notice provided

---

## **6. Backwards Compatibility Layer** (Last Resort)

If critical clients can't migrate, build adapter:

\`\`\`typescript
// Adapter transforms v2 responses to v1 format
class V1toV2Adapter {
  transformResponse(v2Response: V2User): V1User {
    return {
      user: {
        id: v2Response.id,
        name: \`\${v2Response.firstName} \${v2Response.lastName}\`,
        address: \`\${v2Response.address.street}, \${v2Response.address.city}, \${v2Response.address.zipCode}\`
      }
    };
  }
}

// Keep v1 endpoints, but route to v2 internally
app.get('/api/v1/users/:id', async (req, res) => {
  // Call v2 internally
  const v2Response = await fetch(\`/api/v2/users/\${req.params.id}\`);
  const v2Data = await v2Response.json();
  
  // Transform to v1
  const adapter = new V1toV2Adapter();
  const v1Data = adapter.transformResponse(v2Data);
  
  res.json(v1Data);
});
\`\`\`

**Warning**: This adds technical debt. Time-limit this approach.

---

## **7. Success Metrics**

Track:
- % clients migrated: Target 95% by Month 10
- v1 traffic: Target <1% by shutdown
- Support tickets: Should decrease after Month 6
- Client churn: Minimize (<5%)

---

## **Key Takeaways**:

1. **12-month timeline** minimum for major breaking changes
2. **Track everything**: Know which clients use which versions
3. **Communicate early and often**: Email at 10, 6, 3, 1 months
4. **Incentivize migration**: Free months, priority support
5. **Enforce with rate limits**: Month 6 limits push stragglers
6. **Whitelist for critical clients**: Avoid business disruption
7. **410 Gone at sunset**: Clean break, no indefinite v1 support
8. **Hard line on refusers**: Paid extensions or hard cutoff
9. **Remove code after 30 days** of zero traffic
10. **Learn from it**: Update contracts to allow versioning`,
          keyPoints: [
            '12-month migration timeline: Announce early, rate limit mid-way, hard sunset at end',
            'Track client usage by version with middleware logging to Redis',
            'Communicate: Email at announcement, 6 months, 3 months, 1 month before shutdown',
            'Enforce migration: Rate limit v1 to 500 req/day in Month 6 to push stragglers',
            'Whitelist critical clients for temporary extensions, charge enterprise for long extensions',
            'Handle refusers: Paid extensions ($5k/month), managed migration, or hard cutoff',
            'Return 410 Gone after sunset date, remove code after 30 days of zero traffic',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'HTTP is stateless, request-response protocol; HTTPS adds encryption and authentication via TLS',
    'HTTP/2 multiplexes requests and compresses headers; HTTP/3 uses QUIC (UDP) for better performance',
    'TCP provides reliability, ordering, flow control; UDP provides speed with no guarantees',
    'WebSocket enables full-duplex real-time communication; use message broker (Redis pub/sub) for scaling',
    'DNS translates domains to IPs via hierarchical system (root → TLD → authoritative); caching with TTL',
    'RPC allows calling remote functions as if they were local; gRPC uses HTTP/2 + Protocol Buffers',
    'gRPC has 4 communication patterns: Unary, Server Streaming, Client Streaming, Bidirectional Streaming',
    'GraphQL allows clients to request exactly the data they need; solve N+1 problem with DataLoader',
    'Service Discovery enables dynamic service location; use client-side (Netflix Eureka) or server-side (Consul) discovery',
    'Service Mesh (Istio, Linkerd) provides observability, traffic management, and security for microservices',
    'MQTT ideal for IoT pub/sub with QoS levels; AMQP provides enterprise message queuing with RabbitMQ',
    'WebRTC enables P2P audio/video; use STUN/TURN for NAT traversal',
    'Token Bucket algorithm recommended for rate limiting (allows bursts, memory efficient)',
    'Rate limit BEFORE authentication to prevent DDoS; use Redis + Lua scripts for distributed rate limiting',
    'URL path versioning (/api/v2) most common for public REST APIs',
    'Deprecation strategy: Announce → Monitor → Whitelist → Shutdown (4-6 months minimum)',
    'GraphQL avoids versioning by deprecating fields instead of creating new versions',
    'Always set timeouts/deadlines on RPC calls; retry transient errors with exponential backoff',
  ],
  learningObjectives: [
    'Understand HTTP/HTTPS fundamentals and how TLS/SSL provides security',
    'Master HTTP methods, status codes, headers, and caching strategies',
    'Compare HTTP/1.1, HTTP/2, and HTTP/3 protocols and their performance characteristics',
    'Understand TCP vs UDP trade-offs and when to use each protocol',
    'Learn TCP reliability mechanisms: three-way handshake, flow control, congestion control',
    'Master WebSocket architecture for real-time bidirectional communication and scaling patterns',
    'Design DNS infrastructure for global distribution, failover, and DDoS protection',
    'Implement RPC systems using gRPC with streaming patterns and error handling',
    'Design GraphQL schemas and solve common performance issues (N+1, caching)',
    'Implement service discovery patterns for dynamic microservices architecture',
    'Choose appropriate network protocols (MQTT, AMQP, WebRTC) for different use cases',
    'Design distributed rate limiting systems using Redis and token bucket algorithm',
    'Implement API versioning strategies and manage deprecation lifecycle',
    'Build production-ready networking systems with monitoring, security, and fault tolerance',
  ],
};
