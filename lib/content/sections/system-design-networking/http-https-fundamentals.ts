/**
 * HTTP/HTTPS Fundamentals Section
 */

export const httphttpsfundamentalsSection = {
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
};
