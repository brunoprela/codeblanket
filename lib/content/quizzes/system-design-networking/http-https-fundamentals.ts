/**
 * Quiz questions for HTTP/HTTPS Fundamentals section
 */

export const httphttpsfundamentalsQuiz = [
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
   - Cost: Let\'s Encrypt is free, but need automation
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
     * Server\'s public key
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
- Attacker\'s certificate not signed by trusted CA
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
];
